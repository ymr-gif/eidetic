"""llm/catalog/cache.py — in-process snapshot refresh (HANDOFF Phase 3 + 7).

Unit tier — Redis and the DB are both faked (AsyncMock-style stand-ins), no
real network/DB I/O. Covers: the 15s guard, version-unchanged short-circuit,
version-changed Redis-hash reload, Redis-down DB fallback, USE_REDIS=false
straight-to-DB path, and the availability predicate (enabled AND live, or
enabled with <=1 recent failed probe AND last_live_at set — HANDOFF Phase 7:
a model that has never once answered a probe is never "available" just
because it hasn't failed twice yet).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest

import config
from llm.catalog import cache as catalog_cache

ENTRY = {
    "id": "z-ai/glm-5.3-flash", "label": "Glm 5.3 Flash", "status": "live",
    "enabled": True, "fail_count": 0, "price_in": None, "price_out": None,
    "context_window": None, "latency_ms": 100, "supports_tools": None,
    "reasoning": None, "request_extras": None, "min_max_tokens": None,
    "last_live_at": None,
}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    catalog_cache._reset_for_tests()
    monkeypatch.setattr(config, "LLM_BACKEND", "nim", raising=False)
    yield
    catalog_cache._reset_for_tests()


class _FakeRedis:
    def __init__(self, ver=None, entries=None, fail=False):
        self._ver = ver
        self._entries = entries or {}
        self._fail = fail

    async def get(self, key):
        if self._fail:
            raise ConnectionError("redis down")
        assert key == catalog_cache._REDIS_VER_KEY
        return self._ver

    async def hgetall(self, key):
        if self._fail:
            raise ConnectionError("redis down")
        assert key == catalog_cache._REDIS_ENTRIES_KEY
        return {k: json.dumps(v) for k, v in self._entries.items()}

    async def hset(self, key, mapping):
        self._entries = {k: json.loads(v) for k, v in mapping.items()}

    async def hkeys(self, key):
        return list(self._entries.keys())

    async def hdel(self, key, *fields):
        for f in fields:
            self._entries.pop(f, None)

    async def set(self, key, value, ex=None):
        self._ver = value

    async def delete(self, key):
        self._entries = {}


def _patch_redis(monkeypatch, fake):
    monkeypatch.setattr(config, "USE_REDIS", True, raising=False)
    import core.redis_client as rc
    monkeypatch.setattr(rc, "get_redis", lambda: fake)


class TestEnsureFresh:
    @pytest.mark.asyncio
    async def test_homeserver_mode_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_BACKEND", "homeserver", raising=False)
        catalog_cache._last_checked_at = 0.0
        await catalog_cache.ensure_fresh()
        assert catalog_cache.all_entries() == {}

    @pytest.mark.asyncio
    async def test_guard_skips_refresh_within_window(self, monkeypatch):
        import time
        catalog_cache._last_checked_at = time.monotonic()  # just checked
        fake = _FakeRedis(ver="v1", entries={"x": ENTRY})
        _patch_redis(monkeypatch, fake)
        await catalog_cache.ensure_fresh()
        assert catalog_cache.all_entries() == {}  # guard prevented the reload

    @pytest.mark.asyncio
    async def test_unchanged_version_is_a_no_op(self, monkeypatch):
        catalog_cache._last_checked_at = 0.0
        catalog_cache._set_snapshot_ver("v1")
        fake = _FakeRedis(ver="v1", entries={"x": ENTRY})
        _patch_redis(monkeypatch, fake)
        await catalog_cache.ensure_fresh()
        assert catalog_cache.all_entries() == {}  # never touched hgetall's result

    @pytest.mark.asyncio
    async def test_changed_version_reloads_from_entries_hash(self, monkeypatch):
        catalog_cache._last_checked_at = 0.0
        catalog_cache._set_snapshot_ver("v0")
        fake = _FakeRedis(ver="v1", entries={"z-ai/glm-5.3-flash": ENTRY})
        _patch_redis(monkeypatch, fake)
        await catalog_cache.ensure_fresh()
        assert catalog_cache.get_entry("z-ai/glm-5.3-flash") == ENTRY

    @pytest.mark.asyncio
    async def test_redis_down_falls_back_to_db(self, monkeypatch):
        catalog_cache._last_checked_at = 0.0
        fake = _FakeRedis(fail=True)
        _patch_redis(monkeypatch, fake)

        called = {}

        async def _fake_reload():
            called["hit"] = True
            catalog_cache._replace_snapshot({"db-model": ENTRY})

        monkeypatch.setattr(catalog_cache, "_reload_from_db", _fake_reload)
        await catalog_cache.ensure_fresh()
        assert called.get("hit") is True
        assert catalog_cache.get_entry("db-model") == ENTRY

    @pytest.mark.asyncio
    async def test_use_redis_false_goes_straight_to_db(self, monkeypatch):
        monkeypatch.setattr(config, "USE_REDIS", False, raising=False)
        catalog_cache._last_checked_at = 0.0

        async def _fake_reload():
            catalog_cache._replace_snapshot({"db-model": ENTRY})

        monkeypatch.setattr(catalog_cache, "_reload_from_db", _fake_reload)
        await catalog_cache.ensure_fresh()
        assert catalog_cache.get_entry("db-model") == ENTRY


class TestPublish:
    @pytest.mark.asyncio
    async def test_publish_with_explicit_entries_updates_snapshot_and_redis(self, monkeypatch):
        fake = _FakeRedis()
        _patch_redis(monkeypatch, fake)
        await catalog_cache.publish({"m1": ENTRY})
        assert catalog_cache.get_entry("m1") == ENTRY
        assert fake._entries == {"m1": ENTRY}
        assert fake._ver is not None

    @pytest.mark.asyncio
    async def test_publish_without_redis_still_updates_local_snapshot(self, monkeypatch):
        monkeypatch.setattr(config, "USE_REDIS", False, raising=False)
        await catalog_cache.publish({"m1": ENTRY})
        assert catalog_cache.get_entry("m1") == ENTRY


class TestAvailability:
    def _seed(self, entry):
        catalog_cache._replace_snapshot({entry["id"]: entry})

    def test_enabled_live_is_available(self):
        self._seed({**ENTRY, "enabled": True, "status": "live"})
        assert catalog_cache.is_available(ENTRY["id"]) is True

    def test_disabled_is_never_available(self):
        self._seed({**ENTRY, "enabled": False, "status": "live"})
        assert catalog_cache.is_available(ENTRY["id"]) is False

    def test_not_found_is_never_available_even_if_enabled(self):
        self._seed({**ENTRY, "enabled": True, "status": "not_found"})
        assert catalog_cache.is_available(ENTRY["id"]) is False

    def test_one_failed_probe_tolerated_when_previously_live(self):
        self._seed({
            **ENTRY, "enabled": True, "status": "error", "fail_count": 1,
            "last_live_at": "2026-09-19T00:00:00+00:00",
        })
        assert catalog_cache.is_available(ENTRY["id"]) is True

    def test_two_failed_probes_not_available_even_if_previously_live(self):
        self._seed({
            **ENTRY, "enabled": True, "status": "error", "fail_count": 2,
            "last_live_at": "2026-09-19T00:00:00+00:00",
        })
        assert catalog_cache.is_available(ENTRY["id"]) is False

    def test_transient_status_never_live_is_not_available(self):
        """HANDOFF Phase 7 repro: enabled, status=timeout, fail_count=1, but
        last_live_at is null — this model has NEVER answered a probe. Must
        not be tolerated as available just because it hasn't failed twice."""
        self._seed({
            **ENTRY, "enabled": True, "status": "timeout", "fail_count": 1,
            "last_live_at": None,
        })
        assert catalog_cache.is_available(ENTRY["id"]) is False

    def test_transient_status_missing_last_live_at_key_is_not_available(self):
        """A pre-051 snapshot entry with no last_live_at key at all (not just
        None) must fail the same way — entry.get() default, not a KeyError."""
        entry = {**ENTRY, "enabled": True, "status": "error", "fail_count": 1}
        entry.pop("last_live_at", None)
        self._seed(entry)
        assert catalog_cache.is_available(ENTRY["id"]) is False

    def test_missing_entry_is_not_available(self):
        assert catalog_cache.is_available("nope") is False

    def test_available_ids_filters_correctly(self):
        catalog_cache._replace_snapshot({
            "a": {**ENTRY, "id": "a", "enabled": True, "status": "live"},
            "b": {**ENTRY, "id": "b", "enabled": False, "status": "live"},
            "c": {**ENTRY, "id": "c", "enabled": True, "status": "gone"},
        })
        assert catalog_cache.available_ids() == ["a"]
