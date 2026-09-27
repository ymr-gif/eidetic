"""llm/catalog/role_state.py — in-process snapshot of model_role_override
rows (HANDOFF Phase A). Mirrors tests/test_catalog_cache.py's own structure
exactly — same 15s guard / version-check / Redis-down-falls-back-to-DB shape,
applied to the role-override table instead of the model catalog.
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
from llm.catalog import role_state

ENTRY = {
    "role": "llama", "model_id": "vendor/candidate", "pinned": False,
    "promoted_at": "2026-09-26T00:00:00+00:00", "reason": "auto",
    "base_model_id": "vendor/base",
}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    role_state._reset_for_tests()
    monkeypatch.setattr(config, "LLM_BACKEND", "nim", raising=False)
    yield
    role_state._reset_for_tests()


class _FakeRedis:
    def __init__(self, ver=None, entries=None, fail=False):
        self._ver = ver
        self._entries = entries or {}
        self._fail = fail

    async def get(self, key):
        if self._fail:
            raise ConnectionError("redis down")
        assert key == role_state._REDIS_VER_KEY
        return self._ver

    async def hgetall(self, key):
        if self._fail:
            raise ConnectionError("redis down")
        assert key == role_state._REDIS_ENTRIES_KEY
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
        role_state._last_checked_at = 0.0
        await role_state.ensure_fresh()
        assert role_state.all_overrides() == {}

    @pytest.mark.asyncio
    async def test_guard_skips_refresh_within_window(self, monkeypatch):
        import time
        role_state._last_checked_at = time.monotonic()
        fake = _FakeRedis(ver="v1", entries={"llama": ENTRY})
        _patch_redis(monkeypatch, fake)
        await role_state.ensure_fresh()
        assert role_state.all_overrides() == {}

    @pytest.mark.asyncio
    async def test_unchanged_version_is_a_no_op(self, monkeypatch):
        role_state._last_checked_at = 0.0
        role_state._set_snapshot_ver("v1")
        fake = _FakeRedis(ver="v1", entries={"llama": ENTRY})
        _patch_redis(monkeypatch, fake)
        await role_state.ensure_fresh()
        assert role_state.all_overrides() == {}

    @pytest.mark.asyncio
    async def test_changed_version_reloads_from_entries_hash(self, monkeypatch):
        role_state._last_checked_at = 0.0
        role_state._set_snapshot_ver("v0")
        fake = _FakeRedis(ver="v1", entries={"llama": ENTRY})
        _patch_redis(monkeypatch, fake)
        await role_state.ensure_fresh()
        assert role_state.get("llama") == ENTRY

    @pytest.mark.asyncio
    async def test_redis_down_falls_back_to_db(self, monkeypatch):
        role_state._last_checked_at = 0.0
        fake = _FakeRedis(fail=True)
        _patch_redis(monkeypatch, fake)

        called = {}

        async def _fake_reload():
            called["hit"] = True
            role_state._replace_snapshot({"llama": ENTRY})

        monkeypatch.setattr(role_state, "_reload_from_db", _fake_reload)
        await role_state.ensure_fresh()
        assert called.get("hit") is True
        assert role_state.get("llama") == ENTRY

    @pytest.mark.asyncio
    async def test_use_redis_false_goes_straight_to_db(self, monkeypatch):
        monkeypatch.setattr(config, "USE_REDIS", False, raising=False)
        role_state._last_checked_at = 0.0

        async def _fake_reload():
            role_state._replace_snapshot({"llama": ENTRY})

        monkeypatch.setattr(role_state, "_reload_from_db", _fake_reload)
        await role_state.ensure_fresh()
        assert role_state.get("llama") == ENTRY


class TestPublish:
    @pytest.mark.asyncio
    async def test_publish_with_explicit_entries_updates_snapshot_and_redis(self, monkeypatch):
        fake = _FakeRedis()
        _patch_redis(monkeypatch, fake)
        await role_state.publish({"llama": ENTRY})
        assert role_state.get("llama") == ENTRY
        assert fake._entries == {"llama": ENTRY}
        assert fake._ver is not None

    @pytest.mark.asyncio
    async def test_publish_without_redis_still_updates_local_snapshot(self, monkeypatch):
        monkeypatch.setattr(config, "USE_REDIS", False, raising=False)
        await role_state.publish({"llama": ENTRY})
        assert role_state.get("llama") == ENTRY


class TestReaders:
    def test_get_missing_role_is_none(self):
        assert role_state.get("llama") is None

    def test_all_overrides_returns_a_copy(self):
        role_state._replace_snapshot({"llama": ENTRY})
        out = role_state.all_overrides()
        out["llama"] = "mutated"
        assert role_state.get("llama") == ENTRY  # the real snapshot is untouched
