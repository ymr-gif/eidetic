"""llm/catalog/promotion.py — the auto-promotion state machine (HANDOFF
Phase A). Unit tier — no real DB/Redis/NIM.

`role_store` (DB persistence) is monkeypatched to a small in-memory fake
(mirrors the `_wire_store` pattern in tests/test_catalog_scanner.py); the
circuit breaker and role_state's in-process snapshot are the REAL modules,
exercised directly (their own dedicated test files cover their mechanics in
isolation — this file is about the STATE MACHINE built on top of them).
"""
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest

import config
from llm import circuit_breaker as cb
from llm.catalog import cache as catalog_cache
from llm.catalog import promotion
from llm.catalog import role_state

LLAMA     = "vendor/llama-base"
CODER     = "vendor/coder-base"
REASONING = "vendor/reasoning-base"
EMBEDDER  = "vendor/embedder"


class _FakeAuditResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeDB:
    """Stands in for `AsyncSessionLocal()` — `_write_system_audit`'s admin-id
    lookup is the only real `.execute()` call promotion.py makes on this
    session; everything else goes through the monkeypatched role_store."""
    def __init__(self, admin_id: int | None = 1):
        self.admin_id = admin_id
        self.added: list = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def commit(self):
        self.committed = True

    def add(self, obj):
        self.added.append(obj)

    async def execute(self, stmt):
        return _FakeAuditResult(self.admin_id)


class _RoleStoreFake:
    def __init__(self):
        self.overrides: dict[str, SimpleNamespace] = {}
        self.candidates: list = []
        self.set_calls: list = []
        self.candidate_calls: list = []
        self.cleared: list = []
        # On-demand-refresh fakes (root follow-up) — default empty so every
        # pre-existing "no candidate" test keeps behaving exactly as before
        # (an empty refresh set means _refresh_stale_candidates no-ops).
        self.refresh_candidates: list = []
        self.refresh_candidate_calls: list = []
        # Exclusion-reason diagnostics fakes (root observability follow-up).
        self.diagnostics: list = []
        self.diagnose_calls: list = []

    async def get_role_override(self, db, role):
        return self.overrides.get(role)

    async def list_role_overrides(self, db):
        return list(self.overrides.values())

    async def set_role_override(self, db, role, *, model_id, pinned, reason, base_model_id, promoted_at=None):
        row = SimpleNamespace(
            role=role, model_id=model_id, pinned=pinned, reason=reason,
            base_model_id=base_model_id, promoted_at=promoted_at or datetime.now(timezone.utc),
        )
        self.overrides[role] = row
        self.set_calls.append(row)
        return row

    async def clear_role_override(self, db, role):
        existed = role in self.overrides
        self.overrides.pop(role, None)
        self.cleared.append(role)
        return existed

    async def list_promotion_candidates(self, db, *, exclude_ids, max_ttfb_ms, max_stale_min):
        self.candidate_calls.append({"exclude_ids": set(exclude_ids), "max_ttfb_ms": max_ttfb_ms, "max_stale_min": max_stale_min})
        return [c for c in self.candidates if c.id not in exclude_ids]

    async def list_refresh_candidates(self, db, *, exclude_ids, limit):
        self.refresh_candidate_calls.append({"exclude_ids": set(exclude_ids), "limit": limit})
        return [c for c in self.refresh_candidates if c.id not in exclude_ids][:limit]

    async def diagnose_candidates(self, db, *, exclude_ids, max_ttfb_ms, max_stale_min, limit=5):
        self.diagnose_calls.append({"exclude_ids": set(exclude_ids), "max_ttfb_ms": max_ttfb_ms,
                                     "max_stale_min": max_stale_min, "limit": limit})
        return self.diagnostics


def _candidate(id_, ttfb_ms=100):
    return SimpleNamespace(id=id_, ttfb_ms=ttfb_ms)


def _seed_override(fake_store, role, *, model_id, pinned, reason="auto", base_model_id=None, promoted_at=None):
    base = base_model_id or config.MODELS[role]
    row = SimpleNamespace(role=role, model_id=model_id, pinned=pinned, reason=reason,
                          base_model_id=base, promoted_at=promoted_at or datetime.now(timezone.utc))
    fake_store.overrides[role] = row
    role_state._replace_snapshot({
        **role_state._snapshot,
        role: {"role": role, "model_id": model_id, "pinned": pinned, "reason": reason,
               "base_model_id": base, "promoted_at": row.promoted_at.isoformat()},
    })
    return row


async def _open_breaker(model_id: str) -> None:
    for _ in range(5):
        await cb.record_failure(model_id)


def _backdate_unhealthy(model_id: str, seconds_ago: float) -> None:
    cb._unhealthy_since[model_id] = time.time() - seconds_ago


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(config, "MODELS", {"llama": LLAMA, "coder": CODER, "reasoning": REASONING}, raising=False)
    monkeypatch.setattr(config, "FALLBACK_ORDER", ["reasoning", "coder", "llama"], raising=False)
    monkeypatch.setattr(config, "MODEL_AUTO_PROMOTE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LLM_BACKEND", "nim", raising=False)
    monkeypatch.setattr(config, "MODEL_EMBEDDING", EMBEDDER, raising=False)
    monkeypatch.setattr(config, "AUTO_PROMOTE_MIN_DOWN_SEC", 120, raising=False)
    monkeypatch.setattr(config, "AUTO_PROMOTE_COOLDOWN_MIN", 15, raising=False)
    monkeypatch.setattr(config, "AUTO_PROMOTE_RECOVER_MIN", 30, raising=False)
    monkeypatch.setattr(config, "AUTO_PROMOTE_MAX_TTFB_MS", 5000, raising=False)
    monkeypatch.setattr(config, "AUTO_PROMOTE_MAX_STALE_MIN", 30, raising=False)
    monkeypatch.setattr(config, "AUTO_PROMOTE_AUTO_REVERT", True, raising=False)
    # USE_REDIS must be pinned, not inherited: the repo's own .env (present on
    # a real dev machine) sets USE_REDIS=true, which would make every test in
    # this module attempt a REAL Redis connection via the refresh lock
    # (_acquire_refresh_lock/_release_refresh_lock) — same class of ambient-
    # environment leakage as the unit-tier env-independence fix in
    # tests/conftest.py. The two tests that specifically exercise the Redis
    # NX lock override this back to True locally with a fake redis client.
    monkeypatch.setattr(config, "USE_REDIS", False, raising=False)

    role_state._reset_for_tests()
    catalog_cache._reset_for_tests()
    cb._failures.clear(); cb._open.clear(); cb._open_time.clear(); cb._unhealthy_since.clear()
    promotion._recovering_since.clear()

    fake_store = _RoleStoreFake()
    monkeypatch.setattr(promotion, "role_store", fake_store, raising=False)
    monkeypatch.setattr(promotion.role_state, "ensure_fresh", AsyncMock(), raising=False)
    monkeypatch.setattr(promotion.role_state, "publish", AsyncMock(), raising=False)

    fake_db = _FakeDB()
    monkeypatch.setattr("core.db.AsyncSessionLocal", lambda: fake_db, raising=False)

    yield fake_store, fake_db

    role_state._reset_for_tests()
    catalog_cache._reset_for_tests()
    cb._failures.clear(); cb._open.clear(); cb._open_time.clear(); cb._unhealthy_since.clear()
    promotion._recovering_since.clear()


# ── effective_role_model (the hot-path function) ─────────────────────────────

class TestEffectiveRoleModel:
    def test_returns_base_when_disabled(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "MODEL_AUTO_PROMOTE_ENABLED", False, raising=False)
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/other", pinned=False)
        assert promotion.effective_role_model("llama") == LLAMA

    def test_returns_base_in_homeserver_mode(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "LLM_BACKEND", "homeserver", raising=False)
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/other", pinned=False)
        assert promotion.effective_role_model("llama") == LLAMA

    def test_returns_base_when_no_override(self, _reset):
        assert promotion.effective_role_model("llama") == LLAMA

    def test_returns_override_when_enabled_and_present(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/other", pinned=False)
        assert promotion.effective_role_model("llama") == "vendor/other"


# ── consider_promotion ────────────────────────────────────────────────────────

class TestConsiderPromotionFlagOff:
    @pytest.mark.asyncio
    async def test_disabled_flag_is_a_no_op(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "MODEL_AUTO_PROMOTE_ENABLED", False, raising=False)
        fake_store, fake_db = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 999)
        fake_store.candidates = [_candidate("vendor/fast", ttfb_ms=100)]
        result = await promotion.consider_promotion("llama")
        assert result is None
        assert fake_store.set_calls == []
        assert fake_db.added == []

    @pytest.mark.asyncio
    async def test_homeserver_mode_is_a_no_op(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "LLM_BACKEND", "homeserver", raising=False)
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 999)
        fake_store.candidates = [_candidate("vendor/fast", ttfb_ms=100)]
        result = await promotion.consider_promotion("llama")
        assert result is None
        assert fake_store.set_calls == []


class TestConsiderPromotionTrigger:
    @pytest.mark.asyncio
    async def test_no_op_when_breaker_not_open(self, _reset):
        fake_store, _ = _reset
        fake_store.candidates = [_candidate("vendor/fast")]
        result = await promotion.consider_promotion("llama")
        assert result is None
        assert fake_store.set_calls == []

    @pytest.mark.asyncio
    async def test_no_op_on_a_single_failure_not_on_a_single_503(self, _reset):
        fake_store, _ = _reset
        await cb.record_failure(LLAMA)   # 1 of 5 needed to even open the breaker
        fake_store.candidates = [_candidate("vendor/fast")]
        result = await promotion.consider_promotion("llama")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_op_when_open_but_not_down_long_enough(self, _reset):
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 10)   # well under AUTO_PROMOTE_MIN_DOWN_SEC=120
        fake_store.candidates = [_candidate("vendor/fast")]
        result = await promotion.consider_promotion("llama")
        assert result is None
        assert fake_store.set_calls == []

    @pytest.mark.asyncio
    async def test_promotes_when_open_and_down_past_min_down_sec(self, _reset):
        fake_store, fake_db = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = [_candidate("vendor/fast", ttfb_ms=100)]
        result = await promotion.consider_promotion("llama")
        assert result == {"role": "llama", "from": LLAMA, "to": "vendor/fast"}
        assert fake_store.set_calls[0].model_id == "vendor/fast"
        assert fake_store.set_calls[0].reason == "auto"
        assert fake_store.set_calls[0].pinned is False
        assert fake_db.committed is True
        assert any(getattr(o, "action", None) == "model_catalog.auto_promoted" for o in fake_db.added)


class TestConsiderPromotionCandidateFilter:
    @pytest.mark.asyncio
    async def test_no_candidate_does_nothing(self, _reset):
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        result = await promotion.consider_promotion("llama")
        assert result is None
        assert fake_store.set_calls == []

    @pytest.mark.asyncio
    async def test_lowest_ttfb_candidate_wins(self, _reset):
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        # role_store.list_promotion_candidates is contracted to return them
        # already ordered by ttfb_ms ascending — consider_promotion just
        # takes the first.
        fake_store.candidates = [_candidate("vendor/fastest", ttfb_ms=50), _candidate("vendor/slower", ttfb_ms=400)]
        result = await promotion.consider_promotion("llama")
        assert result["to"] == "vendor/fastest"

    @pytest.mark.asyncio
    async def test_excludes_the_failing_model_itself_the_embedder_and_other_roles(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "coder", model_id="vendor/coder-promoted", pinned=False)
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = [_candidate("vendor/fast")]
        await promotion.consider_promotion("llama")
        exclude_ids = fake_store.candidate_calls[-1]["exclude_ids"]
        assert LLAMA in exclude_ids               # the failing model itself
        assert EMBEDDER in exclude_ids            # never the embedder
        assert "vendor/coder-promoted" in exclude_ids  # coder's CURRENT effective model
        assert REASONING in exclude_ids           # reasoning's (unpromoted) base

    @pytest.mark.asyncio
    async def test_candidate_filter_args_forwarded(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "AUTO_PROMOTE_MAX_TTFB_MS", 3000, raising=False)
        monkeypatch.setattr(config, "AUTO_PROMOTE_MAX_STALE_MIN", 45, raising=False)
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = [_candidate("vendor/fast")]
        await promotion.consider_promotion("llama")
        call = fake_store.candidate_calls[-1]
        assert call["max_ttfb_ms"] == 3000
        assert call["max_stale_min"] == 45


def _refresh_row(id_, latency_ms=200, label="Some Model"):
    return SimpleNamespace(id=id_, latency_ms=latency_ms, label=label)


class TestOnDemandRefresh:
    """The root follow-up (2026-09-26): when list_promotion_candidates comes
    back empty, consider_promotion re-probes a small bounded set of
    already-enabled catalog rows (llm.catalog.promotion._refresh_stale_
    candidates) before giving up, so an incident isn't stuck waiting for the
    next 6h scan. Covers: refresh finds a usable candidate → promoted; the
    Redis lock held elsewhere → skipped; refresh finds nothing usable →
    no-op; AUTO_PROMOTE_REFRESH_MAX is respected."""

    def _wire_refresh(self, monkeypatch, *, ttfb_ms=150, ttfb_fail_reason=None, verify_results=None):
        """`verify_results`: dict[model_id] -> verify_model() return dict, or
        a single dict reused for every id when not a dict-of-dicts."""
        calls = {"ttfb": [], "verify": []}

        async def _fake_probe_ttfb(model_id, sem):
            calls["ttfb"].append(model_id)
            return {"ttfb_ms": ttfb_ms, "fail_reason": ttfb_fail_reason}

        async def _fake_verify_model(model_id, sem):
            calls["verify"].append(model_id)
            if verify_results and model_id in verify_results:
                return verify_results[model_id]
            return verify_results or {"tool_ok": True, "reasoning_leak": False, "error": None, "latency_ms": 90}

        monkeypatch.setattr(promotion, "probe_ttfb", _fake_probe_ttfb)
        monkeypatch.setattr(promotion, "verify_model", _fake_verify_model)
        return calls

    def _wire_upsert(self, monkeypatch, fake_store):
        """Fakes llm.catalog.store.upsert_scan_result — records the call and
        simulates the row now qualifying as a live candidate (mirrors what a
        real upsert + a subsequent list_promotion_candidates query would see,
        without needing a real DB)."""
        upserts = []

        async def _fake_upsert(db, model_id, *, status, http_status, latency_ms, label, seed_enabled,
                                reasoning=None, ttfb_ms=None, tool_ok=None, reasoning_leak=None,
                                ttfb_fail_reason=None, verified=False):
            upserts.append({"model_id": model_id, "status": status, "ttfb_ms": ttfb_ms,
                             "tool_ok": tool_ok, "reasoning_leak": reasoning_leak,
                             "ttfb_fail_reason": ttfb_fail_reason, "verified": verified})
            # Mirrors list_promotion_candidates's real WHERE clause: only a row
            # with BOTH a real ttfb_ms and tool_ok=True would ever qualify.
            if tool_ok and ttfb_ms is not None:
                fake_store.candidates.append(_candidate(model_id, ttfb_ms=ttfb_ms))

        monkeypatch.setattr(promotion.store, "upsert_scan_result", _fake_upsert)
        return upserts

    @pytest.mark.asyncio
    async def test_refresh_finds_a_candidate_and_promotes(self, monkeypatch, _reset):
        fake_store, fake_db = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = [_refresh_row("vendor/stale-but-ok")]
        self._wire_refresh(monkeypatch)
        upserts = self._wire_upsert(monkeypatch, fake_store)

        result = await promotion.consider_promotion("llama")

        assert result == {"role": "llama", "from": LLAMA, "to": "vendor/stale-but-ok"}
        assert upserts[0]["model_id"] == "vendor/stale-but-ok"
        assert upserts[0]["status"] == "live"
        assert upserts[0]["verified"] is True
        assert fake_db.committed is True
        # list_promotion_candidates was called twice — once empty, once after refresh
        assert len(fake_store.candidate_calls) == 2

    @pytest.mark.asyncio
    async def test_refresh_finds_nothing_usable_is_a_no_op(self, monkeypatch, _reset):
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = [_refresh_row("vendor/turns-out-broken")]
        self._wire_refresh(monkeypatch, verify_results={
            "vendor/turns-out-broken": {"tool_ok": False, "reasoning_leak": False, "error": "http_500", "latency_ms": None},
        })
        self._wire_upsert(monkeypatch, fake_store)

        result = await promotion.consider_promotion("llama")

        assert result is None
        assert fake_store.set_calls == []
        # a verify error means we never even upsert (never reconfirmed live) —
        # list_promotion_candidates is NOT re-queried since refresh reported
        # nothing usable.
        assert len(fake_store.candidate_calls) == 1

    @pytest.mark.asyncio
    async def test_root_live_stack_repro_reconfirmed_live_but_ttfb_reasoning_only_stays_unpromotable(
        self, monkeypatch, _reset,
    ):
        """The exact root live-stack finding (2026-09-27): muse-glimmer
        reconfirms live/tool_ok=true via the refresh, but its TTFB probe
        structurally fails (spent its whole budget on reasoning_content) —
        `upsert_scan_result` must still be called (this is NOT "never
        probed"), but with `ttfb_ms=None`, so the row is never appended as a
        usable candidate and the role stays on its dead base model."""
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = [_refresh_row("meta/muse-glimmer-30b")]
        self._wire_refresh(monkeypatch, ttfb_ms=None, ttfb_fail_reason="reasoning_only",
                            verify_results={"meta/muse-glimmer-30b": {
                                "tool_ok": True, "reasoning_leak": False, "error": None, "latency_ms": 90,
                            }})
        upserts = self._wire_upsert(monkeypatch, fake_store)

        result = await promotion.consider_promotion("llama")

        assert result is None  # still no promotion — the whole point of the bug report
        assert upserts[0]["model_id"] == "meta/muse-glimmer-30b"
        assert upserts[0]["ttfb_ms"] is None
        assert upserts[0]["ttfb_fail_reason"] == "reasoning_only"
        assert upserts[0]["tool_ok"] is True  # reconfirmed live — recorded, just not ttfb-eligible
        assert fake_store.set_calls == []  # never promoted to it
        assert fake_store.candidates == []  # the fake upsert only appends when ttfb_ms is not None too

    @pytest.mark.asyncio
    async def test_refresh_skipped_when_lock_held_elsewhere(self, monkeypatch, _reset):
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = [_refresh_row("vendor/would-have-worked")]
        self._wire_refresh(monkeypatch)
        self._wire_upsert(monkeypatch, fake_store)
        monkeypatch.setattr(promotion, "_acquire_refresh_lock", AsyncMock(return_value=False))

        result = await promotion.consider_promotion("llama")

        assert result is None
        assert fake_store.refresh_candidate_calls == []  # never even looked for refresh candidates
        assert len(fake_store.candidate_calls) == 1

    @pytest.mark.asyncio
    async def test_refresh_set_is_capped_at_auto_promote_refresh_max(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "AUTO_PROMOTE_REFRESH_MAX", 2, raising=False)
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = [_refresh_row("vendor/a"), _refresh_row("vendor/b"), _refresh_row("vendor/c")]
        self._wire_refresh(monkeypatch)
        self._wire_upsert(monkeypatch, fake_store)

        await promotion.consider_promotion("llama")

        assert fake_store.refresh_candidate_calls[-1]["limit"] == 2

    @pytest.mark.asyncio
    async def test_refresh_lock_uses_redis_nx_when_use_redis_true(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "USE_REDIS", True, raising=False)

        class _FakeRedis:
            def __init__(self):
                self.set_calls = []
                self.deleted = []

            async def set(self, key, value, nx=None, ex=None):
                self.set_calls.append((key, nx, ex))
                return True  # lock acquired

            async def delete(self, key):
                self.deleted.append(key)

        fake_redis = _FakeRedis()
        import core.redis_client as rc
        monkeypatch.setattr(rc, "get_redis", lambda: fake_redis)

        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = []  # nothing to refresh — just checking lock plumbing

        await promotion.consider_promotion("llama")

        # circuit_breaker.record_failure() also writes to Redis (cb:open:...)
        # once USE_REDIS=True — filter to just the refresh lock's own key.
        lock_calls = [c for c in fake_redis.set_calls if c[0] == promotion._REFRESH_LOCK_KEY]
        assert lock_calls == [(promotion._REFRESH_LOCK_KEY, True, promotion._REFRESH_LOCK_TTL)]
        assert promotion._REFRESH_LOCK_KEY in fake_redis.deleted

    @pytest.mark.asyncio
    async def test_refresh_lock_nx_failure_means_lock_held_elsewhere(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "USE_REDIS", True, raising=False)

        class _LockedRedis:
            async def set(self, key, value, nx=None, ex=None):
                return None  # NX set failed — someone else holds it

        import core.redis_client as rc
        monkeypatch.setattr(rc, "get_redis", lambda: _LockedRedis())

        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = [_refresh_row("vendor/would-have-worked")]
        self._wire_refresh(monkeypatch)
        self._wire_upsert(monkeypatch, fake_store)

        result = await promotion.consider_promotion("llama")

        assert result is None
        assert fake_store.refresh_candidate_calls == []

    @pytest.mark.asyncio
    async def test_no_refresh_attempted_when_a_fresh_candidate_already_exists(self, monkeypatch, _reset):
        """The refresh path only fires when list_promotion_candidates comes
        back empty — a normal promotion never touches it."""
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = [_candidate("vendor/already-fresh")]
        calls = self._wire_refresh(monkeypatch)
        self._wire_upsert(monkeypatch, fake_store)

        await promotion.consider_promotion("llama")

        assert fake_store.refresh_candidate_calls == []
        assert calls["ttfb"] == []
        assert calls["verify"] == []


class TestCandidateExclusionLogging:
    """Root observability follow-up (2026-09-27): "no eligible candidate"
    alone gave no reason — the root live-stack test could only find out WHY
    muse-glimmer didn't qualify by querying Postgres by hand. consider_
    promotion now logs a per-candidate exclusion reason on every "no
    candidate" outcome."""

    @pytest.mark.asyncio
    async def test_logs_one_line_per_diagnostic_on_no_candidate(self, monkeypatch, _reset, caplog):
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = []  # refresh finds nothing either
        fake_store.diagnostics = [
            ("vendor/a", "disabled"),
            ("vendor/b", "no_ttfb:reasoning_only"),
            ("vendor/c", "tool_ok_false"),
        ]

        with caplog.at_level("INFO", logger="catalog.promotion"):
            result = await promotion.consider_promotion("llama")

        assert result is None
        assert len(fake_store.diagnose_calls) == 1
        messages = [r.message for r in caplog.records]
        assert any("vendor/a" in m and "disabled" in m for m in messages)
        assert any("vendor/b" in m and "no_ttfb:reasoning_only" in m for m in messages)
        assert any("vendor/c" in m and "tool_ok_false" in m for m in messages)

    @pytest.mark.asyncio
    async def test_diagnostics_forwarded_with_the_same_filter_args(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "AUTO_PROMOTE_MAX_TTFB_MS", 3000, raising=False)
        monkeypatch.setattr(config, "AUTO_PROMOTE_MAX_STALE_MIN", 45, raising=False)
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = []

        await promotion.consider_promotion("llama")

        call = fake_store.diagnose_calls[-1]
        assert call["max_ttfb_ms"] == 3000
        assert call["max_stale_min"] == 45
        assert LLAMA in call["exclude_ids"]

    @pytest.mark.asyncio
    async def test_never_called_when_a_candidate_is_found(self, _reset):
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = [_candidate("vendor/fast")]

        result = await promotion.consider_promotion("llama")

        assert result is not None
        assert fake_store.diagnose_calls == []

    @pytest.mark.asyncio
    async def test_diagnostics_failure_never_blocks_the_no_candidate_outcome(self, monkeypatch, _reset):
        """Defense in depth: even if diagnose_candidates itself blows up,
        consider_promotion must still complete cleanly (never raise)."""
        fake_store, _ = _reset
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = []
        fake_store.refresh_candidates = []

        async def _boom(db, **kwargs):
            raise RuntimeError("diagnostics exploded")

        monkeypatch.setattr(fake_store, "diagnose_candidates", _boom)

        result = await promotion.consider_promotion("llama")

        assert result is None  # no exception propagated


class TestConsiderPromotionGuards:
    @pytest.mark.asyncio
    async def test_pinned_role_never_moves(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/pinned-choice", pinned=True, reason="manual")
        await _open_breaker("vendor/pinned-choice")
        _backdate_unhealthy("vendor/pinned-choice", 200)
        fake_store.candidates = [_candidate("vendor/fast")]
        result = await promotion.consider_promotion("llama")
        assert result is None
        assert fake_store.set_calls == []

    @pytest.mark.asyncio
    async def test_cooldown_blocks_a_second_promotion(self, _reset):
        fake_store, _ = _reset
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)  # < 15min cooldown
        _seed_override(fake_store, "llama", model_id="vendor/already-promoted", pinned=False,
                        promoted_at=recent)
        await _open_breaker("vendor/already-promoted")
        _backdate_unhealthy("vendor/already-promoted", 200)
        fake_store.candidates = [_candidate("vendor/fast")]
        result = await promotion.consider_promotion("llama")
        assert result is None
        assert fake_store.set_calls == []

    @pytest.mark.asyncio
    async def test_a_new_promotion_is_allowed_once_cooldown_has_elapsed(self, _reset):
        fake_store, _ = _reset
        old = datetime.now(timezone.utc) - timedelta(minutes=20)  # > 15min cooldown
        _seed_override(fake_store, "llama", model_id="vendor/already-promoted", pinned=False,
                        promoted_at=old)
        await _open_breaker("vendor/already-promoted")
        _backdate_unhealthy("vendor/already-promoted", 200)
        fake_store.candidates = [_candidate("vendor/fast")]
        result = await promotion.consider_promotion("llama")
        assert result is not None
        assert result["to"] == "vendor/fast"

    @pytest.mark.asyncio
    async def test_unknown_role_is_a_no_op(self, _reset):
        result = await promotion.consider_promotion("not-a-real-role")
        assert result is None


class TestCheckPromotionForFailedModel:
    @pytest.mark.asyncio
    async def test_triggers_for_every_role_currently_served_by_the_failed_model(self, _reset):
        """llama + coder sharing one id (the live 2026-09-25 topology) — a
        failure of that shared id must be able to promote EITHER role
        independently, not just whichever one the caller happened to name."""
        fake_store, _ = _reset
        import config as _config
        shared = LLAMA
        with_shared_coder = dict(_config.MODELS)
        with_shared_coder["coder"] = shared
        import pytest as _pytest  # local, avoid unused-import lint in some setups
        from unittest.mock import patch
        with patch.object(_config, "MODELS", with_shared_coder):
            await _open_breaker(shared)
            _backdate_unhealthy(shared, 200)
            fake_store.candidates = [_candidate("vendor/fast")]
            await promotion.check_promotion_for_failed_model(shared)
        assert {c.role for c in fake_store.set_calls} == {"llama", "coder"}


# ── consider_revert ───────────────────────────────────────────────────────────

class TestConsiderRevert:
    @pytest.mark.asyncio
    async def test_no_op_when_no_override(self, _reset):
        result = await promotion.consider_revert("llama")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_op_when_pinned(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=True, reason="manual")
        catalog_cache._replace_snapshot({LLAMA: {"id": LLAMA, "enabled": True, "status": "live",
                                                  "fail_count": 0, "last_live_at": "2026-09-26T00:00:00+00:00"}})
        result = await promotion.consider_revert("llama")
        assert result is None
        assert fake_store.cleared == []

    @pytest.mark.asyncio
    async def test_no_op_when_base_not_yet_available(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False)
        catalog_cache._replace_snapshot({})  # base LLAMA not in the catalog at all -> unavailable
        result = await promotion.consider_revert("llama")
        assert result is None
        assert fake_store.cleared == []

    @pytest.mark.asyncio
    async def test_recovering_clock_resets_if_base_goes_unavailable_again(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False)
        catalog_cache._replace_snapshot({LLAMA: {"id": LLAMA, "enabled": True, "status": "live",
                                                  "fail_count": 0, "last_live_at": "2026-09-26T00:00:00+00:00"}})
        await promotion.consider_revert("llama")               # starts the recovery clock
        assert "llama" in promotion._recovering_since
        catalog_cache._replace_snapshot({})                    # base blips unavailable again
        await promotion.consider_revert("llama")
        assert "llama" not in promotion._recovering_since       # clock reset, not just paused

    @pytest.mark.asyncio
    async def test_reverts_once_recover_window_elapses(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "AUTO_PROMOTE_RECOVER_MIN", 0, raising=False)  # revert on the very next tick
        fake_store, fake_db = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False)
        catalog_cache._replace_snapshot({LLAMA: {"id": LLAMA, "enabled": True, "status": "live",
                                                  "fail_count": 0, "last_live_at": "2026-09-26T00:00:00+00:00"}})
        result = await promotion.consider_revert("llama")
        assert result == {"role": "llama", "from": "vendor/promoted", "to": LLAMA}
        assert fake_store.cleared == ["llama"]
        assert fake_db.committed is True
        assert any(getattr(o, "action", None) == "model_catalog.auto_reverted" for o in fake_db.added)

    @pytest.mark.asyncio
    async def test_disabled_auto_revert_flag_is_a_no_op(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "AUTO_PROMOTE_AUTO_REVERT", False, raising=False)
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False)
        catalog_cache._replace_snapshot({LLAMA: {"id": LLAMA, "enabled": True, "status": "live",
                                                  "fail_count": 0, "last_live_at": "2026-09-26T00:00:00+00:00"}})
        result = await promotion.consider_revert("llama")
        assert result is None
        assert fake_store.cleared == []

    @pytest.mark.asyncio
    async def test_check_reverts_covers_every_role_and_never_raises_on_one_failure(self, monkeypatch, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False)

        async def _boom(role):
            raise RuntimeError("boom")

        monkeypatch.setattr(promotion, "consider_revert", _boom)
        # Must not raise even though consider_revert always blows up.
        await promotion.check_reverts()


class TestConsiderRevertBaseChangedReconciliation:
    """Root live-stack finding (2026-09-27): promotion carried coder through
    a dead deepseek-v4-flash-0731; the operator then repointed MODEL_CODER to
    a healthy id and restarted — the normal recovery path. Auto-revert never
    fired because `override.base_model_id` (frozen at promotion time, still
    the dead id) can never pass an availability check. consider_revert must
    reconcile against the CURRENT config.MODELS[role], not the stored one."""

    @pytest.mark.asyncio
    async def test_unpinned_override_released_immediately_when_base_changed(self, monkeypatch, _reset):
        """No recovery-window wait, no availability check on the new base —
        released on the very first tick after the config change, regardless
        of AUTO_PROMOTE_RECOVER_MIN (left at its fixture default, 30) and
        regardless of whether the new base is even in the catalog cache at
        all (catalog_cache._replace_snapshot is never called in this test)."""
        fake_store, fake_db = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False,
                        reason="auto", base_model_id="vendor/dead-old-base")
        # config.MODELS["llama"] (LLAMA) != the override's stored base_model_id
        # ("vendor/dead-old-base") — simulates the operator's .env fix + restart.

        result = await promotion.consider_revert("llama")

        assert result == {"role": "llama", "released": "vendor/promoted",
                           "old_base": "vendor/dead-old-base", "new_base": LLAMA}
        assert fake_store.cleared == ["llama"]
        assert fake_db.committed is True
        assert any(getattr(o, "action", None) == "model_catalog.override_released_base_changed"
                   for o in fake_db.added)
        # A DIFFERENT audit action than a normal recovery-based revert.
        assert not any(getattr(o, "action", None) == "model_catalog.auto_reverted" for o in fake_db.added)
        assert "llama" not in promotion._recovering_since

    @pytest.mark.asyncio
    async def test_pinned_override_survives_a_base_change(self, monkeypatch, _reset):
        """A manual pin is a deliberate human choice — it must NOT be
        auto-cleared just because .env changed; only another manual admin
        action (PATCH .../roles/{role}) touches it."""
        fake_store, fake_db = _reset
        _seed_override(fake_store, "llama", model_id="vendor/picked", pinned=True,
                        reason="manual", base_model_id="vendor/old-base")

        result = await promotion.consider_revert("llama")

        assert result is None
        assert fake_store.cleared == []
        assert fake_db.added == []
        assert "llama" not in promotion._recovering_since

    @pytest.mark.asyncio
    async def test_base_unchanged_still_uses_the_normal_recovery_window_path(self, monkeypatch, _reset):
        """Confirms the reconciliation branch and the normal-recovery branch
        are mutually exclusive — when config.MODELS[role] still matches the
        override's stored base, behavior is completely unchanged from before
        this fix: it needs the recovery window, not an immediate release."""
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False)  # base_model_id defaults to LLAMA
        catalog_cache._replace_snapshot({})  # base not yet available

        result = await promotion.consider_revert("llama")

        assert result is None
        assert fake_store.cleared == []  # NOT released — base_model_id still matches config

    @pytest.mark.asyncio
    async def test_a_released_stale_base_override_never_blocks_the_next_promotion(self, monkeypatch, _reset):
        """After consider_revert releases a stale-base override, the role
        must be free to be promoted again on its own merits — nothing left
        over from the old override (cooldown, pinned state) blocks it."""
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/old-promoted", pinned=False,
                        reason="auto", base_model_id="vendor/dead-old-base",
                        promoted_at=datetime.now(timezone.utc))  # "just now" — would still be within cooldown

        release_result = await promotion.consider_revert("llama")
        assert release_result["released"] == "vendor/old-promoted"
        assert "llama" not in fake_store.overrides  # actually cleared, not just marked

        # role_state.publish() is mocked to a no-op by the fixture (real DB/
        # Redis aren't in play here) — simulate what it would really do so
        # effective_role_model reflects the release, same as a real worker's
        # next 15s-guard refresh would see.
        role_state._replace_snapshot({k: v for k, v in role_state._snapshot.items() if k != "llama"})

        # A brand new failure on the (now-healthy-in-config, but currently
        # failing) role — must be free to promote again.
        await _open_breaker(LLAMA)
        _backdate_unhealthy(LLAMA, 200)
        fake_store.candidates = [_candidate("vendor/fresh-candidate")]

        promote_result = await promotion.consider_promotion("llama")

        assert promote_result == {"role": "llama", "from": LLAMA, "to": "vendor/fresh-candidate"}


# ── manual admin overrides ────────────────────────────────────────────────────

class TestManualOverride:
    @pytest.mark.asyncio
    async def test_set_manual_override_is_always_pinned(self, _reset):
        fake_store, _ = _reset
        result = await promotion.set_manual_override(object(), "llama", "vendor/picked")
        assert result == {"role": "llama", "model_id": "vendor/picked", "pinned": True}
        assert fake_store.overrides["llama"].pinned is True
        assert fake_store.overrides["llama"].reason == "manual"

    @pytest.mark.asyncio
    async def test_clear_manual_override_removes_the_row(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/picked", pinned=True, reason="manual")
        await promotion.clear_manual_override(object(), "llama")
        assert fake_store.cleared == ["llama"]

    @pytest.mark.asyncio
    async def test_set_role_pin_toggles_existing_row_without_changing_model(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False, reason="auto")
        result = await promotion.set_role_pin(object(), "llama", True)
        assert result == {"role": "llama", "model_id": "vendor/promoted", "pinned": True}
        assert fake_store.overrides["llama"].pinned is True
        assert fake_store.overrides["llama"].model_id == "vendor/promoted"  # unchanged

    @pytest.mark.asyncio
    async def test_set_role_pin_true_with_no_existing_override_pins_the_base(self, _reset):
        fake_store, _ = _reset
        result = await promotion.set_role_pin(object(), "llama", True)
        assert result == {"role": "llama", "model_id": LLAMA, "pinned": True}
        assert fake_store.overrides["llama"].model_id == LLAMA

    @pytest.mark.asyncio
    async def test_set_role_pin_false_with_no_existing_override_is_a_no_op(self, _reset):
        fake_store, _ = _reset
        result = await promotion.set_role_pin(object(), "llama", False)
        assert result == {"role": "llama", "model_id": LLAMA, "pinned": False}
        assert fake_store.overrides == {}


# ── SSE visibility ────────────────────────────────────────────────────────────

class TestPromotionStatusEvents:
    def test_empty_when_disabled(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "MODEL_AUTO_PROMOTE_ENABLED", False, raising=False)
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False, reason="auto")
        assert promotion.promotion_status_events(["vendor/promoted"]) == []

    def test_empty_when_nothing_promoted(self, _reset):
        assert promotion.promotion_status_events([LLAMA, CODER, REASONING]) == []

    def test_empty_when_the_promoted_model_is_not_in_this_chain(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False, reason="auto")
        assert promotion.promotion_status_events([CODER, REASONING]) == []

    def test_one_event_for_an_auto_promoted_role_in_the_chain(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/promoted", pinned=False, reason="auto")
        events = promotion.promotion_status_events(["vendor/promoted", REASONING])
        assert len(events) == 1
        assert events[0]["type"] == "status"
        assert events[0]["stage"] == "route"
        assert events[0]["level"] == "info"
        assert "llama" in events[0]["detail"]
        assert LLAMA in events[0]["detail"]

    def test_no_event_for_a_manual_pin_never_a_notice_for_a_deliberate_admin_choice(self, _reset):
        fake_store, _ = _reset
        _seed_override(fake_store, "llama", model_id="vendor/picked", pinned=True, reason="manual")
        assert promotion.promotion_status_events(["vendor/picked"]) == []
