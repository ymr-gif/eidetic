"""llm/catalog/scanner.py — the live NIM model catalog scan (HANDOFF Phase 3).

Unit tier — no real DB/Redis/NIM. `llm_client.client` is a fake httpx-shaped
stand-in (mirrors the `_FakeClient` pattern in tests/test_model_extras.py);
the store layer (list_rows/upsert_scan_result/mark_delisted) and
catalog_cache.publish are monkeypatched to plain recorders so the scan's
control flow — filtering, status mapping, concurrency bound, the 429
keep-prior-status rule, delisting, homeserver skip, the Redis NX lock — is
exercised without a real database.
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import httpx
import pytest

import config
import llm.client as llm_client
from llm.catalog import cache as catalog_cache
from llm.catalog import scanner


class _FakeResp:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


class _FakeScanClient:
    """`outcomes[model_id]` drives what the 1-token probe returns for that id:
    'live' | 'reasoning_leak' | 404 | 410 | 429 | 'timeout' | 'error'. Models
    absent from `outcomes` default to 'live'."""

    def __init__(self, listed_models: list[str], outcomes: dict | None = None):
        self.listed_models = listed_models
        self.outcomes = outcomes or {}
        self.post_calls: list[str] = []

    async def get(self, url, headers=None, timeout=None):
        return _FakeResp(200, {"data": [{"id": m} for m in self.listed_models]})

    async def post(self, url, headers=None, json=None, timeout=None):
        model_id = json["model"]
        self.post_calls.append(model_id)
        outcome = self.outcomes.get(model_id, "live")

        if outcome == "timeout":
            raise httpx.TimeoutException("timed out")
        if outcome == "error":
            raise RuntimeError("boom")
        if outcome in (404, 410, 429):
            return _FakeResp(outcome)
        if outcome == "reasoning_leak":
            return _FakeResp(200, {"choices": [{"message": {
                "reasoning_content": "thinking about it...", "content": "",
            }}]})
        return _FakeResp(200, {"choices": [{"message": {"content": "hi"}}]})


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "nim", raising=False)
    monkeypatch.setattr(config, "USE_REDIS", False, raising=False)  # skip the Redis lock/meta path
    monkeypatch.setattr(config, "CATALOG_PROBE_CONCURRENCY", 6, raising=False)
    monkeypatch.setattr(config, "CATALOG_PROBE_TIMEOUT", 5, raising=False)
    monkeypatch.setattr(config, "MODELS", {"llama": "openai/gpt-oss-20b"}, raising=False)
    catalog_cache._reset_for_tests()
    yield
    catalog_cache._reset_for_tests()


class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def commit(self):
        pass


def _wire_store(monkeypatch, *, existing_status=None, upserts=None, delisted_calls=None):
    existing_status = existing_status or {}
    upserts = upserts if upserts is not None else []
    delisted_calls = delisted_calls if delisted_calls is not None else []

    async def _fake_list_rows(db, **kwargs):
        return [SimpleNamespace(id=k, status=v) for k, v in existing_status.items()]

    async def _fake_upsert(db, model_id, *, status, http_status, latency_ms, label, seed_enabled, reasoning=None):
        upserts.append({
            "id": model_id, "status": status, "http_status": http_status,
            "latency_ms": latency_ms, "label": label, "seed_enabled": seed_enabled,
            "reasoning": reasoning,
        })

    async def _fake_mark_delisted(db, seen_ids):
        delisted_calls.append(seen_ids)
        return len(existing_status.keys() - seen_ids)

    async def _fake_publish(entries=None):
        pass

    monkeypatch.setattr(scanner, "list_rows", _fake_list_rows)
    monkeypatch.setattr(scanner, "upsert_scan_result", _fake_upsert)
    monkeypatch.setattr(scanner, "mark_delisted", _fake_mark_delisted)
    monkeypatch.setattr(scanner, "AsyncSessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(catalog_cache, "publish", _fake_publish)
    return upserts, delisted_calls


class TestRunScan:
    @pytest.mark.asyncio
    async def test_homeserver_mode_skips_entirely(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_BACKEND", "homeserver", raising=False)
        result = await scanner.run_scan(trigger="manual")
        assert result == {"skipped": "homeserver"}

    @pytest.mark.asyncio
    async def test_status_mapping_live_not_found_gone_timeout_error(self, monkeypatch):
        models = ["good/live-model", "bad/not-found", "dead/gone-model", "slow/timeout-model", "broken/error-model"]
        fake = _FakeScanClient(models, outcomes={
            "bad/not-found": 404, "dead/gone-model": 410,
            "slow/timeout-model": "timeout", "broken/error-model": "error",
        })
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch)

        result = await scanner.run_scan(trigger="manual")

        by_id = {u["id"]: u for u in upserts}
        assert by_id["good/live-model"]["status"] == "live"
        assert by_id["bad/not-found"]["status"] == "not_found"
        assert by_id["dead/gone-model"]["status"] == "gone"
        assert by_id["slow/timeout-model"]["status"] == "timeout"
        assert by_id["broken/error-model"]["status"] == "error"
        assert result["scanned"] == 5

    @pytest.mark.asyncio
    async def test_429_keeps_prior_status_not_overwritten(self, monkeypatch):
        fake = _FakeScanClient(["flaky/model"], outcomes={"flaky/model": 429})
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch, existing_status={"flaky/model": "live"})

        await scanner.run_scan(trigger="manual")

        assert upserts[0]["status"] == "live"  # kept, not "error" or some 429-derived status
        assert upserts[0]["http_status"] == 429

    @pytest.mark.asyncio
    async def test_429_with_no_prior_row_falls_back_to_error(self, monkeypatch):
        fake = _FakeScanClient(["brandnew/model"], outcomes={"brandnew/model": 429})
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch, existing_status={})

        await scanner.run_scan(trigger="manual")

        assert upserts[0]["status"] == "error"

    @pytest.mark.asyncio
    async def test_non_chat_candidates_are_filtered_out(self, monkeypatch):
        fake = _FakeScanClient(["openai/gpt-oss-20b", "nvidia/nv-embedqa-e5-v5", "some/reranker-model"])
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch)

        await scanner.run_scan(trigger="manual")

        probed_ids = {u["id"] for u in upserts}
        assert "openai/gpt-oss-20b" in probed_ids
        assert "nvidia/nv-embedqa-e5-v5" not in probed_ids
        assert "some/reranker-model" not in probed_ids
        assert "nvidia/nv-embedqa-e5-v5" not in fake.post_calls
        assert "some/reranker-model" not in fake.post_calls

    @pytest.mark.asyncio
    async def test_role_models_seeded_enabled_others_seeded_disabled(self, monkeypatch):
        fake = _FakeScanClient(["openai/gpt-oss-20b", "z-ai/glm-5.3-flash"])
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch)

        await scanner.run_scan(trigger="manual")

        by_id = {u["id"]: u for u in upserts}
        assert by_id["openai/gpt-oss-20b"]["seed_enabled"] is True   # a role model (MODELS["llama"])
        assert by_id["z-ai/glm-5.3-flash"]["seed_enabled"] is False  # not a role model

    @pytest.mark.asyncio
    async def test_delisting_uses_this_scans_candidate_set(self, monkeypatch):
        fake = _FakeScanClient(["still/here"])
        monkeypatch.setattr(llm_client, "client", fake)
        _, delisted_calls = _wire_store(monkeypatch, existing_status={"still/here": "live", "gone/now": "live"})

        await scanner.run_scan(trigger="manual")

        assert delisted_calls == [{"still/here"}]

    @pytest.mark.asyncio
    async def test_reasoning_flag_set_when_content_empty_but_reasoning_content_present(self, monkeypatch):
        fake = _FakeScanClient(["leaky/reasoning-model"], outcomes={"leaky/reasoning-model": "reasoning_leak"})
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch)

        await scanner.run_scan(trigger="manual")

        assert upserts[0]["reasoning"] is True

    @pytest.mark.asyncio
    async def test_reasoning_flag_false_for_a_normal_response(self, monkeypatch):
        fake = _FakeScanClient(["normal/model"])
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch)

        await scanner.run_scan(trigger="manual")

        assert upserts[0]["reasoning"] is False

    @pytest.mark.asyncio
    async def test_probe_concurrency_never_exceeds_configured_limit(self, monkeypatch):
        monkeypatch.setattr(config, "CATALOG_PROBE_CONCURRENCY", 2, raising=False)
        models = [f"vendor/model-{i}" for i in range(6)]

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        class _ConcurrencyTrackingClient(_FakeScanClient):
            async def post(self, url, headers=None, json=None, timeout=None):
                nonlocal in_flight, max_in_flight
                async with lock:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                await asyncio.sleep(0.01)
                async with lock:
                    in_flight -= 1
                return await super().post(url, headers=headers, json=json, timeout=timeout)

        fake = _ConcurrencyTrackingClient(models)
        monkeypatch.setattr(llm_client, "client", fake)
        _wire_store(monkeypatch)

        await scanner.run_scan(trigger="manual")

        assert max_in_flight <= 2

    @pytest.mark.asyncio
    async def test_redis_lock_prevents_concurrent_scans(self, monkeypatch):
        monkeypatch.setattr(config, "USE_REDIS", True, raising=False)

        class _LockedRedis:
            async def set(self, key, value, nx=None, ex=None):
                return None  # NX set fails — someone else holds the lock

        import core.redis_client as rc
        monkeypatch.setattr(rc, "get_redis", lambda: _LockedRedis())

        result = await scanner.run_scan(trigger="manual")
        assert result == {"skipped": "already_running"}

    @pytest.mark.asyncio
    async def test_db_session_only_opened_after_all_network_calls_complete(self, monkeypatch):
        """Never hold a DB session open across the network probes (HANDOFF Phase 3
        rule) — assert AsyncSessionLocal isn't touched until every probe/list call
        has already returned."""
        fake = _FakeScanClient(["a/model", "b/model"])
        monkeypatch.setattr(llm_client, "client", fake)
        upserts, _ = _wire_store(monkeypatch)

        session_opened_at = []

        real_session_factory = scanner.AsyncSessionLocal

        def _tracking_factory():
            session_opened_at.append(len(fake.post_calls))
            return real_session_factory()

        monkeypatch.setattr(scanner, "AsyncSessionLocal", _tracking_factory)

        await scanner.run_scan(trigger="manual")

        # By the time the DB session opens, both probes already ran.
        assert session_opened_at == [2]
