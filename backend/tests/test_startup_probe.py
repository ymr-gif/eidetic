"""api/system.py:probe_models_on_startup — the pre-trip-fix (HANDOFF Phase 3).

Root found 3 deploys in a row where a single transient probe failure (NVIDIA
503 "overloaded", or a cold time-to-first-byte past the old 5s timeout)
pre-tripped a healthy model's circuit breaker, so every deploy started with 2
of 3 roles failed over for 90s. Fix: a longer probe timeout, one retry with a
short backoff before judging a model down, and pre-trip only on a DEFINITIVE
result (401/404/410) or two consecutive failed attempts.

Unit tier — no live NIM. `llm_client.client.post` is a fake sequencer so each
test controls exactly what the two possible attempts return.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest

import api.system as system
import config
import llm.client as llm_client


class _FakeResp:
    def __init__(self, status_code):
        self.status_code = status_code


class _SequencedClient:
    """Returns each entry of `sequence` in order on successive `.post()`
    calls; an entry that's an Exception subclass instance is raised instead
    of returned. Records call count."""
    def __init__(self, sequence: list):
        self._sequence = list(sequence)
        self.calls = 0

    async def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        item = self._sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResp(item)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(config, "MODELS", {"llama": "openai/gpt-oss-20b"}, raising=False)
    monkeypatch.setattr(system, "_STARTUP_PROBE_RETRY_BACKOFF", 0, raising=False)  # no real sleep in tests
    tripped = []

    async def _fake_pre_trip(model_id):
        tripped.append(model_id)

    monkeypatch.setattr(system, "_pre_trip", _fake_pre_trip)
    yield tripped


class TestProbeModelsOnStartup:
    @pytest.mark.asyncio
    async def test_first_attempt_200_ok_no_retry_no_trip(self, monkeypatch, _reset):
        fake = _SequencedClient([200])
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert fake.calls == 1
        assert _reset == []

    @pytest.mark.asyncio
    async def test_transient_failure_then_success_on_retry_no_trip(self, monkeypatch, _reset):
        fake = _SequencedClient([503, 200])
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert fake.calls == 2
        assert _reset == []  # the whole point of the fix: no pre-trip on a transient blip

    @pytest.mark.asyncio
    async def test_transient_failure_twice_pre_trips(self, monkeypatch, _reset):
        fake = _SequencedClient([503, 503])
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert fake.calls == 2
        assert _reset == ["openai/gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_network_exception_then_success_no_trip(self, monkeypatch, _reset):
        fake = _SequencedClient([RuntimeError("connection refused"), 200])
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert fake.calls == 2
        assert _reset == []

    @pytest.mark.asyncio
    async def test_definitive_404_pre_trips_immediately_no_retry(self, monkeypatch, _reset):
        fake = _SequencedClient([404])
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert fake.calls == 1  # no second attempt burned on a definitive result
        assert _reset == ["openai/gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_definitive_410_pre_trips_immediately(self, monkeypatch, _reset):
        fake = _SequencedClient([410])
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert fake.calls == 1
        assert _reset == ["openai/gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_definitive_401_pre_trips_immediately(self, monkeypatch, _reset):
        fake = _SequencedClient([401])
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert fake.calls == 1
        assert _reset == ["openai/gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_multiple_roles_probed_independently(self, monkeypatch, _reset):
        monkeypatch.setattr(config, "MODELS", {
            "llama": "openai/gpt-oss-20b",
            "coder": "deepseek-ai/deepseek-v4-flash-0731",
        }, raising=False)

        class _PerModelClient:
            def __init__(self):
                self.calls = []

            async def post(self, url, headers=None, json=None, timeout=None):
                model = json["model"]
                self.calls.append(model)
                if model == "openai/gpt-oss-20b":
                    return _FakeResp(200)
                return _FakeResp(404)

        fake = _PerModelClient()
        monkeypatch.setattr(llm_client, "client", fake)

        await system.probe_models_on_startup()

        assert _reset == ["deepseek-ai/deepseek-v4-flash-0731"]
