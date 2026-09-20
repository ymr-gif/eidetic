"""Compare mode, Phase 3 additions (HANDOFF): an explicit `compare_models`
pick (max 4, strict-resolved) overriding the default 3-role comparison, and
the new `compare_start` first event naming models + labels.

Unit tier — no live NIM, no DB, no Redis. `llm_client.client` is a fake
stream-shaped stand-in (mirrors `_FakeStreamHTTPClient`/`_FakeSSEResponse` in
tests/test_model_extras.py).
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
from fastapi import HTTPException

import config
import llm.client as llm_client
from llm.catalog import cache as catalog_cache
from llm.catalog.labels import derive_label
from llm.circuit_breaker import _failures, _open, _open_time
from llm.service import compare
from api.chat.model_resolve import resolve_compare_models

LLAMA = config.MODELS["llama"]
CODER = config.MODELS["coder"]
REASONING = config.MODELS["reasoning"]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _failures.clear()
    _open.clear()
    _open_time.clear()
    catalog_cache._reset_for_tests()
    monkeypatch.setattr(config, "STREAM_TOTAL_TIMEOUT", 0, raising=False)
    yield
    _failures.clear()
    _open.clear()
    _open_time.clear()
    catalog_cache._reset_for_tests()


# ── resolve_compare_models() ─────────────────────────────────────────────────

class TestResolveCompareModels:
    def test_none_keeps_default(self):
        assert resolve_compare_models(None) is None

    def test_empty_list_keeps_default(self):
        assert resolve_compare_models([]) is None

    def test_valid_roles_resolved(self):
        assert resolve_compare_models(["llama", "coder"]) == [LLAMA, CODER]

    def test_over_the_cap_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            resolve_compare_models(["a", "b", "c", "d", "e"])
        assert exc.value.status_code == 422
        assert exc.value.detail["error"] == "too_many_compare_models"

    def test_exactly_at_cap_is_allowed(self):
        catalog_cache._replace_snapshot({
            f"m{i}": {"id": f"m{i}", "label": f"M{i}", "status": "live", "enabled": True,
                      "fail_count": 0, "price_in": None, "price_out": None,
                      "context_window": None, "latency_ms": None, "supports_tools": None,
                      "reasoning": None, "request_extras": None, "min_max_tokens": None}
            for i in range(4)
        })
        assert resolve_compare_models(["m0", "m1", "m2", "m3"]) == ["m0", "m1", "m2", "m3"]

    def test_unavailable_id_raises_model_unavailable(self):
        with pytest.raises(HTTPException) as exc:
            resolve_compare_models(["nobody/made-this-up"])
        assert exc.value.status_code == 422
        assert exc.value.detail["error"] == "model_unavailable"


# ── llm.service.compare._label_for() ────────────────────────────────────────

class TestLabelFor:
    def test_catalog_label_preferred(self):
        catalog_cache._replace_snapshot({LLAMA: {
            "id": LLAMA, "label": "Custom Llama Label", "status": "live", "enabled": True,
            "fail_count": 0, "price_in": None, "price_out": None, "context_window": None,
            "latency_ms": None, "supports_tools": None, "reasoning": None,
            "request_extras": None, "min_max_tokens": None,
        }})
        assert compare._label_for(LLAMA) == "Custom Llama Label"

    def test_role_model_without_catalog_entry_uses_role_name(self):
        assert compare._label_for(LLAMA) == "Llama"

    def test_unknown_id_falls_back_to_derived_label(self):
        assert compare._label_for("z-ai/glm-5.3-flash") == derive_label("z-ai/glm-5.3-flash")


# ── compare_streams() ────────────────────────────────────────────────────────

class _FakeStreamCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSSEResponse:
    def __init__(self, content: str):
        self.status_code = 200
        self._content = content

    async def aiter_lines(self):
        payload = {"choices": [{"delta": {"content": self._content}, "finish_reason": None}]}
        yield f"data: {json.dumps(payload)}"
        yield "data: [DONE]"


class _FakeCompareClient:
    def stream(self, method, url, headers=None, json=None, timeout=None):
        model = json["model"]
        return _FakeStreamCM(_FakeSSEResponse(f"hi from {model}"))


@pytest.mark.asyncio
async def test_compare_start_is_first_event_default_role_models(monkeypatch):
    monkeypatch.setattr(llm_client, "client", _FakeCompareClient())

    events = []
    async for ev in compare.compare_streams("hi", [], None, "rid"):
        events.append(ev)

    assert events[0]["type"] == "compare_start"
    assert [m["id"] for m in events[0]["models"]] == list(config.MODELS.values())
    assert all(m["label"] for m in events[0]["models"])


@pytest.mark.asyncio
async def test_explicit_models_override_default_and_are_labeled(monkeypatch):
    monkeypatch.setattr(llm_client, "client", _FakeCompareClient())
    picks = [LLAMA, CODER]

    events = []
    async for ev in compare.compare_streams("hi", [], None, "rid", models=picks):
        events.append(ev)

    assert events[0] == {
        "type": "compare_start",
        "models": [{"id": LLAMA, "label": "Llama"}, {"id": CODER, "label": "Coder"}],
    }


@pytest.mark.asyncio
async def test_tokens_are_tagged_per_model_and_final_done_marks_compare(monkeypatch):
    monkeypatch.setattr(llm_client, "client", _FakeCompareClient())
    picks = [LLAMA, CODER]

    events = []
    async for ev in compare.compare_streams("hi", [], None, "rid", models=picks):
        events.append(ev)

    token_events = [e for e in events if e.get("type") == "token"]
    assert {e["model"] for e in token_events} == set(picks)
    assert all(e["content"] == f"hi from {e['model']}" for e in token_events)

    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) == 1
    assert done_events[0]["compare"] is True
