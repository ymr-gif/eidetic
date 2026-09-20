"""Model-id resolution (HANDOFF Phase 3, live model catalog).

Unit tier — no DB/Redis/NIM. `llm.catalog.cache` is exercised as a plain
in-process dict via its own `_reset_for_tests()` + `_replace_snapshot()`-style
mutation (through the module's private globals, mirroring how
tests/test_model_extras.py monkeypatches config dicts directly).
"""
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
from llm.catalog import cache as catalog_cache
from api.chat.model_resolve import (
    _resolve_model, lock_unavailable_status_event, resolve_effective_model, resolve_model_strict,
)

LLAMA = config.MODELS["llama"]
CODER = config.MODELS["coder"]
CATALOG_ID = "z-ai/glm-5.3-flash"


@pytest.fixture(autouse=True)
def _reset_catalog():
    catalog_cache._reset_for_tests()
    yield
    catalog_cache._reset_for_tests()


def _seed(model_id: str, *, enabled=True, status="live", fail_count=0, last_live_at="2026-09-19T00:00:00+00:00"):
    catalog_cache._replace_snapshot({
        **catalog_cache._snapshot,
        model_id: {
            "id": model_id, "label": model_id, "status": status, "enabled": enabled,
            "fail_count": fail_count, "price_in": None, "price_out": None,
            "context_window": None, "latency_ms": None, "supports_tools": None,
            "reasoning": None, "request_extras": None, "min_max_tokens": None,
            "last_live_at": last_live_at,
        },
    })


class TestResolveModel:
    def test_role_name_resolves_to_its_current_id(self):
        assert _resolve_model("llama") == LLAMA
        assert _resolve_model("coder") == CODER

    def test_literal_role_id_passes_through(self):
        assert _resolve_model(LLAMA) == LLAMA

    def test_none_and_empty_resolve_to_none(self):
        assert _resolve_model(None) is None
        assert _resolve_model("") is None

    def test_unknown_id_not_in_catalog_resolves_to_none(self):
        assert _resolve_model("nobody/made-this-up") is None

    def test_available_catalog_id_resolves(self):
        _seed(CATALOG_ID, enabled=True, status="live")
        assert _resolve_model(CATALOG_ID) == CATALOG_ID

    def test_disabled_catalog_id_resolves_to_none(self):
        _seed(CATALOG_ID, enabled=False, status="live")
        assert _resolve_model(CATALOG_ID) is None

    def test_delisted_catalog_id_resolves_to_none_even_if_enabled(self):
        _seed(CATALOG_ID, enabled=True, status="delisted")
        assert _resolve_model(CATALOG_ID) is None

    def test_not_found_catalog_id_resolves_to_none(self):
        _seed(CATALOG_ID, enabled=True, status="not_found")
        assert _resolve_model(CATALOG_ID) is None

    def test_one_failed_probe_still_tolerated_as_available(self):
        _seed(CATALOG_ID, enabled=True, status="timeout", fail_count=1)
        assert _resolve_model(CATALOG_ID) == CATALOG_ID

    def test_two_failed_probes_no_longer_available(self):
        _seed(CATALOG_ID, enabled=True, status="timeout", fail_count=2)
        assert _resolve_model(CATALOG_ID) is None

    def test_never_live_model_never_resolves_even_with_one_failed_probe(self):
        """HANDOFF Phase 7: a model that has NEVER answered a probe live must
        not resolve just because fail_count hasn't hit 2 yet — repro was
        nvidia/nemotron-3-ultra-550b-a55b, status=timeout, fail_count=1."""
        _seed(CATALOG_ID, enabled=True, status="timeout", fail_count=1, last_live_at=None)
        assert _resolve_model(CATALOG_ID) is None


class TestResolveModelStrict:
    def test_valid_role_name_returns_id(self):
        assert resolve_model_strict("llama") == LLAMA

    def test_unavailable_id_raises_422_model_unavailable(self):
        with pytest.raises(HTTPException) as exc:
            resolve_model_strict("nobody/made-this-up")
        assert exc.value.status_code == 422
        assert exc.value.detail["error"] == "model_unavailable"
        assert exc.value.detail["model"] == "nobody/made-this-up"

    def test_none_raises(self):
        with pytest.raises(HTTPException):
            resolve_model_strict(None)


class TestResolveEffectiveModel:
    def test_no_override_no_lock(self):
        model, unavailable = resolve_effective_model(None, None)
        assert model is None
        assert unavailable is False

    def test_override_wins_over_lock(self):
        model, unavailable = resolve_effective_model("coder", "llama")
        assert model == CODER
        assert unavailable is False

    def test_available_lock_resolves(self):
        model, unavailable = resolve_effective_model(None, "llama")
        assert model == LLAMA
        assert unavailable is False

    def test_unavailable_lock_falls_back_to_none_and_flags(self):
        model, unavailable = resolve_effective_model(None, "some/disabled-catalog-model")
        assert model is None
        assert unavailable is True

    def test_unavailable_lock_with_override_present_is_not_flagged(self):
        # The override takes priority; the (stale) lock being unavailable
        # doesn't matter this turn and shouldn't be reported as a problem.
        model, unavailable = resolve_effective_model("llama", "some/disabled-catalog-model")
        assert model == LLAMA
        assert unavailable is False

    def test_empty_lock_string_is_not_flagged(self):
        model, unavailable = resolve_effective_model(None, "")
        assert model is None
        assert unavailable is False

    def test_lock_becomes_unavailable_after_admin_disables_it(self):
        _seed(CATALOG_ID, enabled=True, status="live")
        model, unavailable = resolve_effective_model(None, CATALOG_ID)
        assert model == CATALOG_ID
        assert unavailable is False

        _seed(CATALOG_ID, enabled=False, status="live")
        model, unavailable = resolve_effective_model(None, CATALOG_ID)
        assert model is None
        assert unavailable is True


class TestLockUnavailableStatusEvent:
    def test_shape_and_message(self):
        event = lock_unavailable_status_event("z-ai/glm-5.3-flash")
        assert event["stage"] == "model"
        assert event["level"] == "error"
        assert event["detail"] == "Locked model z-ai/glm-5.3-flash is no longer available — using Auto"

    def test_end_to_end_with_resolve_effective_model(self):
        # Mirrors api/chat/stream.py:chat_stream's exact call sequence.
        _seed(CATALOG_ID, enabled=False, status="live")
        model, unavailable = resolve_effective_model(None, CATALOG_ID)
        assert unavailable is True
        event = lock_unavailable_status_event(CATALOG_ID)
        assert CATALOG_ID in event["detail"]
        assert model is None
