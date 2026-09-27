"""Unit tests — run with: pytest backend/tests/test.py -v"""
import os
import sys
import pytest

# allow imports from backend/ without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY",  "test-key")
os.environ.setdefault("DATABASE_URL",    "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",       "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY",  "test-secret")


# ── model router ──────────────────────────────────────────────────────────────

from llm.router import classify


@pytest.mark.parametrize("msg,expected", [
    ("write a python function to sort a list", "coder"),
    ("fix this bug in my code",                "coder"),
    ("implement a binary search algorithm",    "coder"),
    ("explain why recursion works",            "reasoning"),
    ("what is the difference between tcp and udp", "reasoning"),
    ("how does garbage collection work",       "reasoning"),
    ("hello, how are you?",                    "llama"),
    ("tell me a joke",                         "llama"),
    ("what is the weather today",              "reasoning"),
])
def test_classify(msg, expected):
    assert classify(msg) == expected


# ── cache key generation ──────────────────────────────────────────────────────

from cache.keys import normalize, make_key


def test_normalize_strips_whitespace():
    assert normalize("  Hello  ") == "hello"


def test_normalize_lowercase():
    assert normalize("UPPER CASE") == "upper case"


def test_make_key_deterministic():
    assert make_key("hello") == make_key("hello")


def test_make_key_different_inputs():
    assert make_key("hello") != make_key("world")


def test_make_key_format():
    key = make_key("test")
    assert len(key) == 64  # sha256 hex digest


# ── auth / password hashing ───────────────────────────────────────────────────

from auth.security import verify_password
from passlib.context import CryptContext

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def test_verify_password_correct():
    hashed = _pwd.hash("mypassword")
    assert verify_password("mypassword", hashed) is True


def test_verify_password_wrong():
    hashed = _pwd.hash("mypassword")
    assert verify_password("wrongpassword", hashed) is False


# ── circuit breaker ───────────────────────────────────────────────────────────

import asyncio
import llm.circuit_breaker as cb


def _clear():
    cb._failures.clear()
    cb._open.clear()
    cb._open_time.clear()
    cb._unhealthy_since.clear()


def test_circuit_breaker_closed_initially():
    _clear()
    assert cb.is_open("test-model") is False


def test_circuit_breaker_opens_after_threshold():
    _clear()
    for _ in range(3):
        asyncio.run(cb.record_failure("test-model"))
    assert cb.is_open("test-model") is True


def test_circuit_breaker_resets_on_success():
    _clear()
    asyncio.run(cb.record_failure("test-model"))
    asyncio.run(cb.record_failure("test-model"))
    cb.record_success("test-model")
    assert cb.is_open("test-model") is False


# ── unhealthy_since (HANDOFF Phase A auto-promotion prerequisite) ────────────

def test_unhealthy_since_none_when_never_failed():
    _clear()
    assert cb.unhealthy_since("fresh-model") is None


def test_unhealthy_since_set_on_first_failure():
    _clear()
    asyncio.run(cb.record_failure("test-model"))
    assert cb.unhealthy_since("test-model") is not None


def test_unhealthy_since_not_overwritten_by_later_failures():
    """The FIRST failure's timestamp is what matters (continuous-downtime
    duration) — a second failure a moment later must not reset the clock."""
    _clear()
    asyncio.run(cb.record_failure("test-model"))
    first = cb.unhealthy_since("test-model")
    asyncio.run(cb.record_failure("test-model"))
    assert cb.unhealthy_since("test-model") == first


def test_unhealthy_since_cleared_by_record_success():
    _clear()
    asyncio.run(cb.record_failure("test-model"))
    cb.record_success("test-model")
    assert cb.unhealthy_since("test-model") is None


def test_unhealthy_since_survives_is_open_cooldown_reset():
    """is_open()'s 90s auto-reset clears `_failures` (gives the model a clean
    slate to re-earn) but must NOT clear `_unhealthy_since` — a model that
    keeps failing every real attempt past its own cooldown is still
    continuously down, which is exactly what auto-promotion's trigger needs
    to see (see llm/circuit_breaker.py's `_unhealthy_since` comment)."""
    _clear()
    asyncio.run(cb.record_failure("test-model"))
    first = cb.unhealthy_since("test-model")
    cb._open["test-model"] = True
    cb._open_time["test-model"] = 0.0  # force the 90s cooldown to have "expired"
    assert cb.is_open("test-model") is False   # triggers the reset path
    assert cb._failures.get("test-model", 0) == 0
    assert cb.unhealthy_since("test-model") == first


# ── config guards ─────────────────────────────────────────────────────────────

def test_models_dict_has_required_keys():
    from config import MODELS
    assert "llama" in MODELS
    assert "coder" in MODELS
    assert "reasoning" in MODELS


def test_models_values_non_empty():
    from config import MODELS
    for name, model_id in MODELS.items():
        assert model_id, f"MODELS['{name}'] is empty"
