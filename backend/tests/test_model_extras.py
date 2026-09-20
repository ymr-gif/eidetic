"""Reasoning-model request budget hotfix (HANDOFF.md Phase 2b/2c, 2026-09-20).

Unit tier — no live NIM, no DB, no Redis. Covers:
  - extras_for(): per-model reasoning-toggle fields from config.MODEL_REQUEST_EXTRAS,
    {} for a model id not in the table (unknown ids / the homeserver alias must
    never get an unverified field)
  - apply_request_extras(): ALWAYS merges the model's extras (Phase 2c dropped the
    old fast=False "thinking stays on" path — see below) and applies
    config.MODEL_MIN_MAX_TOKENS's per-model floor when an EXPLICITLY-set
    max_tokens falls below it; never mutates the caller's dict; an absent
    max_tokens already means uncapped and is left alone
  - Phase 2c policy lock: the reasoning role's own chat turns now get the same
    extras as every other call (root found live that nemotron-3-super leaks
    chain-of-thought into `content` when thinking is left on) — there is no
    per-call "keep thinking on" branch left, so the helper takes no `fast` kwarg
  - end-to-end: llm.nim.call() actually sends the merged extras (and a floored
    max_tokens) in the JSON body when handed the model_params
    apply_request_extras() built
  - defensive: llm.nim never emits a reasoning_content/reasoning delta as a
    token in the SSE stream, and never returns one as `content` from a
    non-stream response — both read only the `content` field, by construction
"""
import asyncio
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
import llm.client as llm_client
from llm import nim
from llm.circuit_breaker import _failures, _open, _open_time
from llm.model_extras import apply_request_extras, extras_for


LLAMA     = "openai/gpt-oss-20b"
REASONING = "nvidia/nemotron-3-super-120b-a12b"
CODER     = "deepseek-ai/deepseek-v4-flash-0731"
UNKNOWN   = "some-future/catalog-model"


@pytest.fixture(autouse=True)
def _reset_circuit_state():
    _failures.clear()
    _open.clear()
    _open_time.clear()
    yield
    _failures.clear()
    _open.clear()
    _open_time.clear()


# ── extras_for() ─────────────────────────────────────────────────────────────

class TestExtrasFor:
    def test_returns_configured_field_per_model(self):
        assert extras_for(LLAMA) == {"reasoning_effort": "low"}
        assert extras_for(REASONING) == {"chat_template_kwargs": {"enable_thinking": False}}
        assert extras_for(CODER) == {"chat_template_kwargs": {"thinking": False}}

    def test_unknown_model_id_gets_no_extras(self):
        # Future catalog ids and the single homeserver alias must never receive
        # a field we haven't verified that model accepts.
        assert extras_for(UNKNOWN) == {}
        assert extras_for("mixtral") == {}

    def test_returns_a_shallow_copy_not_the_config_dict_itself(self):
        out = extras_for(REASONING)
        out["new_key"] = "x"
        assert "new_key" not in config.MODEL_REQUEST_EXTRAS[REASONING]


# ── apply_request_extras() ──────────────────────────────────────────────────

class TestApplyRequestExtras:
    def test_merges_extras_into_a_copy(self):
        params = {"temperature": 0.4}
        merged = apply_request_extras(LLAMA, params)
        assert merged == {"temperature": 0.4, "reasoning_effort": "low"}
        assert params == {"temperature": 0.4}  # caller's dict untouched

    def test_none_model_params_still_gets_extras(self):
        assert apply_request_extras(CODER, None) == {
            "chat_template_kwargs": {"thinking": False},
        }

    def test_unknown_model_gets_plain_passthrough(self):
        params = {"max_tokens": 4}
        assert apply_request_extras(UNKNOWN, params) == {"max_tokens": 4}

    def test_reasoning_role_gets_extras_too_no_thinking_on_path(self):
        # Phase 2c: the reasoning model's own chat turns used to keep thinking
        # ON (fast=False). That path is gone — every call for a listed model
        # gets its extras, including the reasoning role, because thinking-on
        # intermittently leaked chain-of-thought into `content` live.
        merged = apply_request_extras(REASONING, {"temperature": 0.2})
        assert merged["chat_template_kwargs"] == {"enable_thinking": False}


# ── apply_request_extras() — config.MODEL_MIN_MAX_TOKENS floor ─────────────

class TestMinMaxTokensFloor:
    def test_raises_explicit_low_max_tokens_to_configured_floor(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_MIN_MAX_TOKENS", {LLAMA: 512}, raising=False)
        merged = apply_request_extras(LLAMA, {"max_tokens": 60, "temperature": 0.4})
        assert merged["max_tokens"] == 512
        assert merged["temperature"] == 0.4

    def test_leaves_max_tokens_already_at_or_above_floor(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_MIN_MAX_TOKENS", {LLAMA: 512}, raising=False)
        merged = apply_request_extras(LLAMA, {"max_tokens": 2048})
        assert merged["max_tokens"] == 2048

    def test_absent_max_tokens_is_left_absent_not_forced_to_floor(self, monkeypatch):
        # No cap already means uncapped — forcing it to the floor would be a
        # NEW, lower cap on an otherwise-unbounded request.
        monkeypatch.setattr(config, "MODEL_MIN_MAX_TOKENS", {LLAMA: 512}, raising=False)
        merged = apply_request_extras(LLAMA, {"temperature": 0.2})
        assert "max_tokens" not in merged

    def test_model_with_no_configured_floor_is_never_touched(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_MIN_MAX_TOKENS", {LLAMA: 512}, raising=False)
        merged = apply_request_extras(REASONING, {"max_tokens": 1})
        assert merged["max_tokens"] == 1  # REASONING has no entry — untouched

    def test_floor_dict_is_env_tunable_per_model(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_MIN_MAX_TOKENS", {LLAMA: 200}, raising=False)
        merged = apply_request_extras(LLAMA, {"max_tokens": 60})
        assert merged["max_tokens"] == 200

    def test_real_gpt_oss_20b_floor_is_configured(self):
        # Locks the production floor (config.py GPT_OSS_20B_MIN_MAX_TOKENS,
        # default 512) without monkeypatching — gpt-oss-20b has no full
        # reasoning-off switch and starved live at max_tokens=60.
        assert config.MODEL_MIN_MAX_TOKENS.get(LLAMA) == config.GPT_OSS_20B_MIN_MAX_TOKENS


# ── end-to-end: the merged params actually reach the outgoing JSON body ────

class _Resp:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 3}}


class _FakeClient:
    def __init__(self):
        self.bodies: list[dict] = []

    async def post(self, url, headers=None, json=None):
        self.bodies.append(json)
        return _Resp()


def _patch_nim_call_env(monkeypatch):
    monkeypatch.setattr(nim, "is_open", lambda model: False)
    monkeypatch.setattr(nim, "record_success", lambda model: None)
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-secret", raising=False)
    monkeypatch.setattr(config, "LLM_FAILOVER_ENABLED", False, raising=False)


@pytest.mark.asyncio
async def test_nim_call_sends_merged_extras_in_body(monkeypatch):
    # CODER has no configured MODEL_MIN_MAX_TOKENS floor (unlike LLAMA), so this
    # isolates "extras merge" from the floor behavior covered separately below.
    fake = _FakeClient()
    monkeypatch.setattr(llm_client, "client", fake, raising=False)
    _patch_nim_call_env(monkeypatch)

    params = apply_request_extras(CODER, {"max_tokens": 4})
    result = await nim.call(CODER, [{"role": "user", "content": "hi"}], "rid", model_params=params)

    assert result["ok"] is True
    assert fake.bodies[0]["chat_template_kwargs"] == {"thinking": False}
    assert fake.bodies[0]["max_tokens"] == 4


@pytest.mark.asyncio
async def test_nim_call_sends_no_extras_for_unknown_model(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(llm_client, "client", fake, raising=False)
    _patch_nim_call_env(monkeypatch)

    params = apply_request_extras(UNKNOWN, {"max_tokens": 4})
    await nim.call(UNKNOWN, [{"role": "user", "content": "hi"}], "rid", model_params=params)

    assert "reasoning_effort" not in fake.bodies[0]
    assert "chat_template_kwargs" not in fake.bodies[0]


@pytest.mark.asyncio
async def test_nim_call_sends_extras_for_reasoning_role_too(monkeypatch):
    # Phase 2c end-to-end lock: a "real chat turn" on the reasoning model still
    # gets its thinking-off field in the outgoing body — no caller-side branch
    # skips it anymore.
    fake = _FakeClient()
    monkeypatch.setattr(llm_client, "client", fake, raising=False)
    _patch_nim_call_env(monkeypatch)

    params = apply_request_extras(REASONING, {"max_tokens": 80})
    await nim.call(REASONING, [{"role": "user", "content": "hi"}], "rid", model_params=params)

    assert fake.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_nim_call_floors_gpt_oss_20b_low_max_tokens_end_to_end(monkeypatch):
    # Phase 2c end-to-end lock: gpt-oss-20b at reasoning_effort=low still
    # starved live at max_tokens=60 — the production floor must actually reach
    # the outgoing body, not just the helper's return value.
    fake = _FakeClient()
    monkeypatch.setattr(llm_client, "client", fake, raising=False)
    _patch_nim_call_env(monkeypatch)

    params = apply_request_extras(LLAMA, {"max_tokens": 60})
    await nim.call(LLAMA, [{"role": "user", "content": "hi"}], "rid", model_params=params)

    assert fake.bodies[0]["max_tokens"] == config.GPT_OSS_20B_MIN_MAX_TOKENS


# ── defensive: reasoning_content/reasoning deltas never surface as tokens ──

class TestNonStreamExtractNeverReturnsReasoningContent:
    def test_message_with_only_reasoning_content_extracts_nothing(self):
        # A model that puts its hidden reasoning under reasoning_content and
        # leaves `content` empty must not have that reasoning text treated as
        # the answer — nim.call's empty-content path (bad_format) should fire
        # instead, not a leaked chain-of-thought reply.
        data = {"choices": [{"message": {
            "reasoning_content": "Let me think about the cat's name...",
        }}]}
        content, tool_calls = nim._extract(data)
        assert content is None
        assert tool_calls is None

    def test_message_with_reasoning_and_content_returns_only_content(self):
        data = {"choices": [{"message": {
            "reasoning_content": "Let me think about the cat's name...",
            "content":           "Whiskers",
        }}]}
        content, tool_calls = nim._extract(data)
        assert content == "Whiskers"

    def test_message_with_reasoning_field_variant_extracts_nothing(self):
        # Some backends use the bare key "reasoning" instead of
        # "reasoning_content" — same requirement either way.
        data = {"choices": [{"message": {
            "reasoning": "internal chain of thought",
        }}]}
        content, tool_calls = nim._extract(data)
        assert content is None
        assert tool_calls is None


class _FakeStreamCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSSEResponse:
    """status_code=200 + aiter_lines() replaying raw SSE 'data: ...' lines
    built directly from caller-supplied delta dicts (so a test can put
    reasoning_content/reasoning next to, or instead of, content)."""
    def __init__(self, deltas: list[dict]):
        self.status_code = 200
        self._deltas = deltas

    async def aiter_lines(self):
        for delta in self._deltas:
            payload = {"choices": [{"delta": delta, "finish_reason": None}]}
            yield f"data: {json.dumps(payload)}"
        yield "data: [DONE]"


class _FakeStreamHTTPClient:
    def __init__(self, response):
        self._response = response

    def stream(self, method, url, headers=None, json=None, timeout=None):
        return _FakeStreamCM(self._response)


class TestStreamNeverEmitsReasoningDeltasAsTokens:
    @pytest.mark.asyncio
    async def test_reasoning_content_only_delta_yields_nothing(self, monkeypatch):
        fake_response = _FakeSSEResponse([
            {"reasoning_content": "thinking about it..."},
        ])
        monkeypatch.setattr(llm_client, "client", _FakeStreamHTTPClient(fake_response))
        monkeypatch.setattr(config, "STREAM_TOTAL_TIMEOUT", 0)

        chunks = []
        async for chunk in nim.call_stream(REASONING, [{"role": "user", "content": "hi"}], "rid-r1"):
            chunks.append(chunk)

        assert chunks == []  # only `delta.content` is ever read; nothing to yield here
        assert not any(isinstance(c, str) and "thinking" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_reasoning_content_alongside_content_only_content_is_yielded(self, monkeypatch):
        fake_response = _FakeSSEResponse([
            {"reasoning_content": "thinking about it...", "content": "Paris"},
        ])
        monkeypatch.setattr(llm_client, "client", _FakeStreamHTTPClient(fake_response))
        monkeypatch.setattr(config, "STREAM_TOTAL_TIMEOUT", 0)

        chunks = []
        async for chunk in nim.call_stream(REASONING, [{"role": "user", "content": "hi"}], "rid-r2"):
            chunks.append(chunk)

        assert chunks == ["Paris"]

    @pytest.mark.asyncio
    async def test_bare_reasoning_field_variant_never_yielded(self, monkeypatch):
        fake_response = _FakeSSEResponse([
            {"reasoning": "internal chain of thought"},
            {"content": "ok"},
        ])
        monkeypatch.setattr(llm_client, "client", _FakeStreamHTTPClient(fake_response))
        monkeypatch.setattr(config, "STREAM_TOTAL_TIMEOUT", 0)

        chunks = []
        async for chunk in nim.call_stream(REASONING, [{"role": "user", "content": "hi"}], "rid-r3"):
            chunks.append(chunk)

        assert chunks == ["ok"]
