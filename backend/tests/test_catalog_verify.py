"""llm/catalog/verify.py — the tool-call quality probe (HANDOFF Phase A
prerequisite). Unit tier, no real DB/Redis/NIM — `llm_client.client` is a fake
httpx-shaped stand-in (same pattern as tests/test_catalog_scanner.py).
"""
import asyncio
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
from llm.catalog import verify


class _FakeResp:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class _FakePostClient:
    def __init__(self, response: _FakeResp | Exception):
        self._response = response
        self.calls = 0
        self.last_body: dict | None = None

    async def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        self.last_body = json
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeStreamResp:
    """Fake streaming response for probe_ttfb's `.stream()` call.
    `lines`: raw SSE `data: ...` lines (already formatted) to yield in order.
    `status_code`: defaults to 200."""
    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCM:
    def __init__(self, resp: _FakeStreamResp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


class _FakeStreamClient:
    """`stream_result`: either a `_FakeStreamResp` or an Exception to raise
    when `llm_client.client.stream(...)` is entered."""
    def __init__(self, stream_result):
        self._stream_result = stream_result
        self.last_body: dict | None = None

    def stream(self, method, url, headers=None, json=None, timeout=None):
        self.last_body = json
        if isinstance(self._stream_result, Exception):
            raise self._stream_result
        return _FakeStreamCM(self._stream_result)


def _sse(obj) -> str:
    import json as _json
    return f"data: {_json.dumps(obj)}"


def _delta_line(*, content=None, reasoning_content=None) -> str:
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    return _sse({"choices": [{"delta": delta, "finish_reason": None}]})


def _tool_call(args_json: str, name: str = "get_weather") -> dict:
    return {"id": "1", "type": "function", "function": {"name": name, "arguments": args_json}}


class TestToolCallArgsSubsetOfSchema:
    def test_subset_args_pass(self):
        assert verify._tool_call_args_subset_of_schema(_tool_call('{"location": "Paris"}')) is True

    def test_fabricated_extra_keys_fail(self):
        # the ising-calibration-1.5-31b repro — invented keys the schema never had
        tc = _tool_call('{"location": "Paris", "temperature": 20, "humidity": 50, "uv": 3}')
        assert verify._tool_call_args_subset_of_schema(tc) is False

    def test_wrong_function_name_fails(self):
        assert verify._tool_call_args_subset_of_schema(_tool_call('{"location": "Paris"}', name="other_fn")) is False

    def test_malformed_json_fails(self):
        assert verify._tool_call_args_subset_of_schema(_tool_call("{not json")) is False

    def test_non_object_args_fail(self):
        assert verify._tool_call_args_subset_of_schema(_tool_call("[1, 2, 3]")) is False


class TestLooksLikeReasoningLeak:
    def test_short_content_never_flagged(self):
        assert verify._looks_like_reasoning_leak("Sure, here's the weather.") is False

    def test_cot_opener_flagged(self):
        text = "Okay, the user is asking about the weather in Paris, let me work through this carefully."
        assert verify._looks_like_reasoning_leak(text) is True

    def test_clean_long_answer_not_flagged(self):
        text = "The current weather in Paris is 18C and partly cloudy, with light winds from the west."
        assert verify._looks_like_reasoning_leak(text) is False


@pytest.fixture(autouse=True)
def _config_defaults(monkeypatch):
    monkeypatch.setattr(config, "CATALOG_PROBE_TIMEOUT", 5, raising=False)
    monkeypatch.setattr(config, "MODEL_REQUEST_EXTRAS", {}, raising=False)
    monkeypatch.setattr(config, "MODEL_MIN_MAX_TOKENS", {}, raising=False)
    monkeypatch.setattr(config, "CATALOG_TTFB_PROBE_MAX_TOKENS", 64, raising=False)
    monkeypatch.setattr(config, "CATALOG_VERIFY_MAX_TOKENS", 300, raising=False)
    yield


class TestProbeTtfb:
    """Root live-stack finding (2026-09-27): meta/muse-glimmer-30b spends its
    whole (formerly 5-token) probe budget on reasoning_content and hits
    finish_reason="length" before any content delta — probe_ttfb used to
    collapse that to a bare None, indistinguishable from "never probed" and
    excluding the model from promotion with no way to tell why."""

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_returns_failed_ttfb_not_none_collapse(self, monkeypatch):
        """The exact repro: only reasoning_content, then [DONE]."""
        resp = _FakeStreamResp([
            _delta_line(reasoning_content="Say hi"),
            "data: [DONE]",
        ])
        fake = _FakeStreamClient(resp)
        monkeypatch.setattr(llm_client, "client", fake)

        result = await verify.probe_ttfb("meta/muse-glimmer-30b", asyncio.Semaphore(1))

        assert result["ttfb_ms"] is None
        assert result["fail_reason"] == verify.TTFB_FAIL_REASONING_ONLY

    @pytest.mark.asyncio
    async def test_reasoning_then_content_stream_records_content_delta_timing(self, monkeypatch):
        """A model that reasons first and THEN answers must still get a real
        ttfb_ms — the timing is to the content delta, not the reasoning one,
        but a preamble must not itself cause a failure."""
        resp = _FakeStreamResp([
            _delta_line(reasoning_content="thinking..."),
            _delta_line(reasoning_content="still thinking..."),
            _delta_line(content="Hi"),
            "data: [DONE]",
        ])
        fake = _FakeStreamClient(resp)
        monkeypatch.setattr(llm_client, "client", fake)

        result = await verify.probe_ttfb("vendor/reasons-then-answers", asyncio.Semaphore(1))

        assert result["fail_reason"] is None
        assert isinstance(result["ttfb_ms"], int)
        assert result["ttfb_ms"] >= 0

    @pytest.mark.asyncio
    async def test_pure_silence_stream_is_no_content_not_reasoning_only(self, monkeypatch):
        """Stream ends with neither content NOR reasoning_content — a
        DIFFERENT failure mode than the reasoning-preamble repro, must be
        distinguishable in the logs."""
        resp = _FakeStreamResp(["data: [DONE]"])
        fake = _FakeStreamClient(resp)
        monkeypatch.setattr(llm_client, "client", fake)

        result = await verify.probe_ttfb("vendor/silent", asyncio.Semaphore(1))

        assert result["ttfb_ms"] is None
        assert result["fail_reason"] == verify.TTFB_FAIL_NO_CONTENT

    @pytest.mark.asyncio
    async def test_http_error_status_reports_http_error_reason(self, monkeypatch):
        resp = _FakeStreamResp([], status_code=500)
        fake = _FakeStreamClient(resp)
        monkeypatch.setattr(llm_client, "client", fake)

        result = await verify.probe_ttfb("vendor/broken", asyncio.Semaphore(1))

        assert result["ttfb_ms"] is None
        assert result["fail_reason"] == verify.TTFB_FAIL_HTTP_ERROR

    @pytest.mark.asyncio
    async def test_timeout_reports_timeout_reason(self, monkeypatch):
        import httpx
        fake = _FakeStreamClient(httpx.TimeoutException("timed out"))
        monkeypatch.setattr(llm_client, "client", fake)

        result = await verify.probe_ttfb("vendor/slow", asyncio.Semaphore(1))

        assert result["ttfb_ms"] is None
        assert result["fail_reason"] == verify.TTFB_FAIL_TIMEOUT

    @pytest.mark.asyncio
    async def test_generic_exception_reports_error_reason(self, monkeypatch):
        fake = _FakeStreamClient(RuntimeError("boom"))
        monkeypatch.setattr(llm_client, "client", fake)

        result = await verify.probe_ttfb("vendor/whatever", asyncio.Semaphore(1))

        assert result["ttfb_ms"] is None
        assert result["fail_reason"] == verify.TTFB_FAIL_ERROR

    @pytest.mark.asyncio
    async def test_uses_the_configured_ttfb_probe_budget_not_five(self, monkeypatch):
        monkeypatch.setattr(config, "CATALOG_TTFB_PROBE_MAX_TOKENS", 64, raising=False)
        resp = _FakeStreamResp([_delta_line(content="Hi"), "data: [DONE]"])
        fake = _FakeStreamClient(resp)
        monkeypatch.setattr(llm_client, "client", fake)

        await verify.probe_ttfb("vendor/model", asyncio.Semaphore(1))

        assert fake.last_body["max_tokens"] == 64
        assert fake.last_body["max_tokens"] != 5  # the old hardcoded budget that caused the live-stack bug


class TestVerifyModel:
    @pytest.mark.asyncio
    async def test_well_formed_tool_call_passes(self, monkeypatch):
        fake = _FakePostClient(_FakeResp(200, {"choices": [{"message": {
            "content": None, "tool_calls": [_tool_call('{"location": "Paris"}')],
        }}]}))
        monkeypatch.setattr(llm_client, "client", fake)
        result = await verify.verify_model("vendor/good-model")
        assert result == {"tool_ok": True, "reasoning_leak": False, "error": None, "latency_ms": result["latency_ms"]}
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_fabricated_args_fail_tool_ok(self, monkeypatch):
        fake = _FakePostClient(_FakeResp(200, {"choices": [{"message": {
            "content": None,
            "tool_calls": [_tool_call('{"location": "Paris", "temperature": 20, "humidity": 50}')],
        }}]}))
        monkeypatch.setattr(llm_client, "client", fake)
        result = await verify.verify_model("nvidia/ising-calibration-1.5-31b")
        assert result["tool_ok"] is False
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_no_tool_call_fails_tool_ok(self, monkeypatch):
        fake = _FakePostClient(_FakeResp(200, {"choices": [{"message": {"content": "I can't check that."}}]}))
        monkeypatch.setattr(llm_client, "client", fake)
        result = await verify.verify_model("vendor/refuses-tools")
        assert result["tool_ok"] is False

    @pytest.mark.asyncio
    async def test_reasoning_leak_flagged_independently_of_tool_ok(self, monkeypatch):
        fake = _FakePostClient(_FakeResp(200, {"choices": [{"message": {
            "content": "Let me think about this before I answer the question properly.",
            "tool_calls": [_tool_call('{"location": "Paris"}')],
        }}]}))
        monkeypatch.setattr(llm_client, "client", fake)
        result = await verify.verify_model("vendor/leaky-model")
        assert result["tool_ok"] is True       # the tool call itself is still well-formed
        assert result["reasoning_leak"] is True

    @pytest.mark.asyncio
    async def test_http_error_status_fails_gracefully(self, monkeypatch):
        fake = _FakePostClient(_FakeResp(500))
        monkeypatch.setattr(llm_client, "client", fake)
        result = await verify.verify_model("vendor/broken")
        assert result["tool_ok"] is False
        assert result["error"] == "http_500"

    @pytest.mark.asyncio
    async def test_network_exception_fails_gracefully_never_raises(self, monkeypatch):
        fake = _FakePostClient(RuntimeError("boom"))
        monkeypatch.setattr(llm_client, "client", fake)
        result = await verify.verify_model("vendor/unreachable")
        assert result["tool_ok"] is False
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_no_client_returns_unavailable_without_raising(self, monkeypatch):
        monkeypatch.setattr(llm_client, "client", None)
        result = await verify.verify_model("vendor/whatever")
        assert result["tool_ok"] is False
        assert result["error"] == "http_client_unavailable"

    @pytest.mark.asyncio
    async def test_semaphore_is_respected_when_provided(self, monkeypatch):
        """The scanner passes its own concurrency semaphore; a single ad-hoc
        admin-triggered verify (sem=None) must work identically without one."""
        fake = _FakePostClient(_FakeResp(200, {"choices": [{"message": {
            "content": None, "tool_calls": [_tool_call('{"location": "Paris"}')],
        }}]}))
        monkeypatch.setattr(llm_client, "client", fake)
        sem = asyncio.Semaphore(1)
        result = await verify.verify_model("vendor/good-model", sem)
        assert result["tool_ok"] is True
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_uses_the_configured_verify_budget_not_the_old_200(self, monkeypatch):
        """Root follow-up (2026-09-27): bumped alongside probe_ttfb's own
        budget fix so a model with a reasoning preamble before its tool call
        has room to finish it."""
        monkeypatch.setattr(config, "CATALOG_VERIFY_MAX_TOKENS", 300, raising=False)
        fake = _FakePostClient(_FakeResp(200, {"choices": [{"message": {
            "content": None, "tool_calls": [_tool_call('{"location": "Paris"}')],
        }}]}))
        monkeypatch.setattr(llm_client, "client", fake)
        await verify.verify_model("vendor/good-model")
        assert fake.last_body["max_tokens"] == 300

    @pytest.mark.asyncio
    async def test_budget_exhausted_with_zero_tool_calls_fails_safe_not_a_false_positive(self, monkeypatch):
        """Confirms verify_model is not "just lucky" with today's models
        (root follow-up ask #4): a model that spends its ENTIRE budget on
        reasoning and never emits a tool_calls delta at all must still come
        back tool_ok=False, by construction (`bool(tool_calls)` gates first),
        never an ambiguous pass."""
        fake = _FakePostClient(_FakeResp(200, {"choices": [{"message": {
            "content": None, "tool_calls": None, "reasoning_content": "thinking forever...",
        }}]}))
        monkeypatch.setattr(llm_client, "client", fake)
        result = await verify.verify_model("vendor/reasons-forever")
        assert result["tool_ok"] is False
        assert result["error"] is None  # a valid HTTP 200 response — just no tool call in it
