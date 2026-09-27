"""verify_model(id): tool-call quality probe (HANDOFF Phase A prerequisite).

The scanner's existing 1-token probe (llm/catalog/scanner.py) only proves a
model answers AT ALL — it says nothing about whether the model can be trusted
with tool calling, which is what a promoted role actually needs (the chat hot
path almost always offers tools, see backend/CLAUDE.md "AI Agent Tool Loop").
Two real failure modes motivate this file:

  - `nvidia/ising-calibration-1.5-31b` returns a well-formed-LOOKING tool call
    whose arguments are fabricated: asked to check the weather, it invented
    temperature/humidity/UV keys the declared schema never had. A model that
    "calls the tool" but ignores the schema is worse than one that refuses —
    the caller can't tell the difference from the wire shape alone.
  - A model can leak chain-of-thought straight into `content` even with the
    reasoning-off request extras applied (nemotron-3-super, HANDOFF Phase 2c
    live finding) — `reasoning_leak` flags that independently of `tool_ok`.

Raw httpx, same maintenance-traffic pattern as scanner.py: bypasses the
per-model circuit breaker and Prometheus metrics entirely (this is catalog
upkeep, not user chat traffic) and never touches llm.nim.call/call_stream.
"""
import asyncio
import json
import logging
import time

import httpx

import config

logger = logging.getLogger("catalog.verify")


class _NullAsyncCM:
    """No-op async context manager — used when `verify_model`/`probe_ttfb`
    are called with `sem=None` (the admin on-demand check has no concurrency
    bound to share)."""
    async def __aenter__(self): return None
    async def __aexit__(self, *a): return False


_NULL_CM = _NullAsyncCM()

# One fixed tool + prompt for every model — deterministic, comparable results
# across the whole catalog. The schema's own property names are exactly what
# `tool_ok` checks fabricated-argument-key models against.
_PROBE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a named location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
_PROBE_MESSAGES = [
    {"role": "user", "content": "What's the weather like in Paris right now? Use the get_weather tool to check."},
]

# Heuristic chain-of-thought markers (case-insensitive, checked in the first
# 200 chars) — a non-empty `content` opening with one of these while the
# reasoning-off request extras were applied is "thinking" leaking into the
# user-visible field instead of staying in `reasoning_content` (or not
# happening at all). Deliberately narrow/high-precision, mirroring the
# "fail toward NOT flagging" bias used throughout this codebase's other
# heuristic classifiers (llm/router.py's closing-intent veto, etc.).
_REASONING_LEAK_MARKERS = (
    "let me think", "let's think", "let me check", "okay, the user",
    "the user is asking", "i need to", "first, i", "step 1", "thinking:",
)


def _looks_like_reasoning_leak(content: str) -> bool:
    low = content.strip().lower()
    if len(low) < 40:
        return False
    return any(marker in low[:200] for marker in _REASONING_LEAK_MARKERS)


def _tool_call_args_subset_of_schema(tool_call: dict) -> bool:
    """True iff `tool_call`'s JSON arguments parse to an object whose keys are
    all declared in `_PROBE_TOOL_SCHEMA` — the exact check that rejects
    ising-calibration's fabricated temperature/humidity/UV keys."""
    try:
        fn = tool_call.get("function", {})
        if fn.get("name") != _PROBE_TOOL_SCHEMA["function"]["name"]:
            return False
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:
        return False
    if not isinstance(args, dict):
        return False
    allowed = set(_PROBE_TOOL_SCHEMA["function"]["parameters"]["properties"].keys())
    return set(args.keys()) <= allowed


def _auth_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if config.NVIDIA_API_KEY:
        headers["Authorization"] = f"Bearer {config.NVIDIA_API_KEY}"
    return headers


# Failure-reason values probe_ttfb can report — see its own docstring.
# "reasoning_only" is the root live-stack repro (2026-09-27): meta/muse-
# glimmer-30b spends its whole probe budget on reasoning_content and hits
# finish_reason="length" before ever emitting a content delta, even with
# chat_template_kwargs.enable_thinking:false applied (the flag suppresses
# reasoning_content at a LARGER budget but not at a small one — the model
# still emits a preamble, it just needs more tokens to get past it).
TTFB_FAIL_REASONING_ONLY = "reasoning_only"
TTFB_FAIL_NO_CONTENT     = "no_content"
TTFB_FAIL_HTTP_ERROR     = "http_error"
TTFB_FAIL_TIMEOUT        = "timeout"
TTFB_FAIL_ERROR          = "error"


async def probe_ttfb(model_id: str, sem: asyncio.Semaphore) -> dict:
    """Streaming time-to-first-CONTENT-token probe (HANDOFF Phase A
    prerequisite) — called by the scanner only for rows the plain liveness
    probe already found live. Deliberately separate from `latency_ms`: a
    non-stream 1-token call can return in ~1s (fast round trip once the
    WHOLE — possibly slow — generation finishes) while the same model takes
    tens of seconds to stream its FIRST real token (deepseek-v4.1-flash
    measured 1.4s non-stream vs. ~52s TTFB). Must wait for actual
    `delta.content`, not just any SSE line — many backends emit an empty
    role-only delta first, which would understate TTFB for exactly the
    pathological case this exists to catch.

    Returns `{"ttfb_ms": int | None, "fail_reason": str | None}` — never
    raises. `fail_reason` is None only when `ttfb_ms` is set. A model that
    spends the ENTIRE `CATALOG_TTFB_PROBE_MAX_TOKENS` budget on
    `reasoning_content` and never reaches real content is reported as
    `TTFB_FAIL_REASONING_ONLY`, distinct from `TTFB_FAIL_NO_CONTENT` (the
    stream ended with neither content NOR reasoning — a quieter kind of
    nothing) and from transport failures — a structural probe failure must
    never be indistinguishable from "this row has simply never been
    TTFB-probed" (both used to collapse to a bare None). Raw httpx, same
    maintenance-traffic pattern as the rest of this module and scanner.py."""
    import llm.client as llm_client
    from llm.model_extras import apply_request_extras
    async with sem:
        t0 = time.monotonic()
        saw_reasoning = False
        try:
            body = {"model": model_id, "messages": [{"role": "user", "content": "Say hi in one word."}],
                    "max_tokens": config.CATALOG_TTFB_PROBE_MAX_TOKENS, "stream": True,
                    **apply_request_extras(model_id, {})}
            async with llm_client.client.stream(
                "POST", config.NIM_URL, headers=_auth_headers(), json=body, timeout=config.CATALOG_PROBE_TIMEOUT,
            ) as resp:
                if resp.status_code != 200:
                    return {"ttfb_ms": None, "fail_reason": TTFB_FAIL_HTTP_ERROR}
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw in ("", "[DONE]"):
                        continue
                    try:
                        chunk = json.loads(raw)
                        choices = chunk.get("choices") or []
                        delta = (choices[0].get("delta") or {}) if choices else {}
                        content = delta.get("content")
                        reasoning_content = delta.get("reasoning_content") or delta.get("reasoning")
                    except Exception:
                        continue
                    if content:
                        return {"ttfb_ms": int((time.monotonic() - t0) * 1000), "fail_reason": None}
                    if reasoning_content:
                        saw_reasoning = True
        except httpx.TimeoutException:
            return {"ttfb_ms": None, "fail_reason": TTFB_FAIL_TIMEOUT}
        except Exception:
            return {"ttfb_ms": None, "fail_reason": TTFB_FAIL_ERROR}
    return {"ttfb_ms": None, "fail_reason": TTFB_FAIL_REASONING_ONLY if saw_reasoning else TTFB_FAIL_NO_CONTENT}


async def verify_model(model_id: str, sem: asyncio.Semaphore | None = None) -> dict:
    """One-shot tool-call verification. Returns
    `{"tool_ok": bool, "reasoning_leak": bool | None, "error": str | None,
      "latency_ms": int}`. Never raises — any failure (network, timeout,
      malformed response) comes back as `tool_ok=False` with `error` set, so a
      broken probe can never look like a passing one.

    `sem`: the scanner passes its own `CATALOG_PROBE_CONCURRENCY` semaphore
    (bounding this alongside every other scan probe); the admin on-demand
    `POST /admin/models/{id}/verify` endpoint calls this with `sem=None` — a
    single ad-hoc check needs no concurrency bound of its own.

    Budget: `CATALOG_VERIFY_MAX_TOKENS` (default 300, was a hardcoded 200) —
    bumped alongside probe_ttfb's own budget fix (root live-stack finding,
    2026-09-27) so a model with a reasoning preamble before its tool call has
    room to finish it. Already fails SAFE on truncation even at the old
    budget — `tool_ok` requires `bool(tool_calls)` to be true first, so a
    model that spends its whole budget on reasoning_content and never emits
    a tool_calls delta at all correctly comes back `tool_ok=False`, never a
    false positive — this bump only reduces false NEGATIVES on a genuinely
    good model that just reasons for a while first."""
    import llm.client as llm_client
    from llm.model_extras import apply_request_extras

    if llm_client.client is None:
        return {"tool_ok": False, "reasoning_leak": None, "error": "http_client_unavailable", "latency_ms": None}

    body = {
        "model":       model_id,
        "messages":    _PROBE_MESSAGES,
        "tools":       [_PROBE_TOOL_SCHEMA],
        "tool_choice": "auto",
        **apply_request_extras(model_id, {"max_tokens": config.CATALOG_VERIFY_MAX_TOKENS}),
    }

    t0 = time.monotonic()
    try:
        async with sem if sem is not None else _NULL_CM:
            resp = await llm_client.client.post(
                config.NIM_URL, headers=_auth_headers(), json=body, timeout=config.CATALOG_PROBE_TIMEOUT,
            )
    except Exception as e:
        return {"tool_ok": False, "reasoning_leak": None, "error": str(e)[:200], "latency_ms": None}

    latency_ms = int((time.monotonic() - t0) * 1000)
    if resp.status_code != 200:
        return {"tool_ok": False, "reasoning_leak": None, "error": f"http_{resp.status_code}", "latency_ms": latency_ms}

    try:
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return {"tool_ok": False, "reasoning_leak": None, "error": "no_choices", "latency_ms": latency_ms}
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
    except Exception as e:
        return {"tool_ok": False, "reasoning_leak": None, "error": f"malformed_response: {e}"[:200], "latency_ms": latency_ms}

    tool_ok = bool(tool_calls) and _tool_call_args_subset_of_schema(tool_calls[0])
    reasoning_leak = bool(isinstance(content, str) and _looks_like_reasoning_leak(content))

    return {"tool_ok": tool_ok, "reasoning_leak": reasoning_leak, "error": None, "latency_ms": latency_ms}
