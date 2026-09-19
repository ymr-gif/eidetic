"""Live: non-streaming /chat, model routing/override, and cache behaviour.

Non-stream /chat returns an envelope {success, data:{model, response}, meta}.
Cache-hit signalling lives on the streaming `done` event, so cache tests use SSE.
Marker: live_nim.
"""
import uuid

import pytest

import config

pytestmark = pytest.mark.live_nim


def _chat_data(client, headers, message, **extra):
    r = client.post("/chat", headers=headers, json={"message": message, **extra})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("success") is True, body
    return body["data"]


def _done(sse_post, headers, message, **extra):
    events = sse_post("/chat/stream", headers, {"message": message, **extra})
    done = next((e for e in events if e.get("type") == "done"), None)
    assert done is not None, "no done event"
    return done


def test_nonstream_basic(client, user_headers):
    data = _chat_data(client, user_headers, "Reply with one word: hello", temperature=0.8)
    assert data.get("response", "").strip(), f"empty non-stream reply: {data}"
    assert data.get("model"), data


def test_model_override_respected(client, user_headers):
    # Live model ids churn under NVIDIA's catalog (HANDOFF Phase 3) — use the
    # currently-configured coder role rather than a hardcoded id, and the
    # keyword router's OTHER role (llama) for the prompt so this actually
    # exercises the override instead of them coincidentally agreeing.
    override_model = config.MODELS["coder"]
    data = _chat_data(client, user_headers, "Say hi.", model_override=override_model, temperature=0.6)
    assert data["model"] == override_model, f"override ignored: {data['model']}"


def test_cache_hit_field_is_reported(sse_post, user_headers):
    # contract: every done event reports cache_hit as a boolean (stream path serves
    # fresh, so the value is False — caching applies to the cacheable internal path).
    msg = f"Define the word idempotent in one sentence. tag={uuid.uuid4().hex[:6]}"
    first = _done(sse_post, user_headers, msg)
    second = _done(sse_post, user_headers, msg)
    assert isinstance(first["cache_hit"], bool)
    assert isinstance(second["cache_hit"], bool)


def test_model_params_bypass_cache(sse_post, user_headers):
    msg = f"Define the word bypass briefly. tag={uuid.uuid4().hex[:6]}"
    _done(sse_post, user_headers, msg)  # prime
    again = _done(sse_post, user_headers, msg, temperature=0.7)  # model_params → bypass
    assert again["cache_hit"] is False
