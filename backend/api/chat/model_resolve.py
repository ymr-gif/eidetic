"""Model-id resolution: role name/id/live-catalog id -> a routable model id.

Split out of api/chat/helpers.py (pre-split, HANDOFF Phase 3 — helpers.py was
577 lines) into its own module; helpers.py re-exports `_resolve_model` so every
existing `from .helpers import _resolve_model` call site keeps working
unchanged.
"""
from fastapi import HTTPException

from config import MODELS
from llm.catalog import cache as catalog_cache


def _resolve_model(name: str | None) -> str | None:
    """Role name ('llama'/'coder'/'reasoning') -> its current model id. A
    literal role id passes through unchanged. Otherwise (Phase 3): an id the
    live catalog currently considers AVAILABLE (enabled + live, or enabled
    with at most one recent failed probe AND at least one confirmed-live
    probe in its history — Phase 7: a model that has never once answered
    never counts as available) passes through too. Anything else (unknown
    id, disabled catalog model, a catalog id that has gone
    not_found/gone/delisted) returns None so the caller falls back to Auto
    routing rather than sending a request that will just fail."""
    if not name:
        return None
    if name in MODELS:
        return MODELS[name]
    if name in MODELS.values():
        return name
    if catalog_cache.is_available(name):
        return name
    return None


def resolve_effective_model(model_override: str | None, locked_model: str | None) -> tuple[str | None, bool]:
    """Resolve the model for the fallback chain from a per-request override
    plus a conversation's stored lock (api/chat/stream.py:chat_stream).

    Returns (effective_model, lock_unavailable). `lock_unavailable` is True
    iff a lock is set, no override was given, and the locked value no longer
    resolves — e.g. an admin disabled/delisted a catalog pick (Phase 3). The
    caller should fall back to Auto (effective_model is None in that case) and
    surface a status event, WITHOUT touching the stored `locked_model` — the
    lock itself is untouched, only this turn routes around it."""
    override_resolved = _resolve_model(model_override)
    locked_raw = (locked_model or "").strip()
    locked_resolved = _resolve_model(locked_raw) if locked_raw else None
    lock_unavailable = bool(locked_raw) and not override_resolved and locked_resolved is None
    return override_resolved or locked_resolved, lock_unavailable


def resolve_model_strict(name: str | None) -> str:
    """Same resolution as `_resolve_model`, but raises HTTP 422
    `model_unavailable` instead of silently returning None. Used where "no
    such model" is a caller error rather than a routing fallback decision —
    the conversation lock PATCH (api/conversations/lock.py) and the compare
    model picker (api/chat/stream.py)."""
    resolved = _resolve_model(name)
    if resolved is None:
        raise HTTPException(status_code=422, detail={"error": "model_unavailable", "model": name})
    return resolved


def lock_unavailable_status_event(locked_model: str | None) -> dict:
    """The activity-trace entry api/chat/stream.py appends to `ctx["activity"]`
    when `resolve_effective_model` reports `lock_unavailable=True` — surfaced
    to the client as a "status" SSE event (no separate event type needed) and
    persisted on the assistant message like every other context-build stage."""
    return {
        "stage": "model", "level": "error",
        "detail": f"Locked model {locked_model} is no longer available — using Auto",
    }


def resolve_compare_models(raw: list[str] | None, *, max_models: int = 4) -> list[str] | None:
    """Validate + strict-resolve an explicit `ChatRequest.compare_models` pick
    (Phase 3). `None`/empty keeps compare mode's original default (the 3 role
    models — resolved by the caller, llm.service.compare.compare_streams).
    Raises 422 `too_many_compare_models` over the cap, or `model_unavailable`
    (via resolve_model_strict) on the first id that doesn't resolve."""
    if not raw:
        return None
    if len(raw) > max_models:
        raise HTTPException(status_code=422, detail={"error": "too_many_compare_models", "max": max_models})
    return [resolve_model_strict(m) for m in raw]
