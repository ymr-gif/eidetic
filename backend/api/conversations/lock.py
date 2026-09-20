"""Conversation lock (`locked_model`) validation — split out of
api/conversations/crud.py (pre-split, HANDOFF Phase 3).
"""
from models import Conversation

from api.chat.model_resolve import resolve_model_strict


def apply_locked_model(conv: Conversation, raw: str | None) -> None:
    """Set `conv.locked_model` from a PATCH body value.

    Empty/whitespace-only clears the lock. A non-empty value must resolve to a
    currently-routable model (role name/id, or an available live-catalog id,
    Phase 3) — raises HTTP 422 `model_unavailable` (via resolve_model_strict)
    otherwise, so `locked_model` is never persisted as a value the app can't
    actually route to."""
    value = (raw or "").strip()
    if not value:
        conv.locked_model = None
        return
    conv.locked_model = resolve_model_strict(value)
