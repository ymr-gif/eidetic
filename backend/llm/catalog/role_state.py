"""In-process snapshot of `model_role_override` rows (HANDOFF Phase A).

Mirrors llm/catalog/cache.py's own contract exactly — a 15s in-process guard,
a cheap Redis version check, a small hash of entries, DB fallback when Redis
is down, never raises into a chat request — so `effective_role_model()` stays
a synchronous, zero-I/O hot-path read just like `catalog_cache.is_available()`
(see llm/catalog/promotion.py). Kept as its OWN module with its OWN Redis keys
(`roles:ver`/`roles:entries`) rather than folded into cache.py, which is
already past the 200-line convention boundary — the two caches are refreshed
independently but on the SAME 15s guard window, so every worker agrees on
role-override state within one guard window either way (the "rides the same
Redis-versioned snapshot" HANDOFF wording is about the mechanism/guarantee,
not a literal shared dict — flagging this as a judgment call for review).

Inert (`ensure_fresh()` no-ops, `get()` always None) when
`config.MODEL_AUTO_PROMOTE_ENABLED` is false OR `LLM_BACKEND=homeserver` —
`effective_role_model()` never even calls `ensure_fresh()` in that case, so
this module does zero I/O when the feature is off.
"""
import json
import logging
import time
import uuid

import config

logger = logging.getLogger("catalog.role_state")

_REDIS_VER_KEY     = "roles:ver"
_REDIS_ENTRIES_KEY = "roles:entries"
_REFRESH_GUARD_SECONDS = 15

_snapshot: dict[str, dict] = {}       # role -> {model_id, pinned, promoted_at, reason, base_model_id}
_snapshot_ver: str | None = None
_last_checked_at: float = 0.0


def row_to_entry(row) -> dict:
    return {
        "role":          row.role,
        "model_id":      row.model_id,
        "pinned":        bool(row.pinned),
        "promoted_at":   row.promoted_at.isoformat() if row.promoted_at else None,
        "reason":        row.reason,
        "base_model_id": row.base_model_id,
    }


def _replace_snapshot(entries: dict[str, dict]) -> None:
    global _snapshot
    _snapshot = dict(entries)


def _set_snapshot_ver(ver: str | None) -> None:
    global _snapshot_ver
    _snapshot_ver = ver


async def _reload_from_db() -> None:
    from core.db import AsyncSessionLocal
    from llm.catalog.role_store import list_role_overrides

    async with AsyncSessionLocal() as db:
        rows = await list_role_overrides(db)
    _replace_snapshot({r.role: row_to_entry(r) for r in rows})


async def ensure_fresh() -> None:
    """Refresh the snapshot if it might be stale. Safe to call every request —
    the 15s guard makes repeat calls a no-op in the common case. Never raises.
    Callers should skip calling this entirely when the feature is off/inert
    (see llm/catalog/promotion.py:effective_role_model) rather than rely on
    this function alone to short-circuit, so the feature stays truly zero-I/O
    when disabled."""
    global _last_checked_at
    if config.LLM_BACKEND == "homeserver":
        return

    now = time.monotonic()
    if now - _last_checked_at < _REFRESH_GUARD_SECONDS:
        return
    _last_checked_at = now

    if not config.USE_REDIS:
        try:
            await _reload_from_db()
        except Exception:
            logger.warning("[role_state] DB reload failed — serving stale snapshot", exc_info=True)
        return

    try:
        from core.redis_client import get_redis
        redis = get_redis()
        ver = await redis.get(_REDIS_VER_KEY)
        if ver is not None and ver == _snapshot_ver:
            return
        if ver is None:
            await _reload_from_db()
            return
        raw = await redis.hgetall(_REDIS_ENTRIES_KEY)
        _replace_snapshot({k: json.loads(v) for k, v in raw.items()} if raw else {})
        _set_snapshot_ver(ver)
    except Exception:
        logger.warning("[role_state] Redis refresh failed — falling back to DB", exc_info=True)
        try:
            await _reload_from_db()
        except Exception:
            logger.warning("[role_state] DB fallback also failed — serving stale snapshot", exc_info=True)


async def publish(entries: dict[str, dict] | None = None) -> None:
    """Push the current DB state out — called after consider_promotion/
    consider_revert/the admin PATCH endpoint commits. `entries=None` (the
    common case) reloads from the DB first (source of truth, caller already
    committed)."""
    if entries is None:
        try:
            await _reload_from_db()
        except Exception:
            logger.warning("[role_state] publish: DB reload failed", exc_info=True)
            return
        entries = dict(_snapshot)
    else:
        _replace_snapshot(entries)

    if not config.USE_REDIS:
        return
    try:
        from core.redis_client import get_redis
        redis = get_redis()
        ver = uuid.uuid4().hex
        if entries:
            await redis.hset(_REDIS_ENTRIES_KEY, mapping={k: json.dumps(v) for k, v in entries.items()})
            existing_fields = await redis.hkeys(_REDIS_ENTRIES_KEY)
            stale = [f for f in existing_fields if f not in entries]
            if stale:
                await redis.hdel(_REDIS_ENTRIES_KEY, *stale)
        else:
            await redis.delete(_REDIS_ENTRIES_KEY)
        await redis.set(_REDIS_VER_KEY, ver)
        _set_snapshot_ver(ver)
    except Exception:
        logger.warning("[role_state] publish to Redis failed — other workers stay stale until their DB fallback", exc_info=True)


# ── synchronous readers (hot path — no I/O) ──────────────────────────────────

def get(role: str) -> dict | None:
    return _snapshot.get(role)


def all_overrides() -> dict[str, dict]:
    return dict(_snapshot)


def _reset_for_tests() -> None:
    global _snapshot, _snapshot_ver, _last_checked_at
    _snapshot = {}
    _snapshot_ver = None
    _last_checked_at = 0.0
