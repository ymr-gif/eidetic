"""In-process model-catalog snapshot, refreshed cheaply per chat request.

Every API worker process keeps its own snapshot dict (`_snapshot`). Reading it
(`is_available`/`get_entry`/`available_ids`) is synchronous and free — no I/O
on the hot chat path. Freshness is maintained by `ensure_fresh()`, meant to be
awaited once near the start of chat request handling:

  - A 15s in-process guard skips the check entirely on most calls.
  - Past the guard, a cheap Redis GET (`catalog:ver`) says whether ANY worker
    published a change since this snapshot was taken (a scan finishing, or an
    admin PATCH). Version unchanged -> nothing to do.
  - Version changed -> read the small `catalog:entries` hash (one round trip,
    no DB) and rebuild the snapshot from it.
  - Redis unreachable, or USE_REDIS off -> fall back to a DB reload directly;
    if THAT also fails, keep serving the stale snapshot (never raise into a
    chat request over a catalog refresh).

`publish()` is the write side — called by the scanner after a scan commits and
by the admin PATCH endpoint after an edit commits. It reloads this process's
own snapshot from the DB (the source of truth) and pushes it to Redis so every
OTHER worker's next `ensure_fresh()` picks it up.

Inert (`ensure_fresh()` no-ops, `is_available()` always False) when
`LLM_BACKEND=homeserver` — there is no catalog there, MODELS collapses to one
alias and role-based resolution never falls through to it.
"""
import json
import logging
import time
import uuid

import config

logger = logging.getLogger("catalog.cache")

_REDIS_VER_KEY     = "catalog:ver"
_REDIS_ENTRIES_KEY = "catalog:entries"
_REFRESH_GUARD_SECONDS = 15

# Definitive-down statuses never count as available regardless of enabled.
_DEFINITIVE_DOWN = {"not_found", "gone", "delisted"}
# Transient-failure statuses still count as available for ONE probe cycle
# (fail_count<=1) — "tolerate 1 failed probe" (spec). Two in a row flips it.
# HANDOFF Phase 7: tolerance ALSO requires last_live_at to be set — a model
# that has never once answered a probe must never be "available" just
# because it hasn't failed twice yet. "Tolerate 1 failed probe" means a
# model that WAS live and blipped, not one that never worked.
_TRANSIENT = {"timeout", "error"}

_snapshot: dict[str, dict] = {}
_snapshot_ver: str | None = None
_last_checked_at: float = 0.0


def _entry_available(entry: dict) -> bool:
    if not entry.get("enabled"):
        return False
    status = entry.get("status")
    if status == "live":
        return True
    if (
        status in _TRANSIENT
        and (entry.get("fail_count") or 0) <= 1
        and entry.get("last_live_at")
    ):
        return True
    return False


def row_to_entry(row) -> dict:
    """ModelCatalog row -> the plain-dict shape stored in the snapshot / Redis
    hash (JSON-serializable — no datetimes)."""
    return {
        "id":             row.id,
        "label":          row.label or row.id,
        "status":         row.status,
        "enabled":        bool(row.enabled),
        "fail_count":     row.fail_count or 0,
        "price_in":       row.price_in,
        "price_out":      row.price_out,
        "context_window": row.context_window,
        "latency_ms":     row.latency_ms,
        "supports_tools": row.supports_tools,
        "reasoning":      row.reasoning,
        "request_extras": row.request_extras,
        "min_max_tokens": row.min_max_tokens,
        "last_live_at":   row.last_live_at.isoformat() if row.last_live_at else None,
    }


def _replace_snapshot(entries: dict[str, dict]) -> None:
    global _snapshot
    _snapshot = dict(entries)


def _set_snapshot_ver(ver: str | None) -> None:
    global _snapshot_ver
    _snapshot_ver = ver


async def _reload_from_db() -> None:
    from core.db import AsyncSessionLocal
    from llm.catalog.store import list_rows

    async with AsyncSessionLocal() as db:
        rows = await list_rows(db)
    _replace_snapshot({r.id: row_to_entry(r) for r in rows})


async def ensure_fresh() -> None:
    """Refresh the snapshot if it might be stale. Safe to call every request —
    the 15s guard makes repeat calls a no-op in the common case. Never raises."""
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
            logger.warning("[catalog.cache] DB reload failed — serving stale snapshot", exc_info=True)
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
        if raw:
            _replace_snapshot({k: json.loads(v) for k, v in raw.items()})
        else:
            await _reload_from_db()
        _set_snapshot_ver(ver)
    except Exception:
        logger.warning("[catalog.cache] Redis refresh failed — falling back to DB", exc_info=True)
        try:
            await _reload_from_db()
        except Exception:
            logger.warning("[catalog.cache] DB fallback also failed — serving stale snapshot", exc_info=True)


async def publish(entries: dict[str, dict] | None = None) -> None:
    """Push the current DB state out: refresh THIS process's snapshot, then
    (if Redis is up) bump `catalog:ver` + rewrite `catalog:entries` so every
    other worker's next `ensure_fresh()` picks the change up. Called after a
    scan commits or an admin PATCH commits.

    `entries=None` (the common case) reloads from the DB first — the caller
    has already committed, so this is the source of truth."""
    if entries is None:
        try:
            await _reload_from_db()
        except Exception:
            logger.warning("[catalog.cache] publish: DB reload failed", exc_info=True)
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
        logger.warning("[catalog.cache] publish to Redis failed — other workers stay stale until their DB fallback", exc_info=True)


# ── synchronous readers (hot path — no I/O) ──────────────────────────────────

def is_available(model_id: str) -> bool:
    entry = _snapshot.get(model_id)
    return bool(entry) and _entry_available(entry)


def get_entry(model_id: str) -> dict | None:
    return _snapshot.get(model_id)


def available_ids() -> list[str]:
    return [mid for mid, e in _snapshot.items() if _entry_available(e)]


def all_entries() -> dict[str, dict]:
    return dict(_snapshot)


def _reset_for_tests() -> None:
    """Test-only: clear the module-level snapshot state between tests."""
    global _snapshot, _snapshot_ver, _last_checked_at
    _snapshot = {}
    _snapshot_ver = None
    _last_checked_at = 0.0
