"""model_catalog row persistence — thin CRUD wrapper around ModelCatalog.

No caching, no Redis here (that's cache.py) — this module only talks to
Postgres, and every function takes an already-open AsyncSession so callers
(the scanner, the admin endpoints) control the transaction boundary.
"""
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.catalog import ModelCatalog


async def get_row(db: AsyncSession, model_id: str) -> ModelCatalog | None:
    return await db.get(ModelCatalog, model_id)


async def list_rows(
    db: AsyncSession, *, q: str | None = None, status: str | None = None
) -> list[ModelCatalog]:
    stmt = select(ModelCatalog)
    if q:
        stmt = stmt.where(ModelCatalog.id.ilike(f"%{q}%"))
    if status:
        stmt = stmt.where(ModelCatalog.status == status)
    stmt = stmt.order_by(ModelCatalog.id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def upsert_scan_result(
    db: AsyncSession,
    model_id: str,
    *,
    status: str,
    http_status: int | None,
    latency_ms: int | None,
    label: str | None,
    seed_enabled: bool,
    reasoning: bool | None = None,
    ttfb_ms: int | None = None,
    tool_ok: bool | None = None,
    reasoning_leak: bool | None = None,
    ttfb_fail_reason: str | None = None,
    verified: bool = False,
) -> ModelCatalog:
    """Apply one scan probe result to the row for `model_id`, creating it if
    this is the first time the scanner has ever seen this id. `seed_enabled`
    only applies on first sight (role models start enabled, everything else
    starts disabled — admin must opt in); it is never re-applied to an
    existing row, so an admin's explicit enable/disable choice always wins.

    `last_live_at` (migration 051, HANDOFF Phase 7) is set to `now` whenever
    `status == "live"` and left untouched on every other outcome — it never
    goes backward to null once a model has genuinely been live at least
    once. Consulted by llm/catalog/cache.py:_entry_available so a model that
    has only ever timed out/errored (never live) isn't tolerated as
    "available" just because fail_count hasn't reached 2 yet.

    `ttfb_ms`/`tool_ok`/`reasoning_leak` (migration 052, HANDOFF Phase A
    prerequisite) are the scanner's second, stricter probe — run for LIVE
    rows only (the scanner passes `verified=True` exactly then). `verified`
    is the "did we even attempt the second probe this cycle" flag: a model
    that came back not_found/gone/timeout/error this scan gets NO opinion on
    tool_ok/reasoning_leak (left exactly as they were, not reset to None —
    the last time it WAS verified is still informative), while a live model
    always gets a fresh verdict (including a fresh None/False if this cycle's
    verify_model() call itself failed) so a promoted candidate can't coast on
    a stale tool_ok=True from before it regressed.

    `ttfb_fail_reason` (migration 053, root live-stack finding 2026-09-27):
    rides alongside `ttfb_ms` under the same `verified` gate — set (or
    cleared back to null on success) every verified pass, never touched
    otherwise. Distinguishes an ACTIVELY-FAILED TTFB probe (e.g.
    "reasoning_only" — the model spent its whole budget on reasoning_content
    and never reached content) from a row that has simply never been
    TTFB-probed."""
    now = datetime.now(timezone.utc)
    row = await db.get(ModelCatalog, model_id)

    if row is None:
        row = ModelCatalog(
            id=model_id,
            label=label,
            status=status,
            http_status=http_status,
            latency_ms=latency_ms,
            fail_count=0 if status == "live" else 1,
            enabled=seed_enabled,
            reasoning=reasoning,
            last_live_at=now if status == "live" else None,
            ttfb_ms=ttfb_ms if verified else None,
            tool_ok=tool_ok if verified else None,
            reasoning_leak=reasoning_leak if verified else None,
            ttfb_fail_reason=ttfb_fail_reason if verified else None,
            verified_at=now if verified else None,
            first_seen=now,
            last_checked=now,
            updated_at=now,
        )
        db.add(row)
        return row

    row.fail_count = 0 if status == "live" else (row.fail_count or 0) + 1
    row.status = status
    row.http_status = http_status
    row.latency_ms = latency_ms
    if status == "live":
        row.last_live_at = now
    if reasoning is not None:
        row.reasoning = reasoning
    if verified:
        row.ttfb_ms = ttfb_ms
        row.tool_ok = tool_ok
        row.reasoning_leak = reasoning_leak
        row.ttfb_fail_reason = ttfb_fail_reason
        row.verified_at = now
    if not row.label and label:
        row.label = label
    row.last_checked = now
    row.updated_at = now
    return row


async def mark_delisted(db: AsyncSession, seen_ids: set[str]) -> int:
    """Any row NOT in `seen_ids` (this scan's live /v1/models listing) moves to
    status='delisted' — the id used to exist and no longer does. Returns the
    count of rows changed."""
    rows = await list_rows(db)
    now = datetime.now(timezone.utc)
    changed = 0
    for row in rows:
        if row.id not in seen_ids and row.status != "delisted":
            row.status = "delisted"
            row.updated_at = now
            changed += 1
    return changed
