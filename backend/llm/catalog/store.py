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
) -> ModelCatalog:
    """Apply one scan probe result to the row for `model_id`, creating it if
    this is the first time the scanner has ever seen this id. `seed_enabled`
    only applies on first sight (role models start enabled, everything else
    starts disabled — admin must opt in); it is never re-applied to an
    existing row, so an admin's explicit enable/disable choice always wins."""
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
    if reasoning is not None:
        row.reasoning = reasoning
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
