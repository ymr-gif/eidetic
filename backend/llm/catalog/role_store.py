"""model_role_override row persistence + promotion-candidate selection
(HANDOFF Phase A). Split out of store.py (already at the 200-line convention
boundary after migration 052's ttfb_ms/tool_ok/reasoning_leak/verified_at
additions) rather than grown in place — same "new logic in new modules"
convention as api/chat/model_resolve.py's own pre-split.

Every function takes an already-open AsyncSession, same contract as
store.py: the caller (llm/catalog/promotion.py) owns the transaction.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.catalog import ModelCatalog, ModelRoleOverride


async def get_role_override(db: AsyncSession, role: str) -> ModelRoleOverride | None:
    return await db.get(ModelRoleOverride, role)


async def list_role_overrides(db: AsyncSession) -> list[ModelRoleOverride]:
    result = await db.execute(select(ModelRoleOverride))
    return list(result.scalars().all())


async def set_role_override(
    db: AsyncSession,
    role: str,
    *,
    model_id: str,
    pinned: bool,
    reason: str | None,
    base_model_id: str,
    promoted_at: datetime | None = None,
) -> ModelRoleOverride:
    """Create or overwrite the override row for `role` — both
    consider_promotion's automatic path and the admin manual-assign endpoint
    go through this one function, differing only in the kwargs they pass:
    `reason="auto"`/`pinned=False` vs. `reason="manual"`/`pinned=True` (a
    manual assignment always implies pinned, per the HANDOFF design
    decision — enforced by the CALLER, not here, so a future caller isn't
    silently forced into always-pinned)."""
    row = await db.get(ModelRoleOverride, role)
    if row is None:
        row = ModelRoleOverride(role=role)
        db.add(row)
    row.model_id = model_id
    row.pinned = pinned
    row.reason = reason
    row.base_model_id = base_model_id
    row.promoted_at = promoted_at or datetime.now(timezone.utc)
    return row


async def clear_role_override(db: AsyncSession, role: str) -> bool:
    """Remove the override row for `role`, if any. Returns True iff a row was
    actually deleted (False = the role was already unpromoted — a no-op, not
    an error, for both consider_revert and the admin clear action)."""
    row = await db.get(ModelRoleOverride, role)
    if row is None:
        return False
    await db.delete(row)
    return True


async def list_promotion_candidates(
    db: AsyncSession,
    *,
    exclude_ids: set[str],
    max_ttfb_ms: int,
    max_stale_min: int,
) -> list[ModelCatalog]:
    """The HANDOFF Phase A candidate filter, applied at the DB layer (a fresh
    read at decision time — promotion is rare enough that a live query beats
    trusting the possibly-15s-stale in-process catalog snapshot). All must
    hold: `enabled`, `status="live"`, `last_live_at` within `max_stale_min`,
    `ttfb_ms <= max_ttfb_ms`, `tool_ok=True`. `exclude_ids` is the caller's
    job — the embedder, the failing model itself, and every id already
    serving ANOTHER role. Ordered by `ttfb_ms` ascending (fastest first)."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_stale_min)
    stmt = (
        select(ModelCatalog)
        .where(
            ModelCatalog.enabled.is_(True),
            ModelCatalog.status == "live",
            ModelCatalog.last_live_at.isnot(None),
            ModelCatalog.last_live_at >= cutoff,
            ModelCatalog.ttfb_ms.isnot(None),
            ModelCatalog.ttfb_ms <= max_ttfb_ms,
            ModelCatalog.tool_ok.is_(True),
        )
        .order_by(ModelCatalog.ttfb_ms.asc())
    )
    result = await db.execute(stmt)
    return [r for r in result.scalars().all() if r.id not in exclude_ids]


async def list_refresh_candidates(
    db: AsyncSession,
    *,
    exclude_ids: set[str],
    limit: int,
) -> list[ModelCatalog]:
    """The on-demand-refresh candidate set (root follow-up, 2026-09-26):
    every ENABLED catalog row not already excluded, ordered by last-known
    `ttfb_ms` ascending (Postgres puts NULLs last on plain ASC — a
    never-verified row sorts after any row with a known, presumably decent,
    ttfb), capped at `limit`. Deliberately does NOT filter on `status`/
    `last_live_at`/`tool_ok` the way `list_promotion_candidates` does —
    refreshing exactly those stale signals is the point of this query;
    `llm.catalog.promotion._refresh_stale_candidates` re-probes each row
    with `probe_ttfb`/`verify_model` and `list_promotion_candidates` is
    re-run afterward to see if anything now qualifies."""
    stmt = (
        select(ModelCatalog)
        .where(ModelCatalog.enabled.is_(True))
        .order_by(ModelCatalog.ttfb_ms.asc())
    )
    result = await db.execute(stmt)
    rows = [r for r in result.scalars().all() if r.id not in exclude_ids]
    return rows[:limit]


async def diagnose_candidates(
    db: AsyncSession,
    *,
    exclude_ids: set[str],
    max_ttfb_ms: int,
    max_stale_min: int,
    limit: int = 5,
) -> list[tuple[str, str]]:
    """Per-row exclusion reasons for the candidate filter (root observability
    follow-up, 2026-09-27) — before this, "no eligible candidate" gave no
    reason and the only way to find out was querying Postgres by hand (root
    live-stack finding: muse-glimmer looked live/tool_ok/fresh and STILL
    didn't qualify — the reason, `ttfb_ms IS NULL`, was invisible from the
    logs). Checked in the same priority order `list_promotion_candidates`'s
    WHERE clause implies; each row reports the FIRST reason that applies:
    "disabled" / "excluded" (serving another role, the embedder, or the
    failing model itself) / "stale" (not live or last_live_at too old) /
    "no_ttfb" (never measured or measured-and-failed — see
    `ttfb_fail_reason`) / "ttfb_too_high" / "tool_ok_false". Enabled rows are
    listed before disabled ones (the ones worth investigating), each ordered
    by `ttfb_ms` ascending within that. Capped at `limit` — this is a
    diagnostic log line on an already-rare, already-slow path, not a report."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_stale_min)
    stmt = select(ModelCatalog).order_by(ModelCatalog.enabled.desc(), ModelCatalog.ttfb_ms.asc())
    result = await db.execute(stmt)

    out: list[tuple[str, str]] = []
    for row in result.scalars().all():
        if len(out) >= limit:
            break
        if not row.enabled:
            reason = "disabled"
        elif row.id in exclude_ids:
            reason = "excluded"
        elif row.status != "live" or row.last_live_at is None or row.last_live_at < cutoff:
            reason = "stale"
        elif row.ttfb_ms is None:
            reason = f"no_ttfb:{row.ttfb_fail_reason}" if row.ttfb_fail_reason else "no_ttfb:never_probed"
        elif row.ttfb_ms > max_ttfb_ms:
            reason = "ttfb_too_high"
        elif not row.tool_ok:
            reason = "tool_ok_false"
        else:
            continue  # this row actually qualifies — nothing to report
        out.append((row.id, reason))
    return out
