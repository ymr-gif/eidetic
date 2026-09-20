"""Admin model-catalog curation (HANDOFF Phase 3): review scan results, enable/
disable + price/context-window override individual models, trigger a manual
Rescan. Same require_role("admin") + _audit() pattern as the rest of
api/admin/*.py.

HANDOFF Phase 7: enabling a model that has never once answered a live probe
(`last_live_at IS NULL` — migration 051) is refused with 409, `detail="never_live"`
(bare string — frontend does an exact match). Disabling is always allowed
regardless. The row is still returned by GET either way — this only guards
the enable action, never visibility.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

import config
from auth.security import require_role
from core.arq_pool import get_arq_pool
from core.db import get_db
from models import ModelCatalog, User
from llm.catalog import cache as catalog_cache
from llm.catalog.scanner import get_scan_meta, is_scan_running, run_scan
from llm.catalog.store import get_row, list_rows

from .utils import _audit

logger = logging.getLogger("admin.models")

router = APIRouter()

# request_extras is admin-editable but must never carry an unverified field to
# NIM — same allowlist the config.py MODEL_REQUEST_EXTRAS table's two known
# fields use (see backend/CLAUDE.md "reasoning-model request budget hotfix").
_EXTRAS_ALLOWED_KEYS = {"reasoning_effort", "chat_template_kwargs"}
_EXTRAS_MAX_BYTES = 1024


def _row_out(row: ModelCatalog) -> dict:
    return {
        "id":              row.id,
        "label":           row.label,
        "status":          row.status,
        "http_status":     row.http_status,
        "latency_ms":      row.latency_ms,
        "fail_count":      row.fail_count,
        "enabled":         row.enabled,
        "price_in":        row.price_in,
        "price_out":       row.price_out,
        "context_window":  row.context_window,
        "supports_tools":  row.supports_tools,
        "reasoning":       row.reasoning,
        "request_extras":  row.request_extras,
        "min_max_tokens":  row.min_max_tokens,
        "last_live_at":    row.last_live_at.isoformat() if row.last_live_at else None,
        "last_checked":    row.last_checked.isoformat() if row.last_checked else None,
        "first_seen":      row.first_seen.isoformat()   if row.first_seen   else None,
        "updated_at":      row.updated_at.isoformat()   if row.updated_at   else None,
        "is_role_model":   row.id in set(config.MODELS.values()),
    }


@router.get("/models")
async def list_catalog(
    q:      str | None = None,
    status: str | None = None,
    admin:  User        = Depends(require_role("admin")),
    db:     AsyncSession = Depends(get_db),
):
    rows = await list_rows(db, q=q, status=status)
    meta = await get_scan_meta()
    return {"models": [_row_out(r) for r in rows], "scan_meta": meta}


class ModelPatch(BaseModel):
    enabled:        bool | None  = None
    label:          str | None   = None
    price_in:       float | None = Field(None, ge=0, le=100)
    price_out:      float | None = Field(None, ge=0, le=100)
    context_window: int | None   = Field(None, ge=1)
    request_extras: dict | None  = None
    min_max_tokens: int | None   = Field(None, ge=1, le=4096)


@router.patch("/models/{model_id:path}")
async def patch_catalog_model(
    model_id: str,
    body:     ModelPatch,
    admin:    User        = Depends(require_role("admin")),
    db:       AsyncSession = Depends(get_db),
):
    row = await get_row(db, model_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Model not found in catalog")

    updated = body.model_dump(exclude_unset=True)
    if not updated:
        return _row_out(row)

    if updated.get("enabled") is False and model_id in set(config.MODELS.values()):
        raise HTTPException(status_code=409, detail="Cannot disable a role model (llama/coder/reasoning)")

    # HANDOFF Phase 7: never offer a model that was never live. Enabling is
    # blocked (disabling never is — that's always safe) when the row has
    # never once answered a probe with status="live". A row currently
    # status="live" always has last_live_at set (store.py sets it on every
    # live probe; migration 051 backfilled pre-existing live rows), so this
    # only fires for not_found/gone/delisted/timeout/error ids that have
    # NEVER been live — exactly the repro (timeout, fail_count=1, enabled).
    # `detail` is a bare string ("never_live"), matching this endpoint's own
    # existing 409 precedent (the role-model-disable check two lines up) —
    # the frontend does an exact `data.detail === 'never_live'` match and
    # supplies its own user-facing copy, same as it already does for the
    # role-model-disable 409.
    if updated.get("enabled") is True and row.last_live_at is None:
        raise HTTPException(status_code=409, detail="never_live")

    if ("price_in" in updated) != ("price_out" in updated):
        raise HTTPException(status_code=400, detail="price_in and price_out must be set together")

    if "request_extras" in updated and updated["request_extras"] is not None:
        extras = updated["request_extras"]
        if not isinstance(extras, dict) or not set(extras.keys()) <= _EXTRAS_ALLOWED_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"request_extras must be a JSON object with keys from {sorted(_EXTRAS_ALLOWED_KEYS)}",
            )
        if len(json.dumps(extras)) > _EXTRAS_MAX_BYTES:
            raise HTTPException(status_code=400, detail=f"request_extras must be <= {_EXTRAS_MAX_BYTES} bytes")

    for key, value in updated.items():
        setattr(row, key, value)
    row.updated_at = datetime.now(timezone.utc)

    await _audit(db, admin, "model_catalog.updated", detail={"model_id": model_id, "changes": updated})
    await db.commit()
    await catalog_cache.publish()

    return _row_out(row)


@router.post("/models/rescan", status_code=202)
async def rescan_models(
    admin: User        = Depends(require_role("admin")),
    db:    AsyncSession = Depends(get_db),
):
    if config.LLM_BACKEND == "homeserver":
        raise HTTPException(status_code=400, detail="Model catalog is inert in homeserver mode")
    if await is_scan_running():
        raise HTTPException(status_code=409, detail="A scan is already running")

    await _audit(db, admin, "model_catalog.rescan", detail={})
    await db.commit()

    pool = get_arq_pool()
    if pool:
        await pool.enqueue_job("scan_model_catalog_job", trigger="manual")
    else:
        import asyncio
        asyncio.create_task(run_scan(trigger="manual"))

    return {"queued": True}
