"""Admin role-override curation (HANDOFF Phase A auto-promotion). Split out
of api/admin/models.py (already near the 200-line convention boundary) as
its own sub-router — same require_role("admin") + _audit() pattern as every
other api/admin/*.py file, mounted alongside it in api/admin/__init__.py.

Registered BEFORE api/admin/models.py's `PATCH /models/{model_id:path}` in
api/admin/__init__.py's include_router order: Starlette matches routes in
registration order across the WHOLE app regardless of which file/sub-router
they came from, and that catch-all path converter would otherwise swallow
`/models/roles/{role}` as a literal model id containing a slash.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

import config
from auth.security import require_role
from core.db import get_db
from models import ModelRoleOverride, User
from llm.catalog import role_state
from llm.catalog.promotion import clear_manual_override, set_manual_override, set_role_pin
from llm.catalog.role_store import get_role_override

from .utils import _audit

router = APIRouter()


def _role_row_out(role: str, override: ModelRoleOverride | None) -> dict:
    base = config.MODELS[role]
    return {
        "role":        role,
        "base":        base,
        "effective":   override.model_id if override else base,
        "pinned":      bool(override.pinned) if override else False,
        "promoted_at": override.promoted_at.isoformat() if override and override.promoted_at else None,
        "reason":      override.reason if override else None,
    }


@router.get("/models/roles")
async def list_role_overrides(
    admin: User        = Depends(require_role("admin")),
    db:    AsyncSession = Depends(get_db),
):
    await role_state.ensure_fresh()
    return {"roles": [_role_row_out(role, await get_role_override(db, role)) for role in config.MODELS]}


class RoleOverridePatch(BaseModel):
    # Both fields optional and independently settable — `model_id` present
    # (even as null) drives an assign/clear; `pinned` alone (no `model_id`
    # key) just toggles the pin without touching which model serves the
    # role. See llm/catalog/promotion.py's set_manual_override/set_role_pin.
    model_id: str | None = None
    pinned:   bool | None = None


@router.patch("/models/roles/{role}")
async def patch_role_override(
    role:  str,
    body:  RoleOverridePatch,
    admin: User        = Depends(require_role("admin")),
    db:    AsyncSession = Depends(get_db),
):
    if role not in config.MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown role '{role}'")

    updated = body.model_dump(exclude_unset=True)
    if "model_id" in updated:
        model_id = updated["model_id"]
        if model_id is None:
            await clear_manual_override(db, role)
            await _audit(db, admin, "model_catalog.role_override_cleared", detail={"role": role})
        else:
            # Lazy import — avoids a module-load-time cycle with api.chat.*
            # (api/chat/model_resolve.py has no reason to import api.admin).
            from api.chat.model_resolve import resolve_model_strict
            resolved = resolve_model_strict(model_id)  # 422 model_unavailable on a bad id
            await set_manual_override(db, role, resolved)
            await _audit(db, admin, "model_catalog.role_override_set",
                         detail={"role": role, "model_id": resolved})
    elif "pinned" in updated:
        await set_role_pin(db, role, updated["pinned"])
        await _audit(db, admin, "model_catalog.role_pin_toggled",
                      detail={"role": role, "pinned": updated["pinned"]})

    if updated:
        await db.commit()
        await role_state.publish()

    return _role_row_out(role, await get_role_override(db, role))
