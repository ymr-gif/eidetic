"""GET /api/models (HANDOFF Phase 3) — the model picker's data source: the 3
named roles plus every currently-AVAILABLE live-catalog model. Any logged-in
user (no admin gate — everyone needs this to pick a model in the ⌘K palette,
compare picker, or conversation-lock select); the admin-only curation surface
is api/admin/models.py.
"""
from fastapi import APIRouter, Depends

import config
from auth.security import get_current_user
from llm.catalog import cache as catalog_cache
from llm.catalog import role_state
from llm.catalog.promotion import effective_role_model
from models import User

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_models(current_user: User = Depends(get_current_user)):
    await catalog_cache.ensure_fresh()
    await role_state.ensure_fresh()

    role_ids = set(config.MODELS.values())
    out = []

    for role, model_id in config.MODELS.items():
        entry = catalog_cache.get_entry(model_id) or {}
        # HANDOFF Phase A: `effective`/`promoted` are always present (default
        # to the base id / False) so the shape is stable whether or not
        # auto-promotion is enabled — effective_role_model is a no-op when
        # the flag is off.
        effective_id = effective_role_model(role)
        out.append({
            "id":             model_id,
            "label":          entry.get("label") or role.capitalize(),
            "role":           role,
            "status":         entry.get("status", "live"),
            "latency_ms":     entry.get("latency_ms"),
            "context_window": config.CONTEXT_WINDOWS.get(model_id) or entry.get("context_window"),
            "effective":      effective_id,
            "promoted":       effective_id != model_id,
        })

    for model_id in catalog_cache.available_ids():
        if model_id in role_ids:
            continue  # already listed above with its role
        entry = catalog_cache.get_entry(model_id) or {}
        out.append({
            "id":             model_id,
            "label":          entry.get("label") or model_id,
            "role":           None,
            "status":         entry.get("status", "live"),
            "latency_ms":     entry.get("latency_ms"),
            "context_window": entry.get("context_window"),
        })

    return {"models": out}
