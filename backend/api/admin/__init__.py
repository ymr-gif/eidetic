from .users import router as _users_router
from .audit import router as _audit_router
from .env import router as _env_router
from .system import router as _system_router
from .memory import router as _memory_router
from .model_roles import router as _model_roles_router
from .models import router as _models_router
from fastapi import APIRouter

router = APIRouter(prefix="/admin", tags=["admin"])
router.include_router(_users_router)
router.include_router(_audit_router)
router.include_router(_env_router)
router.include_router(_system_router)
router.include_router(_memory_router)
# _model_roles_router BEFORE _models_router: its `/models/roles/{role}` PATCH
# must be matched before _models_router's catch-all `/models/{model_id:path}`
# PATCH (see api/admin/model_roles.py's module docstring).
router.include_router(_model_roles_router)
router.include_router(_models_router)
