"""Admin role-override endpoints (HANDOFF Phase A): GET/PATCH
/admin/models/roles/{role}. Unit tier — FastAPI TestClient with the DB
session and auth dependency overridden (mirrors tests/test_admin_models.py);
llm.catalog.role_store / role_state / promotion are monkeypatched module
attributes on api.admin.model_roles so no real DB/Redis is touched.
"""
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

pytest.importorskip("arq")  # importing api.admin.model_roles pulls in the api.admin package (-> .models -> core.arq_pool)

import config
import api.admin.model_roles as admin_roles  # noqa: E402
from auth.security import get_current_user
from core.db import get_db
from models import ModelRoleOverride, User

ADMIN = User(id=1, username="admin", role="admin", is_active=True)
PLAIN = User(id=2, username="user", role="user", is_active=True)

LLAMA     = "vendor/llama-base"
CODER     = "vendor/coder-base"
REASONING = "vendor/reasoning-base"


def _make_client(db_mock, *, user=ADMIN):
    app = FastAPI()
    app.include_router(admin_roles.router, prefix="/admin")

    async def _fake_user():
        return user

    async def _fake_get_db():
        yield db_mock

    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_db] = _fake_get_db
    return TestClient(app)


def _mock_db():
    db = AsyncMock(spec=object)
    db.add = MagicMock()
    db.commit = AsyncMock()
    return db


def _override_row(**overrides):
    defaults = dict(role="llama", model_id="vendor/promoted", pinned=False,
                     reason="auto", base_model_id=LLAMA,
                     promoted_at=datetime.now(timezone.utc))
    defaults.update(overrides)
    return ModelRoleOverride(**defaults)


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setattr(config, "MODELS", {"llama": LLAMA, "coder": CODER, "reasoning": REASONING}, raising=False)
    monkeypatch.setattr(admin_roles, "role_state", MagicMock(ensure_fresh=AsyncMock(), publish=AsyncMock()))
    yield


class TestListRoleOverrides:
    def test_requires_admin(self):
        mock_db = _mock_db()
        client = _make_client(mock_db, user=PLAIN)
        resp = client.get("/admin/models/roles", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 403

    def test_no_overrides_returns_base_for_every_role(self, monkeypatch):
        async def _fake_get(db, role):
            return None
        monkeypatch.setattr(admin_roles, "get_role_override", _fake_get)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.get("/admin/models/roles", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200
        roles = {r["role"]: r for r in resp.json()["roles"]}
        assert roles["llama"] == {"role": "llama", "base": LLAMA, "effective": LLAMA, "pinned": False,
                                   "promoted_at": None, "reason": None}
        assert set(roles) == {"llama", "coder", "reasoning"}

    def test_promoted_role_shows_effective_id(self, monkeypatch):
        row = _override_row()

        async def _fake_get(db, role):
            return row if role == "llama" else None
        monkeypatch.setattr(admin_roles, "get_role_override", _fake_get)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.get("/admin/models/roles", headers={"Authorization": "Bearer x"})
        roles = {r["role"]: r for r in resp.json()["roles"]}
        assert roles["llama"]["effective"] == "vendor/promoted"
        assert roles["llama"]["reason"] == "auto"
        assert roles["llama"]["pinned"] is False


class TestPatchRoleOverride:
    def test_requires_admin(self):
        mock_db = _mock_db()
        client = _make_client(mock_db, user=PLAIN)
        resp = client.patch("/admin/models/roles/llama", json={"model_id": "vendor/x"},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 403

    def test_unknown_role_404s(self):
        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.patch("/admin/models/roles/not-a-role", json={"model_id": "vendor/x"},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 404

    def test_assign_a_model_resolves_strict_and_pins(self, monkeypatch):
        set_calls = []

        async def _fake_set_manual(db, role, model_id, *, reason="manual"):
            set_calls.append((role, model_id, reason))
            return {"role": role, "model_id": model_id, "pinned": True}
        monkeypatch.setattr(admin_roles, "set_manual_override", _fake_set_manual)

        def _fake_resolve_strict(name):
            return name  # pretend it always resolves — resolve_model_strict is SYNC
        monkeypatch.setattr("api.chat.model_resolve.resolve_model_strict", _fake_resolve_strict)

        async def _fake_get(db, role):
            return _override_row(model_id="vendor/picked", pinned=True, reason="manual")
        monkeypatch.setattr(admin_roles, "get_role_override", _fake_get)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.patch("/admin/models/roles/llama", json={"model_id": "vendor/picked"},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200
        assert set_calls == [("llama", "vendor/picked", "manual")]
        assert resp.json()["pinned"] is True
        admin_roles.role_state.publish.assert_awaited_once()

    def test_assign_unresolvable_model_is_422(self, monkeypatch):
        from fastapi import HTTPException

        def _fake_resolve_strict(name):  # resolve_model_strict is SYNC
            raise HTTPException(status_code=422, detail={"error": "model_unavailable", "model": name})
        monkeypatch.setattr("api.chat.model_resolve.resolve_model_strict", _fake_resolve_strict)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.patch("/admin/models/roles/llama", json={"model_id": "nobody/made-this-up"},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 422

    def test_clear_override(self, monkeypatch):
        cleared = []

        async def _fake_clear(db, role):
            cleared.append(role)
        monkeypatch.setattr(admin_roles, "clear_manual_override", _fake_clear)

        async def _fake_get(db, role):
            return None
        monkeypatch.setattr(admin_roles, "get_role_override", _fake_get)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.patch("/admin/models/roles/llama", json={"model_id": None},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200
        assert cleared == ["llama"]
        assert resp.json()["effective"] == LLAMA
        admin_roles.role_state.publish.assert_awaited_once()

    def test_pin_only_does_not_touch_model_id(self, monkeypatch):
        pin_calls = []

        async def _fake_pin(db, role, pinned):
            pin_calls.append((role, pinned))
            return {"role": role, "model_id": "vendor/existing", "pinned": pinned}
        monkeypatch.setattr(admin_roles, "set_role_pin", _fake_pin)

        async def _fake_get(db, role):
            return _override_row(model_id="vendor/existing", pinned=True, reason="auto")
        monkeypatch.setattr(admin_roles, "get_role_override", _fake_get)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.patch("/admin/models/roles/llama", json={"pinned": True},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200
        assert pin_calls == [("llama", True)]

    def test_empty_body_is_a_no_op_returns_current_state(self, monkeypatch):
        async def _fake_get(db, role):
            return None
        monkeypatch.setattr(admin_roles, "get_role_override", _fake_get)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.patch("/admin/models/roles/llama", json={},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200
        assert resp.json()["effective"] == LLAMA
        admin_roles.role_state.publish.assert_not_awaited()
        mock_db.commit.assert_not_awaited()
