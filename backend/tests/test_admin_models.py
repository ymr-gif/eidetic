"""Admin model-catalog endpoints (HANDOFF Phase 3): GET/PATCH /admin/models,
POST /admin/models/rescan. Unit tier — FastAPI TestClient with the DB session
and auth dependency overridden (mirrors tests/test_notifications.py);
llm.catalog.store / scanner / cache.publish are monkeypatched module
attributes on api.admin.models so no real DB/Redis/NIM is touched.
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

pytest.importorskip("arq")  # host has no arq; api.admin.__init__ -> .models -> core.arq_pool needs it (runs in-container/CI)

import config
import api.admin.models as admin_models  # noqa: E402
from auth.security import get_current_user
from core.db import get_db
from models import ModelCatalog, User

ADMIN = User(id=1, username="admin", role="admin", is_active=True)
PLAIN = User(id=2, username="user", role="user", is_active=True)


def _make_client(db_mock, *, user=ADMIN):
    app = FastAPI()
    app.include_router(admin_models.router, prefix="/admin")

    async def _fake_user():
        return user

    async def _fake_get_db():
        yield db_mock

    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_db] = _fake_get_db
    return TestClient(app)


def _mock_db():
    """AsyncMock(spec=object) rejects .add()/.commit() (not on `object`) — every
    route here goes through _audit(), which calls db.add()."""
    db = AsyncMock(spec=object)
    db.add = MagicMock()
    db.commit = AsyncMock()
    return db


def _row(**overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        id="z-ai/glm-5.3-flash", label="Glm 5.3 Flash", status="live", http_status=200,
        latency_ms=120, fail_count=0, enabled=False, price_in=None, price_out=None,
        context_window=None, supports_tools=None, reasoning=None, request_extras=None,
        min_max_tokens=None, last_checked=now, first_seen=now, updated_at=now,
    )
    defaults.update(overrides)
    return ModelCatalog(**defaults)


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setattr(admin_models, "catalog_cache", MagicMock(publish=AsyncMock()))
    monkeypatch.setattr(config, "MODELS", {"llama": "openai/gpt-oss-20b"}, raising=False)
    monkeypatch.setattr(config, "LLM_BACKEND", "nim", raising=False)
    yield


class TestListCatalog:
    def test_requires_admin(self):
        mock_db = _mock_db()
        client = _make_client(mock_db, user=PLAIN)
        resp = client.get("/admin/models", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 403

    def test_returns_rows_plus_scan_meta(self, monkeypatch):
        rows = [_row()]

        async def _fake_list_rows(db, **kwargs):
            return rows

        async def _fake_get_scan_meta():
            return {"ran_at": "2026-09-20T00:00:00+00:00", "scanned": 5}

        monkeypatch.setattr(admin_models, "list_rows", _fake_list_rows)
        monkeypatch.setattr(admin_models, "get_scan_meta", _fake_get_scan_meta)

        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.get("/admin/models", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"][0]["id"] == "z-ai/glm-5.3-flash"
        assert body["models"][0]["is_role_model"] is False
        assert body["scan_meta"]["scanned"] == 5


class TestPatchCatalogModel:
    def test_404_for_unknown_model(self, monkeypatch):
        async def _fake_get_row(db, model_id):
            return None

        monkeypatch.setattr(admin_models, "get_row", _fake_get_row)
        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.patch("/admin/models/nope/nothing", json={"enabled": True},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 404

    def test_enable_and_price_pair_and_publishes(self, monkeypatch):
        row = _row(enabled=False)

        async def _fake_get_row(db, model_id):
            assert model_id == "z-ai/glm-5.3-flash"
            return row

        monkeypatch.setattr(admin_models, "get_row", _fake_get_row)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.patch(
            "/admin/models/z-ai/glm-5.3-flash",
            json={"enabled": True, "price_in": 1.0, "price_out": 2.0},
            headers={"Authorization": "Bearer x"},
        )
        assert resp.status_code == 200
        assert row.enabled is True
        assert row.price_in == 1.0
        assert row.price_out == 2.0
        admin_models.catalog_cache.publish.assert_awaited_once()

    def test_price_pair_must_be_set_together(self, monkeypatch):
        row = _row()

        async def _fake_get_row(db, model_id):
            return row

        monkeypatch.setattr(admin_models, "get_row", _fake_get_row)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.patch("/admin/models/z-ai/glm-5.3-flash", json={"price_in": 1.0},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 400

    def test_disabling_a_role_model_is_409(self, monkeypatch):
        row = _row(id="openai/gpt-oss-20b", enabled=True)

        async def _fake_get_row(db, model_id):
            return row

        monkeypatch.setattr(admin_models, "get_row", _fake_get_row)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.patch("/admin/models/openai/gpt-oss-20b", json={"enabled": False},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 409

    def test_request_extras_key_allowlist_enforced(self, monkeypatch):
        row = _row()

        async def _fake_get_row(db, model_id):
            return row

        monkeypatch.setattr(admin_models, "get_row", _fake_get_row)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.patch(
            "/admin/models/z-ai/glm-5.3-flash",
            json={"request_extras": {"some_random_field": True}},
            headers={"Authorization": "Bearer x"},
        )
        assert resp.status_code == 400

    def test_request_extras_allowed_key_accepted(self, monkeypatch):
        row = _row()

        async def _fake_get_row(db, model_id):
            return row

        monkeypatch.setattr(admin_models, "get_row", _fake_get_row)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.patch(
            "/admin/models/z-ai/glm-5.3-flash",
            json={"request_extras": {"reasoning_effort": "low"}},
            headers={"Authorization": "Bearer x"},
        )
        assert resp.status_code == 200
        assert row.request_extras == {"reasoning_effort": "low"}

    def test_min_max_tokens_bounds_enforced_by_schema(self, monkeypatch):
        row = _row()

        async def _fake_get_row(db, model_id):
            return row

        monkeypatch.setattr(admin_models, "get_row", _fake_get_row)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.patch("/admin/models/z-ai/glm-5.3-flash", json={"min_max_tokens": 999999},
                             headers={"Authorization": "Bearer x"})
        assert resp.status_code == 422  # pydantic Field(le=4096)


class TestRescan:
    def test_homeserver_mode_400(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_BACKEND", "homeserver", raising=False)
        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.post("/admin/models/rescan", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 400

    def test_already_running_409(self, monkeypatch):
        async def _running():
            return True

        monkeypatch.setattr(admin_models, "is_scan_running", _running)
        mock_db = _mock_db()
        client = _make_client(mock_db)
        resp = client.post("/admin/models/rescan", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 409

    def test_queued_202_via_arq_pool(self, monkeypatch):
        async def _not_running():
            return False

        pool = MagicMock()
        pool.enqueue_job = AsyncMock()

        monkeypatch.setattr(admin_models, "is_scan_running", _not_running)
        monkeypatch.setattr(admin_models, "get_arq_pool", lambda: pool)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.post("/admin/models/rescan", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 202
        assert resp.json() == {"queued": True}
        pool.enqueue_job.assert_awaited_once_with("scan_model_catalog_job", trigger="manual")

    def test_queued_202_inline_when_no_arq_pool(self, monkeypatch):
        async def _not_running():
            return False

        async def _fake_run_scan(trigger):
            return {"scanned": 0}

        monkeypatch.setattr(admin_models, "is_scan_running", _not_running)
        monkeypatch.setattr(admin_models, "get_arq_pool", lambda: None)
        monkeypatch.setattr(admin_models, "run_scan", _fake_run_scan)
        mock_db = _mock_db()
        client = _make_client(mock_db)

        resp = client.post("/admin/models/rescan", headers={"Authorization": "Bearer x"})
        assert resp.status_code == 202
        assert resp.json() == {"queued": True}
