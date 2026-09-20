"""llm/catalog/store.py — model_catalog row persistence (HANDOFF Phase 3 + 7).

Unit tier — a minimal fake AsyncSession stands in for Postgres (`.get()`
returns a pre-seeded row or None, `.add()` records the new row). Covers the
Phase 7 addition: `upsert_scan_result` sets `last_live_at` whenever a probe
returns status="live" (new row or existing), and never clears/changes it on
any other outcome.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest

from llm.catalog import store
from models.catalog import ModelCatalog


class _FakeSession:
    def __init__(self, existing: ModelCatalog | None = None):
        self._existing = existing
        self.added: list[ModelCatalog] = []

    async def get(self, model, model_id):
        return self._existing

    def add(self, row):
        self.added.append(row)


def _existing_row(**overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        id="vendor/model", label="Model", status="live", http_status=200,
        latency_ms=100, fail_count=0, enabled=True, last_live_at=None,
        first_seen=now, last_checked=now, updated_at=now,
    )
    defaults.update(overrides)
    return ModelCatalog(**defaults)


class TestUpsertScanResultLastLiveAt:
    @pytest.mark.asyncio
    async def test_new_row_live_sets_last_live_at(self):
        db = _FakeSession(existing=None)
        row = await store.upsert_scan_result(
            db, "vendor/model", status="live", http_status=200, latency_ms=100,
            label="Model", seed_enabled=False,
        )
        assert row.last_live_at is not None
        assert db.added == [row]

    @pytest.mark.asyncio
    async def test_new_row_timeout_leaves_last_live_at_none(self):
        db = _FakeSession(existing=None)
        row = await store.upsert_scan_result(
            db, "vendor/model", status="timeout", http_status=None, latency_ms=None,
            label="Model", seed_enabled=False,
        )
        assert row.last_live_at is None

    @pytest.mark.asyncio
    async def test_new_row_error_leaves_last_live_at_none(self):
        db = _FakeSession(existing=None)
        row = await store.upsert_scan_result(
            db, "vendor/model", status="error", http_status=500, latency_ms=10,
            label="Model", seed_enabled=False,
        )
        assert row.last_live_at is None

    @pytest.mark.asyncio
    async def test_existing_never_live_row_goes_live_sets_last_live_at(self):
        existing = _existing_row(status="timeout", fail_count=1, last_live_at=None)
        db = _FakeSession(existing=existing)
        row = await store.upsert_scan_result(
            db, "vendor/model", status="live", http_status=200, latency_ms=90,
            label=None, seed_enabled=False,
        )
        assert row is existing
        assert row.last_live_at is not None
        assert row.fail_count == 0

    @pytest.mark.asyncio
    async def test_existing_live_row_stays_live_refreshes_last_live_at(self):
        old = datetime(2026, 9, 1, tzinfo=timezone.utc)
        existing = _existing_row(status="live", fail_count=0, last_live_at=old)
        db = _FakeSession(existing=existing)
        row = await store.upsert_scan_result(
            db, "vendor/model", status="live", http_status=200, latency_ms=90,
            label=None, seed_enabled=False,
        )
        assert row.last_live_at is not None
        assert row.last_live_at > old

    @pytest.mark.asyncio
    async def test_existing_live_row_blips_to_timeout_keeps_prior_last_live_at(self):
        """The core Phase 7 invariant: last_live_at is a high-water mark, never
        cleared by a subsequent transient failure."""
        old = datetime(2026, 9, 1, tzinfo=timezone.utc)
        existing = _existing_row(status="live", fail_count=0, last_live_at=old)
        db = _FakeSession(existing=existing)
        row = await store.upsert_scan_result(
            db, "vendor/model", status="timeout", http_status=None, latency_ms=None,
            label=None, seed_enabled=False,
        )
        assert row.last_live_at == old
        assert row.fail_count == 1

    @pytest.mark.asyncio
    async def test_existing_never_live_row_stays_never_live_on_second_failure(self):
        existing = _existing_row(status="timeout", fail_count=1, last_live_at=None)
        db = _FakeSession(existing=existing)
        row = await store.upsert_scan_result(
            db, "vendor/model", status="error", http_status=502, latency_ms=None,
            label=None, seed_enabled=False,
        )
        assert row.last_live_at is None
        assert row.fail_count == 2
