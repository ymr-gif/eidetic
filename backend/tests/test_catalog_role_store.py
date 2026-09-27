"""llm/catalog/role_store.py — model_role_override row CRUD + promotion
candidate selection (HANDOFF Phase A). Unit tier — a minimal fake
AsyncSession stands in for Postgres (mirrors tests/test_catalog_store.py's
own pattern). `list_promotion_candidates`'s SQL WHERE predicate itself
(enabled/status/last_live_at/ttfb_ms/tool_ok) is SQLAlchemy-declarative and
not meaningfully exercised without a real Postgres connection (infra tier);
these tests cover the one piece of real Python logic in that function — the
`exclude_ids` post-filter — by feeding a fake `execute()` a pre-set row list.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest

from llm.catalog import role_store
from models.catalog import ModelCatalog, ModelRoleOverride


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, *, get_map=None, execute_rows=None):
        self._get_map = get_map or {}
        self._execute_rows = execute_rows or []
        self.added: list = []
        self.deleted: list = []

    async def get(self, model, key):
        return self._get_map.get(key)

    def add(self, row):
        self.added.append(row)
        # so a subsequent get() sees what was just added, matching real
        # SQLAlchemy identity-map behavior closely enough for these tests
        if isinstance(row, ModelRoleOverride):
            self._get_map[row.role] = row

    async def delete(self, row):
        self.deleted.append(row)
        self._get_map = {k: v for k, v in self._get_map.items() if v is not row}

    async def execute(self, stmt):
        return _FakeResult(self._execute_rows)


class TestRoleOverrideCRUD:
    @pytest.mark.asyncio
    async def test_get_role_override_none_when_absent(self):
        db = _FakeSession()
        assert await role_store.get_role_override(db, "llama") is None

    @pytest.mark.asyncio
    async def test_set_role_override_creates_new_row(self):
        db = _FakeSession()
        row = await role_store.set_role_override(
            db, "llama", model_id="vendor/candidate", pinned=False, reason="auto",
            base_model_id="vendor/base",
        )
        assert row.role == "llama"
        assert row.model_id == "vendor/candidate"
        assert row.pinned is False
        assert row.reason == "auto"
        assert row.base_model_id == "vendor/base"
        assert row.promoted_at is not None
        assert db.added == [row]

    @pytest.mark.asyncio
    async def test_set_role_override_overwrites_existing_row_in_place(self):
        existing = ModelRoleOverride(role="llama", model_id="vendor/old", pinned=False,
                                      reason="auto", base_model_id="vendor/base")
        db = _FakeSession(get_map={"llama": existing})
        row = await role_store.set_role_override(
            db, "llama", model_id="vendor/new", pinned=True, reason="manual",
            base_model_id="vendor/base",
        )
        assert row is existing
        assert row.model_id == "vendor/new"
        assert row.pinned is True
        assert row.reason == "manual"
        assert db.added == []  # no new row — same object updated

    @pytest.mark.asyncio
    async def test_clear_role_override_returns_true_and_deletes_when_present(self):
        existing = ModelRoleOverride(role="llama", model_id="vendor/x", pinned=False,
                                      reason="auto", base_model_id="vendor/base")
        db = _FakeSession(get_map={"llama": existing})
        result = await role_store.clear_role_override(db, "llama")
        assert result is True
        assert db.deleted == [existing]
        assert await role_store.get_role_override(db, "llama") is None

    @pytest.mark.asyncio
    async def test_clear_role_override_returns_false_when_absent(self):
        db = _FakeSession()
        result = await role_store.clear_role_override(db, "llama")
        assert result is False
        assert db.deleted == []

    @pytest.mark.asyncio
    async def test_list_role_overrides_returns_all_rows(self):
        rows = [
            ModelRoleOverride(role="llama", model_id="a", pinned=False, reason="auto", base_model_id="base-a"),
            ModelRoleOverride(role="coder", model_id="b", pinned=True, reason="manual", base_model_id="base-b"),
        ]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.list_role_overrides(db)
        assert result == rows


def _candidate(id_, ttfb_ms=100):
    return ModelCatalog(id=id_, status="live", enabled=True, ttfb_ms=ttfb_ms, tool_ok=True,
                         last_live_at=datetime.now(timezone.utc))


class TestListPromotionCandidates:
    @pytest.mark.asyncio
    async def test_excludes_named_ids(self):
        rows = [_candidate("a/model"), _candidate("b/model"), _candidate("c/model")]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.list_promotion_candidates(
            db, exclude_ids={"b/model"}, max_ttfb_ms=5000, max_stale_min=30,
        )
        assert {r.id for r in result} == {"a/model", "c/model"}

    @pytest.mark.asyncio
    async def test_empty_exclude_set_keeps_everything_the_query_returned(self):
        rows = [_candidate("a/model"), _candidate("b/model")]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.list_promotion_candidates(
            db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30,
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_excluding_every_candidate_returns_empty(self):
        rows = [_candidate("a/model")]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.list_promotion_candidates(
            db, exclude_ids={"a/model"}, max_ttfb_ms=5000, max_stale_min=30,
        )
        assert result == []


class TestListRefreshCandidates:
    """The on-demand-refresh query (root follow-up) — no status/last_live_at/
    tool_ok filtering (that's the whole point: it targets STALE rows), just
    enabled + exclude_ids + a cap."""

    @pytest.mark.asyncio
    async def test_excludes_named_ids(self):
        rows = [_candidate("a/model"), _candidate("b/model"), _candidate("c/model")]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.list_refresh_candidates(db, exclude_ids={"b/model"}, limit=10)
        assert {r.id for r in result} == {"a/model", "c/model"}

    @pytest.mark.asyncio
    async def test_cap_is_respected(self):
        rows = [_candidate(f"vendor/model-{i}") for i in range(5)]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.list_refresh_candidates(db, exclude_ids=set(), limit=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cap_applies_after_exclusion(self):
        rows = [_candidate(f"vendor/model-{i}") for i in range(5)]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.list_refresh_candidates(
            db, exclude_ids={"vendor/model-0", "vendor/model-1"}, limit=2,
        )
        assert len(result) == 2
        assert "vendor/model-0" not in {r.id for r in result}

    @pytest.mark.asyncio
    async def test_empty_when_nothing_enabled(self):
        db = _FakeSession(execute_rows=[])
        result = await role_store.list_refresh_candidates(db, exclude_ids=set(), limit=3)
        assert result == []


def _row(id_, **overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(id=id_, status="live", enabled=True, ttfb_ms=100, tool_ok=True,
                     last_live_at=now, ttfb_fail_reason=None)
    defaults.update(overrides)
    return ModelCatalog(**defaults)


class TestDiagnoseCandidates:
    """Root observability follow-up (2026-09-27) — closes the gap where "no
    eligible candidate" gave no per-model reason and the only way to find out
    was querying Postgres by hand."""

    @pytest.mark.asyncio
    async def test_disabled_row_reported_as_disabled(self):
        db = _FakeSession(execute_rows=[_row("a/model", enabled=False)])
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "disabled")]

    @pytest.mark.asyncio
    async def test_excluded_row_reported_as_excluded(self):
        db = _FakeSession(execute_rows=[_row("a/model")])
        result = await role_store.diagnose_candidates(db, exclude_ids={"a/model"}, max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "excluded")]

    @pytest.mark.asyncio
    async def test_not_live_row_reported_as_stale(self):
        db = _FakeSession(execute_rows=[_row("a/model", status="timeout")])
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "stale")]

    @pytest.mark.asyncio
    async def test_old_last_live_at_reported_as_stale(self):
        old = datetime.now(timezone.utc) - timedelta(minutes=999)
        db = _FakeSession(execute_rows=[_row("a/model", last_live_at=old)])
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "stale")]

    @pytest.mark.asyncio
    async def test_null_ttfb_never_probed_reported_distinctly_from_failed(self):
        db = _FakeSession(execute_rows=[_row("a/model", ttfb_ms=None, ttfb_fail_reason=None)])
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "no_ttfb:never_probed")]

    @pytest.mark.asyncio
    async def test_null_ttfb_with_a_fail_reason_reports_it(self):
        """The root live-stack repro shape: reconfirmed live/tool_ok=true,
        but the TTFB probe itself structurally failed."""
        db = _FakeSession(execute_rows=[_row("meta/muse-glimmer-30b", ttfb_ms=None, ttfb_fail_reason="reasoning_only")])
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("meta/muse-glimmer-30b", "no_ttfb:reasoning_only")]

    @pytest.mark.asyncio
    async def test_ttfb_too_high_reported(self):
        db = _FakeSession(execute_rows=[_row("a/model", ttfb_ms=9000)])
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "ttfb_too_high")]

    @pytest.mark.asyncio
    async def test_tool_ok_false_reported(self):
        db = _FakeSession(execute_rows=[_row("a/model", tool_ok=False)])
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "tool_ok_false")]

    @pytest.mark.asyncio
    async def test_a_row_that_actually_qualifies_is_skipped_not_reported(self):
        """Shouldn't normally happen (diagnose_candidates is only called when
        list_promotion_candidates already found nothing), but must not crash
        or misreport if it does."""
        db = _FakeSession(execute_rows=[_row("a/model")])  # enabled, live, fresh, ttfb ok, tool_ok
        result = await role_store.diagnose_candidates(db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30)
        assert result == []

    @pytest.mark.asyncio
    async def test_capped_at_limit(self):
        rows = [_row(f"vendor/model-{i}", enabled=False) for i in range(10)]
        db = _FakeSession(execute_rows=rows)
        result = await role_store.diagnose_candidates(
            db, exclude_ids=set(), max_ttfb_ms=5000, max_stale_min=30, limit=3,
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_first_applicable_reason_wins_priority_order(self):
        """A disabled row is reported as "disabled" even if it would ALSO be
        stale/excluded/etc — disabled is checked first."""
        db = _FakeSession(execute_rows=[_row("a/model", enabled=False, status="timeout", ttfb_ms=None)])
        result = await role_store.diagnose_candidates(db, exclude_ids={"a/model"}, max_ttfb_ms=5000, max_stale_min=30)
        assert result == [("a/model", "disabled")]
