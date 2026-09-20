"""model_catalog.last_live_at — never offer a model that was never live (HANDOFF Phase 7)

Root found a real hole in Phase 3's availability rule: `_entry_available`
tolerated any transient status (`timeout`/`error`) with `fail_count<=1` as
available — including a model that has NEVER once answered a probe (repro:
`nvidia/nemotron-3-ultra-550b-a55b` enabled with status `timeout`,
fail_count 1, appeared in `GET /api/models` for a normal user despite never
being live). "Tolerate 1 failed probe" was meant for a model that WAS live
and just blipped, not one that never worked.

`last_live_at` is set by the scanner (llm/catalog/store.py:upsert_scan_result)
whenever a probe returns status="live"; left alone on every other outcome.
`_entry_available` (llm/catalog/cache.py) now also requires it to be set
before tolerating a transient-status blip.

Backfill: existing rows already sitting at status='live' (created under the
old rule) get `last_live_at` set from their own `last_checked`/`updated_at`
so a model that IS live right now is never retroactively blocked from being
(re-)enabled by the new admin-PATCH 409 check — only a model that has
genuinely never been seen live gets that guard.

Revision ID: 051
Revises: 050
Create Date: 2026-09-20
"""
from alembic import op
import sqlalchemy as sa

revision      = "051"
down_revision = "050"
branch_labels = None
depends_on    = None


def upgrade():
    op.add_column(
        "model_catalog",
        sa.Column("last_live_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        UPDATE model_catalog
        SET last_live_at = COALESCE(last_checked, updated_at)
        WHERE status = 'live' AND last_live_at IS NULL
        """
    )


def downgrade():
    op.drop_column("model_catalog", "last_live_at")
