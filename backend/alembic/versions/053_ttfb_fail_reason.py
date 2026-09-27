"""model_catalog.ttfb_fail_reason — distinguish a failed TTFB probe from
"never probed" (root live-stack finding, HANDOFF Phase A follow-up 2026-09-27)

Live-stack repro: enabling `meta/muse-glimmer-30b` as a promotion candidate,
`probe_ttfb` returned None even after the on-demand refresh reconfirmed the
model `status=live`/`tool_ok=true` — the model spends its whole (5-token)
probe budget on `reasoning_content` and hits `finish_reason="length"` before
ever emitting a `content` delta. A bare `ttfb_ms IS NULL` made that
structural failure indistinguishable from a row that has simply never been
TTFB-probed, and it silently excluded the model from every candidate list
with no logged reason.

Fixed alongside this migration (llm/catalog/verify.py, config.py): the probe
budget is bumped well past a typical reasoning preamble
(`CATALOG_TTFB_PROBE_MAX_TOKENS`, default 64, was hardcoded 5) and
`probe_ttfb` now reports WHY it failed instead of a bare None. This column
persists that reason; a null `ttfb_ms` already excludes the row from
promotion candidacy regardless of this column's value — it is purely
diagnostic.

Revision ID: 053
Revises: 052
Create Date: 2026-09-27
"""
from alembic import op
import sqlalchemy as sa

revision      = "053"
down_revision = "052"
branch_labels = None
depends_on    = None


def upgrade():
    op.add_column("model_catalog", sa.Column("ttfb_fail_reason", sa.String(40), nullable=True))


def downgrade():
    op.drop_column("model_catalog", "ttfb_fail_reason")
