"""model_catalog table — admin-curated live NIM model catalog (HANDOFF Phase 3)

Live model catalog (user-approved 2026-09-19, spec:
plans/do-we-need-updates-ethereal-lerdorf.md § B-A). The scanner
(llm/catalog/scanner.py) walks NIM's /v1/models every 6h + on manual Rescan,
probes every chat-candidate id with a 1-token call, and upserts one row per
id here. Admin curates `enabled`; `status`/`http_status`/`latency_ms`/
`fail_count` are scanner-owned. `price_in`/`price_out`/`context_window` are
admin overrides consulted before the static config.py tables
(llm/catalog/pricing.py, llm/router.py:get_context_limit).

`request_extras`/`min_max_tokens` (added to the original spec by root after
Phase 2c, before this migration was cut) let an admin apply the same
reasoning-toggle-field / max_tokens-floor mechanism config.py's
MODEL_REQUEST_EXTRAS/MODEL_MIN_MAX_TOKENS give the three seeded role models to
a newly-enabled catalog model — llm/model_extras.py checks the catalog
override first. `reasoning` is set by the scanner's probe (empty `content` +
non-empty `reasoning_content` on the 1-token call) so the admin UI can flag
"reasoning model — set extras" before anyone hits nemotron-3-super's
chain-of-thought-leak bug on a brand new id.

Revision ID: 050
Revises: 049
Create Date: 2026-09-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision      = "050"
down_revision = "049"
branch_labels = None
depends_on    = None


def upgrade():
    op.create_table(
        "model_catalog",
        sa.Column("id",              sa.String(200), primary_key=True),
        sa.Column("label",           sa.String(200), nullable=True),
        sa.Column("status",          sa.String(20),  nullable=False, server_default="error"),
        sa.Column("http_status",     sa.Integer(),   nullable=True),
        sa.Column("latency_ms",      sa.Integer(),   nullable=True),
        sa.Column("fail_count",      sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("enabled",         sa.Boolean(),   nullable=False, server_default="false"),
        sa.Column("price_in",        sa.Float(),     nullable=True),
        sa.Column("price_out",       sa.Float(),     nullable=True),
        sa.Column("context_window",  sa.Integer(),   nullable=True),
        sa.Column("supports_tools",  sa.Boolean(),   nullable=True),
        sa.Column("reasoning",       sa.Boolean(),   nullable=True),
        sa.Column("request_extras",  postgresql.JSONB(), nullable=True),
        sa.Column("min_max_tokens",  sa.Integer(),   nullable=True),
        sa.Column("last_checked",    sa.DateTime(timezone=True), nullable=True),
        sa.Column("first_seen",      sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at",      sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_model_catalog_enabled_status", "model_catalog", ["enabled", "status"])


def downgrade():
    op.drop_index("ix_model_catalog_enabled_status", table_name="model_catalog")
    op.drop_table("model_catalog")
