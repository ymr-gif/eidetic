"""Auto-promotion: keep serving when a ROLE model dies (HANDOFF Phase A)

Two additions, both inert until `MODEL_AUTO_PROMOTE_ENABLED=true`:

1. Four new `model_catalog` columns for a stricter, on-top-of-liveness probe
   (llm/catalog/verify.py:verify_model + the scanner's new streaming TTFB
   measurement) — `ttfb_ms`, `tool_ok`, `verified_at`, `reasoning_leak`. Only
   populated for rows the scanner already sees as `status="live"`; every
   existing row gets NULLs here until its next scan.

2. New `model_role_override` table — at most one row per role
   ("llama"/"coder"/"reasoning"), present only while that role is routed away
   from its `.env` base model (auto-promoted or manually pinned). Absence of
   a row means "serve config.MODELS[role] as always" — the table starts
   empty, so every existing deployment's routing is unaffected until a
   promotion or manual admin assignment actually writes a row.

Revision ID: 052
Revises: 051
Create Date: 2026-09-26
"""
from alembic import op
import sqlalchemy as sa

revision      = "052"
down_revision = "051"
branch_labels = None
depends_on    = None


def upgrade():
    op.add_column("model_catalog", sa.Column("ttfb_ms",        sa.Integer(),               nullable=True))
    op.add_column("model_catalog", sa.Column("tool_ok",        sa.Boolean(),               nullable=True))
    op.add_column("model_catalog", sa.Column("verified_at",    sa.DateTime(timezone=True), nullable=True))
    op.add_column("model_catalog", sa.Column("reasoning_leak", sa.Boolean(),               nullable=True))

    op.create_table(
        "model_role_override",
        sa.Column("role",          sa.String(20),  primary_key=True),
        sa.Column("model_id",      sa.String(200), nullable=False),
        sa.Column("pinned",        sa.Boolean(),   nullable=False, server_default="false"),
        sa.Column("promoted_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason",        sa.String(500), nullable=True),
        sa.Column("base_model_id", sa.String(200), nullable=False),
    )


def downgrade():
    op.drop_table("model_role_override")
    op.drop_column("model_catalog", "reasoning_leak")
    op.drop_column("model_catalog", "verified_at")
    op.drop_column("model_catalog", "tool_ok")
    op.drop_column("model_catalog", "ttfb_ms")
