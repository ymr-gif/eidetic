from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from core.db import Base


class ModelCatalog(Base):
    """One row per NIM model id the scanner has ever seen (HANDOFF Phase 3,
    migration 050). `id` is the raw NIM model id (e.g. "z-ai/glm-5.3-flash") —
    used as the primary key rather than a surrogate since it IS the natural
    key everywhere else (MODELS dict values, ChatRequest.model_override, ...).

    `request_extras`/`min_max_tokens` are admin-editable overrides consulted by
    llm/model_extras.py BEFORE the static config.MODEL_REQUEST_EXTRAS /
    MODEL_MIN_MAX_TOKENS tables (see llm/catalog/cache.py). `reasoning` is set
    by the scanner's probe (empty `content` + non-empty `reasoning_content`) so
    the admin UI can warn "reasoning model — set extras" on a newly-live id.

    `last_live_at` (migration 051, HANDOFF Phase 7): the last time a probe
    actually returned status="live" for this id. Set by
    llm/catalog/store.py:upsert_scan_result whenever a scan probe is live;
    left untouched on every other outcome. Closes the "never offer a model
    that was never live" gap — a model that has only ever timed out/errored
    (fail_count<=1, "tolerate 1 failed probe") must not be treated as
    available just because it hasn't failed twice yet.
    """
    __tablename__ = "model_catalog"

    id:              Mapped[str]             = mapped_column(String(200), primary_key=True)
    label:           Mapped[str | None]      = mapped_column(String(200), nullable=True)
    status:          Mapped[str]             = mapped_column(String(20), nullable=False, default="error", server_default="error")
    http_status:     Mapped[int | None]      = mapped_column(Integer, nullable=True)
    latency_ms:      Mapped[int | None]      = mapped_column(Integer, nullable=True)
    fail_count:      Mapped[int]             = mapped_column(Integer, nullable=False, default=0, server_default="0")
    enabled:         Mapped[bool]            = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    price_in:        Mapped[float | None]    = mapped_column(Float, nullable=True)
    price_out:       Mapped[float | None]    = mapped_column(Float, nullable=True)
    context_window:  Mapped[int | None]      = mapped_column(Integer, nullable=True)
    supports_tools:  Mapped[bool | None]     = mapped_column(Boolean, nullable=True)  # null in v1 — not yet probed
    reasoning:       Mapped[bool | None]     = mapped_column(Boolean, nullable=True)
    request_extras:  Mapped[dict | None]     = mapped_column(JSONB, nullable=True)
    min_max_tokens:  Mapped[int | None]      = mapped_column(Integer, nullable=True)
    last_live_at:    Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_checked:    Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen:      Mapped[datetime]        = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at:      Mapped[datetime]        = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_model_catalog_enabled_status", "enabled", "status"),
    )
