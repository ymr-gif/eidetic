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

    `ttfb_ms`/`tool_ok`/`verified_at`/`reasoning_leak` (migration 052, HANDOFF
    Phase A prerequisite): a second, stricter probe run for LIVE rows only,
    on top of the plain liveness probe above. `ttfb_ms` is a STREAMING
    time-to-first-token measurement — separate from `latency_ms` (a non-stream
    1-token call can return in ~1s while the same model takes tens of seconds
    to stream its first real token; deepseek-v4.1-flash measured 1.4s
    non-stream vs. ~52s TTFB). `tool_ok`/`reasoning_leak` come from
    llm/catalog/verify.py:verify_model — a fixed one-tool prompt asserting a
    well-formed tool call whose argument keys are a SUBSET of the declared
    schema (catches a model that fabricates argument keys the schema never
    declared, e.g. ising-calibration-1.5-31b inventing temperature/humidity/UV
    as "arguments"). These four columns are the auto-promotion candidate
    filter's health/quality signal (llm/catalog/promotion.py) — never
    consulted by ordinary chat routing.

    `ttfb_fail_reason` (migration 053, root live-stack finding 2026-09-27):
    when `ttfb_ms` couldn't be measured, this records WHY, so a model that
    was actively probed and structurally failed (e.g. it spent its whole
    token budget on `reasoning_content` and never reached a `content` delta
    — the meta/muse-glimmer-30b repro) is distinguishable from a row that
    has simply never been TTFB-probed at all — both used to collapse to a
    bare `ttfb_ms IS NULL` with no way to tell them apart short of reading
    `verified_at`. Set by llm/catalog/verify.py:probe_ttfb's caller
    (store.py:upsert_scan_result) alongside ttfb_ms on every verified pass —
    cleared back to null the moment a probe succeeds. Values: "reasoning_only"
    (saw reasoning_content, never content), "no_content" (stream ended with
    neither), "http_error", "timeout", "error". Purely diagnostic — a null
    ttfb_ms already excludes the row from promotion candidacy either way
    (see llm/catalog/role_store.py:list_promotion_candidates).
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
    ttfb_ms:         Mapped[int | None]      = mapped_column(Integer, nullable=True)
    tool_ok:         Mapped[bool | None]     = mapped_column(Boolean, nullable=True)
    verified_at:     Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reasoning_leak:  Mapped[bool | None]     = mapped_column(Boolean, nullable=True)
    ttfb_fail_reason: Mapped[str | None]     = mapped_column(String(40), nullable=True)
    last_checked:    Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen:      Mapped[datetime]        = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at:      Mapped[datetime]        = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_model_catalog_enabled_status", "enabled", "status"),
    )


class ModelRoleOverride(Base):
    """One row per role CURRENTLY routed away from its `.env` base model
    (HANDOFF Phase A, migration 052). Absence of a row for a role means "no
    override" — that role serves its config.MODELS[role] base id.

    Written by two paths: llm/catalog/promotion.py's automatic
    consider_promotion()/consider_revert() (`reason="auto"`, `pinned=False`
    so consider_revert can later remove it), and an admin's manual
    PATCH /api/admin/models/roles/{role} (`reason="manual"`, ALWAYS
    `pinned=True` — a manual assignment implies pinned per the HANDOFF design
    decision, so auto-revert/auto-promote never move it out from under the
    admin). A row can also represent "pin the base model itself"
    (`model_id == base_model_id`, `pinned=True`) — the only way to freeze a
    role that ISN'T currently promoted.

    `base_model_id` is a copy of config.MODELS[role] AT THE TIME this row was
    written — kept on the row (not re-read from config.MODELS at revert time)
    so a `.env` edit made while a promotion is active still reverts to the
    model that was actually overridden, not whatever config.MODELS now says.
    """
    __tablename__ = "model_role_override"

    role:          Mapped[str]             = mapped_column(String(20), primary_key=True)
    model_id:      Mapped[str]             = mapped_column(String(200), nullable=False)
    pinned:        Mapped[bool]            = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    promoted_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reason:        Mapped[str | None]      = mapped_column(String(500), nullable=True)
    base_model_id: Mapped[str]             = mapped_column(String(200), nullable=False)
