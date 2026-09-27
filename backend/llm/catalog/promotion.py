"""Auto-promotion state machine: keep serving a role when its model dies
(HANDOFF Phase A, root-approved design — see backend/HANDOFF.md).

Ships INERT: `_enabled()` gates every write path (`consider_promotion`,
`consider_revert`, `check_reverts`) and `effective_role_model()` — so with
`MODEL_AUTO_PROMOTE_ENABLED=false` (the default) or `LLM_BACKEND=homeserver`,
`effective_role_model(role)` always returns `config.MODELS[role]` and nothing
else in this module ever touches the DB/Redis. This is the ONLY function
called from the hot chat-request path (llm/router.py, llm/service/stream.py,
api/chat/model_resolve.py) — synchronous, zero I/O, reads llm.catalog.
role_state's in-process snapshot exactly like llm.catalog.cache.is_available().

State machine, in the order actually checked:
  consider_promotion(role):
    1. pinned override already set for this role -> no-op (never auto-moves).
    2. an override was set within AUTO_PROMOTE_COOLDOWN_MIN -> no-op (at most
       one promotion per role per cooldown window — flap guard).
    3. the model CURRENTLY effective for this role must have an OPEN circuit
       breaker AND have been continuously unhealthy (llm.circuit_breaker.
       unhealthy_since) for >= AUTO_PROMOTE_MIN_DOWN_SEC — "not on a single
       503"; a breaker that keeps re-tripping past its own 90s cooldown is
       exactly what makes this true, see circuit_breaker.py's own comment.
    4. candidate filter (llm.catalog.role_store.list_promotion_candidates):
       enabled + status=live + last_live_at fresh + ttfb_ms + tool_ok, minus
       the failing model, the embedder, and every id already effective for
       ANOTHER role. No candidate -> on-demand refresh (see below), then
       re-check ONCE more; still nothing -> log + metric, do nothing (never
       promote something unverified). Winner = lowest ttfb_ms.
    5. write the override row (reason="auto", pinned=False so consider_revert
       can later remove it), audit row, bump the counter, publish.

  On-demand refresh (`_refresh_stale_candidates`, root follow-up 2026-09-26):
    `AUTO_PROMOTE_MAX_STALE_MIN` (30min default) is far tighter than the
    scanner's 6h cadence, so mid-incident there is usually NO fresh candidate
    even though several enabled models are actually fine — the feature would
    no-op exactly when it's needed. Rather than widen the staleness window
    (would promote a model we haven't actually re-checked) or tighten the
    global scan cadence (real NIM cost, no benefit most of the time), when
    step 4 finds nothing this synchronously re-probes a SMALL bounded set of
    already-enabled catalog rows (ordered by last-known ttfb, capped at
    `AUTO_PROMOTE_REFRESH_MAX`) with the existing `probe_ttfb`/`verify_model`
    and — for any that reconfirm live — upserts fresh ttfb_ms/tool_ok/
    reasoning_leak/status=live/last_live_at, then step 4 runs again once.
    Guarded by a short Redis NX lock (`catalog:refresh:lock`, 60s TTL) so
    concurrent failing turns (possibly across different roles) can't
    stampede NIM with duplicate refreshes; a held lock or nothing to refresh
    just skips straight to the no-candidate outcome, same as before this
    follow-up existed.

  consider_revert(role) (called from the scheduler's existing tick):
    1. no override, or it already equals the base ("pin the base" marker,
       nothing to revert) -> no-op.
    2. **Config-base reconciliation (root live-stack finding, 2026-09-27).**
       Compare `config.MODELS[role]` (the LIVE, currently-configured base)
       against `override.base_model_id` (frozen at PROMOTION time). If they
       differ, the operator has done the normal thing after an outage: fixed
       the dead id in `.env` and restarted. An unpinned (auto) override must
       NOT survive that — the new base is an explicit human decision that
       outranks an automatic promotion, and `override.base_model_id` would
       otherwise be a permanently-dead id whose own availability check can
       never pass (the original bug: the role stayed on the promoted model
       forever, since step 3's own catalog-availability check was being run
       against the WRONG, dead base). The override is released IMMEDIATELY —
       no availability check on the new base first (if it's also broken,
       the ordinary promotion path picks that up on the next failure, same
       as any other role) — audited as a DISTINCT action
       (`model_catalog.override_released_base_changed`, not
       `auto_reverted`, since this isn't "recovered", it's "superseded"),
       never gated by `AUTO_PROMOTE_RECOVER_MIN`. A PINNED override is
       treated the OPPOSITE way: it is a deliberate manual choice (every
       pinned row today came from the admin endpoint, never from
       consider_promotion) and is left alone — only another manual admin
       action clears/reassigns it. Either way this runs on `consider_revert`'s
       existing cadence, so `effective_role_model(role)` never keeps
       returning a stale-base override for longer than one scheduler tick.
    3. (base unchanged) if pinned -> no-op (never auto-reverts regardless of
       base availability). Else the base model must be catalog-available
       continuously for `AUTO_PROMOTE_RECOVER_MIN` minutes. "Continuously" is
       tracked by `_recovering_since`, a scheduler-process-local dict (module
       state) — reset to "not yet recovered" the instant the base looks
       unavailable again, and reset entirely if the scheduler process
       restarts (accepted: worst case, the 30-minute clock restarts, never a
       false-positive early revert). This is a documented judgment call, not
       something HANDOFF specified a storage mechanism for — flagged for
       review.
    4. clear the override, audit row (`model_catalog.auto_reverted`), bump
       the counter, publish.

Both write paths open their own AsyncSessionLocal (same self-contained-service
pattern as services/demo.py) since they're invoked from disparate contexts:
the chat-request failure path (fire-and-forget, must never block a response)
and the scheduler tick (its own separate process, no shared session).
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

import config
from llm import circuit_breaker
from llm.catalog import cache as catalog_cache
from llm.catalog import role_state
from llm.catalog import role_store
from llm.catalog import store
from llm.catalog.verify import probe_ttfb, verify_model
from observability import metrics

logger = logging.getLogger("catalog.promotion")

# Scheduler-process-local "how long has the base model looked available"
# tracker for consider_revert — see the module docstring's point 2 above.
_recovering_since: dict[str, float] = {}

# On-demand-refresh coordination lock (root follow-up, 2026-09-26) — see
# _refresh_stale_candidates. Mirrors llm/catalog/scanner.py's own
# catalog:scan:lock pattern (short TTL, best-effort — a stampede here wastes
# NIM calls, it doesn't corrupt data, so failing open on a Redis error is
# the right default).
_REFRESH_LOCK_KEY = "catalog:refresh:lock"
_REFRESH_LOCK_TTL = 60


def _enabled() -> bool:
    return config.MODEL_AUTO_PROMOTE_ENABLED and config.LLM_BACKEND != "homeserver"


def effective_role_model(role: str) -> str:
    """The model id that should ACTUALLY serve `role` right now: an active
    override if one exists, else `config.MODELS[role]`. Synchronous, zero
    I/O — the ONLY promotion-aware function on the hot chat path. Callers
    that need a fresh view should call `role_state.ensure_fresh()` earlier in
    the request (mirrors `catalog_cache.ensure_fresh()`'s existing contract);
    this function itself never awaits anything."""
    base = config.MODELS[role]
    if not _enabled():
        return base
    override = role_state.get(role)
    if override and override.get("model_id"):
        return override["model_id"]
    return base


async def _write_system_audit(db, action: str, detail: dict) -> None:
    """AdminAuditLog requires a real `admin_id` (NOT NULL FK) — there is no
    admin user "performing" an automatic promotion/revert, so this attributes
    the row to the first admin account found and marks `detail.actor` so it
    is never confused with a real admin action. If no admin user exists at
    all (shouldn't happen — the app seeds one) the audit write is skipped
    with a warning; it must never block the actual override write."""
    try:
        from sqlalchemy import select
        from models import AdminAuditLog, User
        admin_id = (
            await db.execute(select(User.id).where(User.role == "admin").order_by(User.id).limit(1))
        ).scalar_one_or_none()
        if admin_id is None:
            logger.warning("[promotion] no admin user found — skipping audit row for %s", action)
            return
        db.add(AdminAuditLog(
            id=uuid.uuid4(), admin_id=admin_id, action=action,
            detail={**detail, "actor": "system"},
        ))
    except Exception:
        logger.warning("[promotion] audit write failed for action=%s", action, exc_info=True)


async def _acquire_refresh_lock() -> bool:
    if not config.USE_REDIS:
        return True
    try:
        from core.redis_client import get_redis
        return bool(await get_redis().set(_REFRESH_LOCK_KEY, "1", nx=True, ex=_REFRESH_LOCK_TTL))
    except Exception:
        logger.warning("[promotion] refresh lock acquire failed — proceeding without coordination", exc_info=True)
        return True  # fail open: a stampede wastes NIM calls, it never corrupts data


async def _release_refresh_lock() -> None:
    if not config.USE_REDIS:
        return
    try:
        from core.redis_client import get_redis
        await get_redis().delete(_REFRESH_LOCK_KEY)
    except Exception:
        logger.warning("[promotion] refresh lock release failed", exc_info=True)


async def _refresh_stale_candidates(db, role: str, exclude_ids: set[str]) -> bool:
    """On-demand refresh (root follow-up, 2026-09-26) — see the module
    docstring's own section. Returns True iff at least one row was actually
    reconfirmed live (worth the caller re-querying list_promotion_candidates);
    False means a plain no-op (lock held elsewhere, nothing enabled to
    refresh, or every refreshed row still failed to reconfirm) — the caller
    falls through to the ordinary "no candidate" outcome either way."""
    if not await _acquire_refresh_lock():
        logger.info("[promotion] role=%s refresh skipped — lock held elsewhere", role)
        return False
    try:
        stale = await role_store.list_refresh_candidates(
            db, exclude_ids=exclude_ids, limit=config.AUTO_PROMOTE_REFRESH_MAX,
        )
        if not stale:
            return False

        sem = asyncio.Semaphore(max(1, config.AUTO_PROMOTE_REFRESH_MAX))

        async def _refresh_one(row):
            ttfb_result = await probe_ttfb(row.id, sem)
            verify_result = await verify_model(row.id, sem)
            return row, ttfb_result, verify_result

        results = await asyncio.gather(*(_refresh_one(r) for r in stale))

        reconfirmed_count = 0
        for row, ttfb_result, verify_result in results:
            if verify_result.get("error") is not None:
                # Could not reconfirm liveness this cycle (timeout/network/
                # http error) — leave the row's prior status/last_live_at
                # exactly as they were, same "never clear on a blip" rule
                # store.upsert_scan_result already follows for the scanner.
                continue
            await store.upsert_scan_result(
                db, row.id, status="live", http_status=200, latency_ms=row.latency_ms,
                label=row.label, seed_enabled=False,
                ttfb_ms=ttfb_result["ttfb_ms"], tool_ok=verify_result["tool_ok"],
                reasoning_leak=verify_result["reasoning_leak"],
                ttfb_fail_reason=ttfb_result["fail_reason"], verified=True,
            )
            reconfirmed_count += 1
        logger.info("[promotion] role=%s on-demand refresh: %d/%d row(s) reconfirmed live",
                    role, reconfirmed_count, len(stale))
        return reconfirmed_count > 0
    finally:
        await _release_refresh_lock()


async def _log_candidate_exclusions(db, role: str, exclude_ids: set[str]) -> None:
    """Closes the observability gap the root live-stack test hit: "no
    eligible candidate" alone gave no reason, and the only way to find out
    WHY a specific model didn't qualify was querying model_catalog by hand
    (repro: muse-glimmer showed status=live/tool_ok=true/last_live_at=13s ago
    and STILL didn't qualify — the reason, ttfb_ms IS NULL, was invisible).
    Logged at info (this fires on every "no candidate" outcome, which can be
    routine — e.g. right after a fresh deploy before the first scan — not
    necessarily alarming on its own); never raises, never blocks the actual
    promotion decision."""
    try:
        diagnostics = await role_store.diagnose_candidates(
            db, exclude_ids=exclude_ids,
            max_ttfb_ms=config.AUTO_PROMOTE_MAX_TTFB_MS,
            max_stale_min=config.AUTO_PROMOTE_MAX_STALE_MIN,
        )
        for model_id, reason in diagnostics:
            logger.info("[promotion] role=%s candidate excluded id=%s reason=%s", role, model_id, reason)
    except Exception:
        logger.warning("[promotion] role=%s candidate-exclusion diagnostics failed", role, exc_info=True)


async def consider_promotion(role: str) -> dict | None:
    """See the module docstring's state machine. Returns a dict describing
    the promotion on success, else None (including every no-op path — a
    "nothing happened" is not an error). Never raises: called fire-and-forget
    from the stream failure path and must never surface to the user's turn."""
    if not _enabled() or role not in config.MODELS:
        return None

    try:
        await role_state.ensure_fresh()
        current = effective_role_model(role)

        from core.db import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            existing = await role_store.get_role_override(db, role)
            if existing and existing.pinned:
                return None
            if existing and existing.promoted_at:
                elapsed_min = (datetime.now(timezone.utc) - existing.promoted_at).total_seconds() / 60
                if elapsed_min < config.AUTO_PROMOTE_COOLDOWN_MIN:
                    return None

            if not circuit_breaker.is_open(current):
                return None
            since = circuit_breaker.unhealthy_since(current)
            if since is None or (time.time() - since) < config.AUTO_PROMOTE_MIN_DOWN_SEC:
                return None

            exclude_ids = {current, config.MODEL_EMBEDDING}
            for other_role in config.MODELS:
                if other_role != role:
                    exclude_ids.add(effective_role_model(other_role))

            candidates = await role_store.list_promotion_candidates(
                db, exclude_ids=exclude_ids,
                max_ttfb_ms=config.AUTO_PROMOTE_MAX_TTFB_MS,
                max_stale_min=config.AUTO_PROMOTE_MAX_STALE_MIN,
            )
            if not candidates and await _refresh_stale_candidates(db, role, exclude_ids):
                # Persist the refresh now, independent of whether a candidate
                # turns up below — these are legitimate fresher catalog rows
                # either way, and closing the session without a commit would
                # silently roll them back (a plain `return None` right after
                # this, on a still-empty result, must not discard them).
                await db.commit()
                await catalog_cache.publish()  # other workers' ensure_fresh() picks up the fresher rows too
                candidates = await role_store.list_promotion_candidates(
                    db, exclude_ids=exclude_ids,
                    max_ttfb_ms=config.AUTO_PROMOTE_MAX_TTFB_MS,
                    max_stale_min=config.AUTO_PROMOTE_MAX_STALE_MIN,
                )
            if not candidates:
                logger.warning("[promotion] role=%s down since=%s — no eligible candidate, staying on %s",
                               role, since, current)
                await _log_candidate_exclusions(db, role, exclude_ids)
                metrics.record_auto_promotion(role, "no_candidate")
                return None

            winner = candidates[0]
            await role_store.set_role_override(
                db, role, model_id=winner.id, pinned=False, reason="auto",
                base_model_id=config.MODELS[role],
            )
            await _write_system_audit(db, "model_catalog.auto_promoted", {
                "role": role, "from": current, "to": winner.id, "ttfb_ms": winner.ttfb_ms,
            })
            await db.commit()

        await role_state.publish()
        metrics.record_auto_promotion(role, "promote")
        logger.warning("[promotion] role=%s promoted %s -> %s (ttfb_ms=%s)", role, current, winner.id, winner.ttfb_ms)
        return {"role": role, "from": current, "to": winner.id}
    except Exception:
        logger.warning("[promotion] consider_promotion failed role=%s", role, exc_info=True)
        return None


async def consider_revert(role: str) -> dict | None:
    """See the module docstring's state machine. Called from the scheduler's
    existing periodic tick (services/scheduler_worker.py), never from a chat
    request. Never raises."""
    if not _enabled() or not config.AUTO_PROMOTE_AUTO_REVERT or role not in config.MODELS:
        return None

    try:
        from core.db import AsyncSessionLocal
        outcome: dict | None = None  # set inside the `async with` block, used after it closes

        async with AsyncSessionLocal() as db:
            override = await role_store.get_role_override(db, role)
            if override is None or override.model_id == override.base_model_id:
                _recovering_since.pop(role, None)
                return None

            current_base = config.MODELS[role]

            if current_base != override.base_model_id:
                # Root live-stack finding (2026-09-27): base_model_id is
                # frozen at promotion time. The operator has since repointed
                # this role's .env base to a DIFFERENT model — the normal
                # recovery path (fix the dead id, restart) — so the OLD base
                # this override was tracking may be permanently dead and can
                # never pass the availability check below. See the module
                # docstring's consider_revert section for the full reasoning.
                if override.pinned:
                    # Manual choice — survives a config change on purpose;
                    # only another manual admin action touches it.
                    _recovering_since.pop(role, None)
                    return None
                old_model, old_base = override.model_id, override.base_model_id
                await role_store.clear_role_override(db, role)
                await _write_system_audit(db, "model_catalog.override_released_base_changed", {
                    "role": role, "released_model": old_model,
                    "old_base": old_base, "new_base": current_base,
                })
                await db.commit()
                outcome = {"kind": "released", "released": old_model,
                           "old_base": old_base, "new_base": current_base}
            else:
                if override.pinned:
                    _recovering_since.pop(role, None)
                    return None

                await catalog_cache.ensure_fresh()
                if not catalog_cache.is_available(current_base):
                    _recovering_since.pop(role, None)
                    return None

                now = time.monotonic()
                started = _recovering_since.setdefault(role, now)
                if (now - started) < config.AUTO_PROMOTE_RECOVER_MIN * 60:
                    return None

                promoted_from = override.model_id
                await role_store.clear_role_override(db, role)
                await _write_system_audit(db, "model_catalog.auto_reverted", {
                    "role": role, "from": promoted_from, "to": current_base,
                })
                await db.commit()
                outcome = {"kind": "reverted", "from": promoted_from, "to": current_base}

        _recovering_since.pop(role, None)
        await role_state.publish()
        if outcome["kind"] == "released":
            metrics.record_auto_promotion(role, "released_base_changed")
            logger.warning(
                "[promotion] role=%s .env base changed (%s -> %s) — releasing auto override %s",
                role, outcome["old_base"], outcome["new_base"], outcome["released"],
            )
            return {"role": role, "released": outcome["released"],
                    "old_base": outcome["old_base"], "new_base": outcome["new_base"]}
        metrics.record_auto_promotion(role, "revert")
        logger.warning("[promotion] role=%s auto-reverted %s -> %s", role, outcome["from"], outcome["to"])
        return {"role": role, "from": outcome["from"], "to": outcome["to"]}
    except Exception:
        logger.warning("[promotion] consider_revert failed role=%s", role, exc_info=True)
        return None


async def check_promotion_for_failed_model(model_id: str) -> None:
    """Convenience for call sites that only know WHICH MODEL just failed
    (llm/service/stream.py's fallback loop) — finds every role currently
    served by `model_id` (a shared id serves more than one role today: llama
    and coder both route to the same base model, see backend/CLAUDE.md
    "Active Models") and runs consider_promotion for each. Safe to
    fire-and-forget (`asyncio.create_task`) from a request handler — never
    raises, never blocks, and consider_promotion's own cooldown/pin checks
    make repeat calls on every failing request cheap no-ops."""
    if not _enabled():
        return
    for role in config.MODELS:
        if effective_role_model(role) == model_id:
            await consider_promotion(role)


async def check_reverts() -> None:
    """Called from the scheduler's existing `__sync__` (5-minute) tick — the
    only sufficiently-frequent existing periodic job, reused rather than
    adding a new cron entry (HANDOFF: "call consider_revert from the existing
    scheduler tick"). No-ops entirely (one cheap config check, no I/O) when
    the feature is off. Never raises — a revert-check failure must never
    break the schedule sync it's piggybacked on."""
    if not _enabled():
        return
    for role in config.MODELS:
        try:
            await consider_revert(role)
        except Exception:
            # Belt-and-suspenders: consider_revert already wraps its own body
            # in a try/except, but this loop must survive even a bug in that
            # guard itself — one role's failure must never skip the rest or
            # break the scheduler tick it's piggybacked on.
            logger.warning("[promotion] check_reverts: role=%s raised unexpectedly", role, exc_info=True)


async def set_manual_override(db, role: str, model_id: str, *, reason: str = "manual") -> dict:
    """Admin explicit assignment (PATCH /api/admin/models/roles/{role} with a
    non-null model_id) — ALWAYS pinned, a manual choice never auto-moves.
    Caller has already validated `model_id` resolves to something real."""
    await role_store.set_role_override(
        db, role, model_id=model_id, pinned=True, reason=reason, base_model_id=config.MODELS[role],
    )
    return {"role": role, "model_id": model_id, "pinned": True}


async def set_role_pin(db, role: str, pinned: bool) -> dict:
    """Toggle `pinned` WITHOUT changing which model serves the role (PATCH
    with only `pinned`, no `model_id`). If no override row exists yet and
    `pinned=True`, this pins the role to its current base model — the only
    way to freeze a role that isn't currently promoted."""
    existing = await role_store.get_role_override(db, role)
    if existing is not None:
        existing.pinned = pinned
        return {"role": role, "model_id": existing.model_id, "pinned": pinned}
    if pinned:
        base = config.MODELS[role]
        await role_store.set_role_override(
            db, role, model_id=base, pinned=True, reason="manual_pin_base", base_model_id=base,
        )
        return {"role": role, "model_id": base, "pinned": True}
    return {"role": role, "model_id": config.MODELS[role], "pinned": False}


async def clear_manual_override(db, role: str) -> None:
    """PATCH with `model_id: null` — remove any override, revert to base,
    unpinned (eligible for auto-promotion again)."""
    await role_store.clear_role_override(db, role)


def promotion_status_events(fallback_chain: list[str]) -> list[dict]:
    """One SSE status event per role whose EFFECTIVE model is auto-promoted
    (reason="auto", never a manual pin) AND appears in this turn's
    fallback_chain — HANDOFF's mandatory visibility requirement ("a silent
    model swap is worse than an error"). Synchronous, reads only the
    in-process role_state snapshot — safe to call from inside the SSE
    generator without an extra await."""
    if not _enabled():
        return []
    events = []
    for role, base in config.MODELS.items():
        override = role_state.get(role)
        if not override or override.get("reason") != "auto":
            continue
        effective_id = override.get("model_id")
        if not effective_id or effective_id == base or effective_id not in fallback_chain:
            continue
        entry = catalog_cache.get_entry(effective_id) or {}
        from llm.catalog.labels import derive_label
        label = entry.get("label") or derive_label(effective_id)
        events.append({
            "type": "status", "stage": "route", "level": "info",
            "detail": f"{role} is temporarily served by {label} (auto — {base} unavailable)",
        })
    return events
