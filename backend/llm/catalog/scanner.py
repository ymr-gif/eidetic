"""Live NIM model catalog scanner (HANDOFF Phase 3).

`run_scan()`:
  1. GET /v1/models (same auth-header logic as llm/nim.py, but a raw httpx call
     — the scanner is maintenance traffic, never routed through the per-model
     circuit breaker or Prometheus metrics).
  2. Filter to chat candidates (llm.catalog.labels.is_chat_candidate).
  3. Probe every candidate with a 1-token chat-completions call, bounded
     concurrency (config.CATALOG_PROBE_CONCURRENCY), each capped at
     config.CATALOG_PROBE_TIMEOUT.
  4. Upsert one model_catalog row per candidate (llm.catalog.store), mark any
     previously-seen id absent from this scan's listing as delisted.
  5. Publish the refreshed snapshot (llm.catalog.cache.publish) and record
     scan metadata for the admin panel + startup skip-check.

No-ops in homeserver mode (there is no catalog there). Guarded by a Redis NX
lock (`catalog:scan:lock`, 900s TTL) so the 6h cron and a manual Rescan can
never run concurrently — POST /admin/models/rescan surfaces that as 409.

Never holds a DB session across the network calls: the listing + every probe
complete BEFORE `AsyncSessionLocal()` is opened; the session only does the
(local, fast) upsert pass.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import httpx

import config
from core.db import AsyncSessionLocal
from llm.endpoint import _probe_url
from llm.catalog import cache as catalog_cache
from llm.catalog.labels import derive_label, is_chat_candidate
from llm.catalog.store import list_rows, mark_delisted, upsert_scan_result

logger = logging.getLogger("catalog.scanner")

_LOCK_KEY = "catalog:scan:lock"
_LOCK_TTL = 900
_META_KEY = "catalog:scan:meta"

# Sentinel: "leave the row's current status exactly as it is" (429 — a
# rate-limit response says nothing about whether the model is actually live).
_KEEP_PRIOR_STATUS = "__keep__"


def _auth_headers() -> dict:
    """Mirrors llm/nim.py's `_auth_headers` for the NIM endpoint (no failover
    primary here — the catalog only ever describes the NIM catalog)."""
    headers = {"Content-Type": "application/json"}
    if config.NVIDIA_API_KEY:
        headers["Authorization"] = f"Bearer {config.NVIDIA_API_KEY}"
    return headers


async def _list_models() -> list[str]:
    import llm.client as llm_client
    if llm_client.client is None:
        return []
    url = _probe_url(config.NIM_URL)
    resp = await llm_client.client.get(url, headers=_auth_headers(), timeout=config.CATALOG_PROBE_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return [row.get("id") for row in data.get("data", []) if row.get("id")]


async def _probe_one(model_id: str, sem: asyncio.Semaphore) -> dict:
    """One raw-httpx 1-token probe — bypasses llm.nim entirely (no circuit
    breaker read/write, no Prometheus counters): this is scanner traffic."""
    import llm.client as llm_client
    async with sem:
        t0 = time.monotonic()
        try:
            resp = await llm_client.client.post(
                config.NIM_URL,
                headers=_auth_headers(),
                json={"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=config.CATALOG_PROBE_TIMEOUT,
            )
        except httpx.TimeoutException:
            return {"id": model_id, "status": "timeout", "http_status": None,
                    "latency_ms": int((time.monotonic() - t0) * 1000), "reasoning": None}
        except Exception as e:
            logger.warning("[catalog.scanner] probe error model=%s err=%s", model_id, e)
            return {"id": model_id, "status": "error", "http_status": None,
                    "latency_ms": int((time.monotonic() - t0) * 1000), "reasoning": None}

        latency_ms = int((time.monotonic() - t0) * 1000)

        if resp.status_code == 200:
            reasoning = None
            try:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    reasoning_content = message.get("reasoning_content") or message.get("reasoning")
                    reasoning = bool(reasoning_content) and not (isinstance(content, str) and content.strip())
            except Exception:
                pass
            return {"id": model_id, "status": "live", "http_status": 200, "latency_ms": latency_ms, "reasoning": reasoning}
        if resp.status_code == 404:
            return {"id": model_id, "status": "not_found", "http_status": 404, "latency_ms": latency_ms, "reasoning": None}
        if resp.status_code == 410:
            return {"id": model_id, "status": "gone", "http_status": 410, "latency_ms": latency_ms, "reasoning": None}
        if resp.status_code == 429:
            return {"id": model_id, "status": _KEEP_PRIOR_STATUS, "http_status": 429, "latency_ms": latency_ms, "reasoning": None}
        return {"id": model_id, "status": "error", "http_status": resp.status_code, "latency_ms": latency_ms, "reasoning": None}


async def _acquire_lock() -> bool:
    if not config.USE_REDIS:
        return True
    from core.redis_client import get_redis
    return bool(await get_redis().set(_LOCK_KEY, "1", nx=True, ex=_LOCK_TTL))


async def _release_lock() -> None:
    if not config.USE_REDIS:
        return
    try:
        from core.redis_client import get_redis
        await get_redis().delete(_LOCK_KEY)
    except Exception:
        logger.warning("[catalog.scanner] lock release failed", exc_info=True)


async def run_scan(trigger: str = "cron") -> dict:
    """Run one catalog scan. `trigger`: 'cron' | 'startup' | 'manual'.
    Returns a summary dict; `{"skipped": "homeserver"}` / `{"skipped": "already_running"}`
    on a no-op."""
    if config.LLM_BACKEND == "homeserver":
        return {"skipped": "homeserver"}

    if not await _acquire_lock():
        return {"skipped": "already_running"}

    started = time.monotonic()
    try:
        model_ids = await _list_models()
        candidates = [m for m in model_ids if is_chat_candidate(m)]

        sem = asyncio.Semaphore(max(1, config.CATALOG_PROBE_CONCURRENCY))
        probe_results = await asyncio.gather(*(_probe_one(m, sem) for m in candidates))

        role_ids = set(config.MODELS.values())
        summary: dict[str, int] = {}

        async with AsyncSessionLocal() as db:
            existing_status = {r.id: r.status for r in await list_rows(db)}
            for result in probe_results:
                status = result["status"]
                if status == _KEEP_PRIOR_STATUS:
                    status = existing_status.get(result["id"], "error")
                    summary["kept_429"] = summary.get("kept_429", 0) + 1
                else:
                    summary[status] = summary.get(status, 0) + 1
                await upsert_scan_result(
                    db, result["id"],
                    status=status, http_status=result["http_status"],
                    latency_ms=result["latency_ms"], label=derive_label(result["id"]),
                    seed_enabled=result["id"] in role_ids, reasoning=result["reasoning"],
                )
            delisted = await mark_delisted(db, set(candidates))
            await db.commit()

        await catalog_cache.publish()

        meta = {
            "trigger": trigger,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "scanned": len(candidates),
            "delisted": delisted,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        if config.USE_REDIS:
            try:
                from core.redis_client import get_redis
                await get_redis().set(_META_KEY, json.dumps(meta))
            except Exception:
                logger.warning("[catalog.scanner] failed to persist scan meta", exc_info=True)

        logger.info("[catalog.scanner] scan complete trigger=%s scanned=%d delisted=%d summary=%s",
                    trigger, len(candidates), delisted, summary)
        return meta
    finally:
        await _release_lock()


async def get_scan_meta() -> dict | None:
    if not config.USE_REDIS:
        return None
    try:
        from core.redis_client import get_redis
        raw = await get_redis().get(_META_KEY)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def is_scan_running() -> bool:
    if not config.USE_REDIS:
        return False
    try:
        from core.redis_client import get_redis
        return bool(await get_redis().exists(_LOCK_KEY))
    except Exception:
        return False
