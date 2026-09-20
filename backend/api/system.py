import asyncio
import logging
import os
import platform
import socket
import time
from functools import lru_cache

import psutil

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import text

import llm.client as llm_client
import config
from llm.circuit_breaker import record_failure, _THRESHOLD
from llm.model_extras import apply_request_extras
from core.db import AsyncSessionLocal
from core.redis_client import get_redis
from observability.prom_metrics import CONTENT_TYPE_LATEST, export_metrics

router = APIRouter(tags=["system"])
logger = logging.getLogger("system")

_PING_TIMEOUT = 5  # seconds per check (the /health endpoint's own pings — unchanged)

# Startup model probe — deliberately separate from _PING_TIMEOUT above (found
# 2026-09-20, HANDOFF Phase 3: 3 deploys in a row started with 2 of 3 roles
# failed over for 90s). A transient NVIDIA "503 overloaded" or a cold
# time-to-first-byte past a 5s timeout used to pre-trip the breaker for a
# model that was actually healthy. Now: a longer, env-tunable timeout, one
# retry with a short backoff before judging the model down, and pre-trip only
# on a DEFINITIVE result (401/404/410) or two consecutive failed attempts.
_STARTUP_PROBE_TIMEOUT  = int(os.getenv("STARTUP_PROBE_TIMEOUT", "15"))
_STARTUP_PROBE_RETRY_BACKOFF = 2  # seconds between the two probe attempts
_STARTUP_PROBE_DEFINITIVE_STATUSES = {401, 404, 410}


class ResponseMeta(BaseModel):
    request_id: str


class SuccessResponse(BaseModel):
    success: bool = True
    data: dict
    meta: ResponseMeta


class ErrorResponse(BaseModel):
    success: bool = False
    error: dict
    meta: ResponseMeta


async def _ping_nim() -> dict:
    t = time.monotonic()
    try:
        resp = await llm_client.client.post(
            config.NIM_URL,
            headers={"Authorization": f"Bearer {config.NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json={"model": config.MODELS["llama"], "messages": [{"role": "user", "content": "hi"}],
                  **apply_request_extras(config.MODELS["llama"], {"max_tokens": 1})},
            timeout=_PING_TIMEOUT,
        )
        latency = int((time.monotonic() - t) * 1000)
        if resp.status_code == 200:
            return {"status": "ok", "latency_ms": latency}
        return {"status": "error", "detail": f"http_{resp.status_code}", "latency_ms": latency}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:120]}


async def _ping_embedding() -> dict:
    t = time.monotonic()
    try:
        resp = await llm_client.client.post(
            config.NIM_EMBEDDING_URL,
            headers={"Authorization": f"Bearer {config.NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json={"model": config.MODEL_EMBEDDING, "input": ["ping"], "input_type": "passage", "encoding_format": "float"},
            timeout=_PING_TIMEOUT,
        )
        latency = int((time.monotonic() - t) * 1000)
        if resp.status_code == 200:
            return {"status": "ok", "latency_ms": latency}
        return {"status": "error", "detail": f"http_{resp.status_code}", "latency_ms": latency}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:120]}


async def _ping_redis() -> dict:
    t = time.monotonic()
    try:
        r = get_redis()
        await asyncio.wait_for(r.ping(), timeout=3)
        latency = int((time.monotonic() - t) * 1000)
        return {"status": "ok", "latency_ms": latency}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:120]}


async def _ping_db() -> dict:
    t = time.monotonic()
    try:
        async with AsyncSessionLocal() as db:
            await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=2)
        latency = int((time.monotonic() - t) * 1000)
        return {"status": "ok", "latency_ms": latency}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:120]}


async def _startup_probe_attempt(model_id: str) -> tuple[bool, int | None, str | None]:
    """One probe attempt. Returns (ok, http_status, error_str)."""
    try:
        resp = await llm_client.client.post(
            config.NIM_URL,
            headers={"Authorization": f"Bearer {config.NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}],
                  **apply_request_extras(model_id, {"max_tokens": 1})},
            timeout=_STARTUP_PROBE_TIMEOUT,
        )
        return resp.status_code == 200, resp.status_code, None
    except Exception as e:
        return False, None, str(e)[:80]


async def _pre_trip(model_id: str) -> None:
    for _ in range(_THRESHOLD):
        await record_failure(model_id)


async def probe_models_on_startup() -> None:
    async def _probe(role: str, model_id: str) -> None:
        ok, status_code, err = await _startup_probe_attempt(model_id)
        if ok:
            logger.info("[probe] model=%s ok", role)
            return

        if status_code in _STARTUP_PROBE_DEFINITIVE_STATUSES:
            # Auth/not-found/gone — a retry a couple seconds later won't change
            # the answer; pre-trip immediately.
            logger.warning("[probe] model=%s status=%s (definitive) — pre-tripping circuit", role, status_code)
            await _pre_trip(model_id)
            return

        # Transient (timeout/5xx/network error/"overloaded") — retry once
        # before judging the model down, so one flaky response at boot doesn't
        # fail over a healthy model for the next 90s.
        logger.warning("[probe] model=%s first attempt failed status=%s err=%s — retrying once", role, status_code, err)
        await asyncio.sleep(_STARTUP_PROBE_RETRY_BACKOFF)
        ok2, status_code2, err2 = await _startup_probe_attempt(model_id)
        if ok2:
            logger.info("[probe] model=%s ok on retry", role)
            return

        logger.warning("[probe] model=%s failed twice (status=%s err=%s) — pre-tripping circuit", role, status_code2, err2)
        await _pre_trip(model_id)

    await asyncio.gather(*(_probe(role, model_id) for role, model_id in config.MODELS.items()))


@router.get("/health")
async def health():
    nim, emb, redis, db = await asyncio.gather(
        _ping_nim(),
        _ping_embedding(),
        _ping_redis(),
        _ping_db(),
        return_exceptions=False,
    )

    checks = {"nim": nim, "embedding": emb, "redis": redis, "db": db}
    all_ok = all(c["status"] == "ok" for c in checks.values())

    return {
        "status": "ok" if all_ok else "degraded",
        "models": list(config.MODELS.keys()),
        "checks": checks,
    }


@router.get("/breakers")
async def breakers():
    """Per-model circuit-breaker state for the frontend telemetry strip.

    In-process view (the enforced one) — the Redis keys are only startup
    persistence, so this endpoint reflects what routing actually does.
    """
    from llm import circuit_breaker
    state = {role: circuit_breaker.is_open(model_id) for role, model_id in config.MODELS.items()}
    return {"models": state, "any_open": any(state.values())}


@router.get("/metrics")
def metrics_endpoint():
    try:
        return Response(content=export_metrics(), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        logger.exception("[metrics] export failed")
        return Response(content="# metrics export failed\n", media_type="text/plain", status_code=200)


# ── /hardware ──────────────────────────────────────────────────────────────────

def _gb(b: int) -> float:
    return round(b / (1024**3), 2)





@lru_cache(maxsize=1)
def _gpu_init():
    import pynvml
    pynvml.nvmlInit()
    return pynvml


def _get_gpu_info() -> dict | None:
    try:
        pynvml = _gpu_init()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "name": pynvml.nvmlDeviceGetName(handle).decode(),
            "vram_total_gb": _gb(mem.total),
            "vram_used_gb": _gb(mem.used),
            "temp_c": int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
            "load_pct": float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
        }
    except Exception:
        return None


def _get_uptime() -> str:
    delta = time.time() - psutil.boot_time()
    days, rem = divmod(int(delta), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return f"{days}d {hours}h {minutes}m"


@router.get("/system/hardware")
@router.get("/hardware")
async def hardware():
    freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_pct = await asyncio.to_thread(psutil.cpu_percent, interval=0.1)

    return {
        "cpu": {
            "name": platform.processor() or "Unknown",
            "freq_ghz": round(freq.current / 1000, 2) if freq else 0.0,
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "usage_pct": cpu_pct,
        },
        "ram": {
            "total_gb": _gb(mem.total),
            "used_gb": _gb(mem.used),
            "available_gb": _gb(mem.available),
        },
        "gpu": _get_gpu_info(),
        "disk": {
            "total_gb": _gb(disk.total),
            "free_gb": _gb(disk.free),
        },
        "uptime": _get_uptime(),
        "hostname": socket.gethostname(),
    }
