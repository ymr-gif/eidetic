import logging
import time

from observability import metrics
from observability import observability
from observability import events

logger = logging.getLogger("circuit_breaker")

_THRESHOLD = 5
_COOLDOWN  = 90

_failures: dict[str, int]    = {}
_open:     dict[str, bool]   = {}
_open_time: dict[str, float] = {}
# First-failure timestamp since the last success (HANDOFF Phase A,
# auto-promotion). Distinct from `_open_time`: `is_open()` resets `_failures`
# to 0 after `_COOLDOWN` (90s) so a model can re-earn a clean slate, but that
# reset is NOT evidence of recovery — nothing re-probes the model, it just
# gets one more real request to prove itself. `_unhealthy_since` is only ever
# cleared by an actual `record_success()`, so a model that keeps failing every
# real attempt (re-opening the breaker again and again past each cooldown)
# still reports how long it has been continuously down. Consulted by
# llm/catalog/promotion.py:consider_promotion (promote only after
# AUTO_PROMOTE_MIN_DOWN_SEC of continuous failure, "not on a single 503").
_unhealthy_since: dict[str, float] = {}


def _redis_key(model: str) -> str:
    return f"cb:open:{model}"


def is_open(model: str) -> bool:
    if model not in _open:
        return False
    if time.time() - _open_time[model] > _COOLDOWN:
        _open.pop(model, None)
        _failures[model] = 0
        logger.info("[circuit] reset model=%s", model)
        return False
    return True


async def record_failure(model: str) -> None:
    _failures[model] = _failures.get(model, 0) + 1
    if model not in _unhealthy_since:
        _unhealthy_since[model] = time.time()
    if _failures[model] >= _THRESHOLD:
        _open[model]      = True
        _open_time[model] = time.time()
        metrics.record_circuit_trip(model)
        try:
            from config import USE_REDIS
            if USE_REDIS:
                from core.redis_client import get_redis
                await get_redis().set(_redis_key(model), "1", ex=_COOLDOWN)
        except Exception:
            pass
        try:
            await observability.publish_error_event(
                events.error_event(error_type="circuit_open", model=model)
            )
        except Exception:
            pass
        logger.warning("[circuit] opened model=%s", model)


def record_success(model: str) -> None:
    _failures[model] = 0
    _unhealthy_since.pop(model, None)
    _open.pop(model, None)
    try:
        from config import USE_REDIS
        if USE_REDIS:
            import asyncio
            from core.redis_client import get_redis
            asyncio.get_event_loop().create_task(get_redis().delete(_redis_key(model)))
    except Exception:
        pass


def unhealthy_since(model: str) -> float | None:
    """`time.time()` of the first consecutive failure since the last success,
    or None if the model is currently healthy (no failure recorded since its
    last success, or it has never failed). See the `_unhealthy_since` comment
    above for why this is independent of `is_open()`'s 90s cooldown reset."""
    return _unhealthy_since.get(model)


async def restore_circuit_state() -> None:
    try:
        from config import USE_REDIS, MODELS
        if not USE_REDIS:
            return
        from core.redis_client import get_redis
        r = get_redis()
        for model_str in MODELS.values():
            if await r.exists(_redis_key(model_str)):
                _open[model_str]      = True
                _open_time[model_str] = time.time()
                logger.warning("[circuit] restored open state for model=%s", model_str)
    except Exception as e:
        logger.warning("[circuit] restore_circuit_state failed: %s", e)
