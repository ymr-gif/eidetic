import asyncio
import logging

from config import MODELS
from llm.nim import call_stream
from llm.model_extras import apply_request_extras

logger = logging.getLogger("service")


async def compare_streams(
    message:      str,
    common_msgs:  list[dict],
    model_params: dict | None,
    request_id:   str,
):
    """Run all models concurrently; yield tagged token events."""
    queue  = asyncio.Queue()
    models = list(MODELS.values())

    async def _run(model: str) -> None:
        try:
            msgs = common_msgs + [{"role": "user", "content": message}]
            # Same reasoning-toggle extras as any other chat turn (Phase 2c):
            # applied to every model in the comparison, including reasoning.
            _params = apply_request_extras(model, model_params)
            async for chunk in call_stream(model, msgs, request_id, _params):
                await queue.put({"type": "token", "content": chunk, "model": model})
        except Exception as e:
            logger.warning("[compare] %s failed: %s", model, e)
        await queue.put({"__done__": model})

    tasks = [asyncio.create_task(_run(m)) for m in models]
    done  = 0

    while done < len(models):
        item = await queue.get()
        if "__done__" in item:
            done += 1
        else:
            yield item

    yield {"type": "done", "compare": True, "model": "compare", "cache_hit": False, "fallback_used": False}
    await asyncio.gather(*tasks, return_exceptions=True)
