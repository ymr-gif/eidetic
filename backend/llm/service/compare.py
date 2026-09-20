import asyncio
import logging

from config import MODELS
from llm.nim import call_stream
from llm.model_extras import apply_request_extras
from llm.catalog import cache as catalog_cache
from llm.catalog.labels import derive_label

logger = logging.getLogger("service")


def _label_for(model_id: str) -> str:
    entry = catalog_cache.get_entry(model_id)
    if entry and entry.get("label"):
        return entry["label"]
    role = next((k for k, v in MODELS.items() if v == model_id), None)
    if role:
        return role.capitalize()
    return derive_label(model_id)


async def compare_streams(
    message:      str,
    common_msgs:  list[dict],
    model_params: dict | None,
    request_id:   str,
    models:       list[str] | None = None,
):
    """Run all models concurrently; yield tagged token events.

    `models` (Phase 3, live model catalog) overrides the default 3-role
    comparison with an explicit id list (already strict-resolved by the
    caller — api/chat/stream.py). The first event is always `compare_start`
    naming exactly which models + labels are being compared, so the frontend
    can build result cards before the first token arrives."""
    queue    = asyncio.Queue()
    selected = list(models) if models else list(MODELS.values())

    yield {
        "type": "compare_start",
        "models": [{"id": m, "label": _label_for(m)} for m in selected],
    }

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

    tasks = [asyncio.create_task(_run(m)) for m in selected]
    done  = 0

    while done < len(selected):
        item = await queue.get()
        if "__done__" in item:
            done += 1
        else:
            yield item

    yield {"type": "done", "compare": True, "model": "compare", "cache_hit": False, "fallback_used": False}
    await asyncio.gather(*tasks, return_exceptions=True)
