"""get_pricing(): the one place cost calculation should look up a model's
per-1M-token rates. Replaces the old `MODEL_PRICING.get(model, {})` pattern
(which billed an unpriced model at $0 — the demo per-account/global caps
didn't bind) with a three-tier lookup, catalog first:

  1. model_catalog row's price_in/price_out (admin override, Phase 3)
  2. config.MODEL_PRICING[model_id] (the static table)
  3. config.DEFAULT_MODEL_PRICE_IN/OUT (user decision: unpriced != free)

Every call site building a Message.cost_usd should go through this, not
MODEL_PRICING directly — see api/chat/background.py, api/chat/usage_ledger.py.
"""
import config
from llm.catalog import cache as catalog_cache


def get_pricing(model_id: str) -> dict:
    entry = catalog_cache.get_entry(model_id)
    if entry and entry.get("price_in") is not None and entry.get("price_out") is not None:
        return {"input": entry["price_in"], "output": entry["price_out"]}

    if model_id in config.MODEL_PRICING:
        return config.MODEL_PRICING[model_id]

    return {"input": config.DEFAULT_MODEL_PRICE_IN, "output": config.DEFAULT_MODEL_PRICE_OUT}
