"""Per-model request-body adjustments for the reasoning-model budget hotfix
(Phase 2b/2c, 2026-09-20 — see HANDOFF.md + backend/CLAUDE.md).

All three live NIM chat models are reasoning models: hidden reasoning tokens are
spent out of `max_tokens` before any visible `content` appears. Phase 2b applied
the "fast" field from `config.MODEL_REQUEST_EXTRAS` only to auxiliary calls
(titles, classifiers, graph extraction, summarizer/compaction, insights,
probes), keeping thinking ON for the `reasoning` role's own chat turns. Phase 2c
dropped that split: root found live that nemotron-3-super intermittently leaks
chain-of-thought straight into `content` when thinking is on, which is
unacceptable on a user-facing turn — so the extras now apply to EVERY call for
a listed model, always, no `fast`/thinking-on path left. Some models still
have no full reasoning-off switch (gpt-oss-20b's `reasoning_effort: "low"`) and
can starve a small max_tokens budget even so; `config.MODEL_MIN_MAX_TOKENS`
raises an explicitly-set max_tokens that's too low for that model. An absent
max_tokens already means "uncapped", so it's left alone either way.

Every outgoing chat-completion call site builds its `model_params` through
`apply_request_extras()` before handing it to `llm.nim.call`/`call_stream` (or,
for the two raw-httpx startup probes in api/system.py, merges the same dict
straight into the JSON body). Unknown model ids — a future catalog id with no
admin-set override, or the single alias `LLM_BACKEND=homeserver` collapses
MODELS to — are not in MODEL_REQUEST_EXTRAS/MODEL_MIN_MAX_TOKENS, so they get
no extras and no floor: never send an unverified field to a model that hasn't
confirmed it accepts it.

Phase 3 (live model catalog): both lookups check the model_catalog row's
admin-editable `request_extras`/`min_max_tokens` FIRST — an admin enabling a
new reasoning-leaking catalog model can apply the same fix without a code
change. Falls through to the static config.py tables, then to no override at
all. The catalog cache (llm/catalog/cache.py) is an in-process dict read
synchronously here — no I/O on this call path.
"""
import config
from llm.catalog import cache as catalog_cache


def extras_for(model_id: str) -> dict:
    """The reasoning-toggle field(s) for `model_id`: catalog override first,
    else config.MODEL_REQUEST_EXTRAS, else {} (unknown ids, homeserver alias)."""
    entry = catalog_cache.get_entry(model_id)
    if entry and entry.get("request_extras"):
        return dict(entry["request_extras"])
    return dict(config.MODEL_REQUEST_EXTRAS.get(model_id, {}))


def apply_request_extras(model_id: str, model_params: dict | None) -> dict:
    """Return a COPY of `model_params` with `model_id`'s reasoning-toggle extras
    always merged in, plus its max_tokens floor applied if one is configured and
    an explicitly-set max_tokens falls below it. Never mutates the caller's
    dict — safe to call with a shared/default dict."""
    merged = dict(model_params or {})
    merged.update(extras_for(model_id))

    entry = catalog_cache.get_entry(model_id)
    if entry and entry.get("min_max_tokens") is not None:
        floor = entry["min_max_tokens"]
    else:
        floor = config.MODEL_MIN_MAX_TOKENS.get(model_id)

    if floor is not None:
        current = merged.get("max_tokens")
        if isinstance(current, int) and current < floor:
            merged["max_tokens"] = floor

    return merged
