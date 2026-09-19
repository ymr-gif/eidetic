"""Per-model rate-limit bucket resolution — split out of rate_limiter.py
(pre-split, HANDOFF Phase 3) so the Phase 3 shared-catalog-bucket logic has
its own home. `check_model_rate` in rate_limiter.py still owns the Redis
sliding-window mechanics; this module only decides WHICH bucket + limit apply
to an explicitly-selected model.

A role id (llama/coder/reasoning) keeps its own named bucket, unchanged. Any
other explicitly-picked id — a live-catalog model, Phase 3 — shares ONE bucket
(`config.MODEL_RATE_LIMITS["catalog"]`, default 20/60s) rather than getting an
unbounded per-model allowance; `demo_*` ephemeral accounts get half of it.
"""
import config
from services.demo import is_ephemeral_demo


def resolve_bucket(full_model_name: str, username: str) -> tuple[str, int, int] | None:
    """Return (bucket_key, limit, window) for an explicitly-selected model, or
    None if nothing should be enforced for it."""
    role_key = next((k for k, v in config.MODELS.items() if v == full_model_name), None)
    if role_key is not None:
        if role_key not in config.MODEL_RATE_LIMITS:
            return None
        limit_count, window = config.MODEL_RATE_LIMITS[role_key]
        return role_key, limit_count, window

    limit_count, window = config.MODEL_RATE_LIMITS.get("catalog", (20, 60))
    if config.DEMO_EPHEMERAL_ENABLED and is_ephemeral_demo(username):
        limit_count = max(1, limit_count // 2)
    return "catalog", limit_count, window
