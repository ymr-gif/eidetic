"""rate_limiter/model_limits.py — per-model rate-limit bucket resolution
(pre-split out of rate_limiter.py, HANDOFF Phase 3). Covers: role ids keep
their own named bucket unchanged; any other explicitly-picked (catalog) id
shares ONE bucket; ephemeral demo accounts get half that shared bucket's
limit; a role with no configured limit enforces nothing.

Unit tier — no Redis (resolve_bucket is pure; check_model_rate's Redis
mechanics are exercised indirectly by the pre-existing rate_limiter tests via
fail-open, unaffected by this pre-split).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest

import config
from rate_limiter.model_limits import resolve_bucket

LLAMA = config.MODELS["llama"]
CODER = config.MODELS["coder"]
REASONING = config.MODELS["reasoning"]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(config, "DEMO_EPHEMERAL_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "MODEL_RATE_LIMITS", {
        "llama": (15, 60), "coder": (10, 60), "reasoning": (5, 60), "catalog": (20, 60),
    }, raising=False)
    yield


class TestRoleBuckets:
    def test_role_id_uses_its_own_named_bucket(self):
        assert resolve_bucket(LLAMA, "alice") == ("llama", 15, 60)
        assert resolve_bucket(CODER, "alice") == ("coder", 10, 60)
        assert resolve_bucket(REASONING, "alice") == ("reasoning", 5, 60)

    def test_role_with_no_configured_limit_enforces_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_RATE_LIMITS", {"catalog": (20, 60)}, raising=False)
        assert resolve_bucket(LLAMA, "alice") is None


class TestCatalogSharedBucket:
    def test_unknown_catalog_id_shares_the_catalog_bucket(self):
        assert resolve_bucket("z-ai/glm-5.3-flash", "alice") == ("catalog", 20, 60)

    def test_two_different_catalog_ids_share_the_same_bucket_key(self):
        a = resolve_bucket("z-ai/glm-5.3-flash", "alice")
        b = resolve_bucket("qwen/qwen3-max", "alice")
        assert a[0] == b[0] == "catalog"

    def test_missing_catalog_entry_in_rate_limits_defaults_to_20_60(self, monkeypatch):
        monkeypatch.setattr(config, "MODEL_RATE_LIMITS", {"llama": (15, 60)}, raising=False)
        assert resolve_bucket("z-ai/glm-5.3-flash", "alice") == ("catalog", 20, 60)


class TestDemoHalving:
    def test_demo_account_gets_half_the_catalog_bucket(self, monkeypatch):
        monkeypatch.setattr(config, "DEMO_EPHEMERAL_ENABLED", True, raising=False)
        assert resolve_bucket("z-ai/glm-5.3-flash", "demo_abc123") == ("catalog", 10, 60)

    def test_demo_account_role_bucket_is_unaffected(self, monkeypatch):
        # Halving only applies to the shared catalog bucket, per spec — role
        # buckets (llama/coder/reasoning) are untouched by the demo flag.
        monkeypatch.setattr(config, "DEMO_EPHEMERAL_ENABLED", True, raising=False)
        assert resolve_bucket(LLAMA, "demo_abc123") == ("llama", 15, 60)

    def test_flag_off_demo_account_gets_full_catalog_bucket(self, monkeypatch):
        monkeypatch.setattr(config, "DEMO_EPHEMERAL_ENABLED", False, raising=False)
        assert resolve_bucket("z-ai/glm-5.3-flash", "demo_abc123") == ("catalog", 20, 60)

    def test_non_demo_username_gets_full_catalog_bucket(self, monkeypatch):
        monkeypatch.setattr(config, "DEMO_EPHEMERAL_ENABLED", True, raising=False)
        assert resolve_bucket("z-ai/glm-5.3-flash", "alice") == ("catalog", 20, 60)

    def test_halving_never_rounds_down_to_zero(self, monkeypatch):
        monkeypatch.setattr(config, "DEMO_EPHEMERAL_ENABLED", True, raising=False)
        monkeypatch.setattr(config, "MODEL_RATE_LIMITS", {"catalog": (1, 60)}, raising=False)
        _, limit, _ = resolve_bucket("z-ai/glm-5.3-flash", "demo_abc123")
        assert limit >= 1
