"""get_pricing() lookup order (HANDOFF Phase 3): catalog override -> static
MODEL_PRICING -> DEFAULT_MODEL_PRICE_IN/OUT. Also locks that no call site
outside llm/catalog/pricing.py itself still reads MODEL_PRICING.get(...) —
that pattern billed an unpriced model at $0, silently defeating the demo
per-account/global cost caps.

Unit tier — no DB/Redis/NIM.
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL",   "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL",      "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import pytest

import config
from llm.catalog import cache as catalog_cache
from llm.catalog.pricing import get_pricing

LLAMA = config.MODELS["llama"]


@pytest.fixture(autouse=True)
def _reset_catalog():
    catalog_cache._reset_for_tests()
    yield
    catalog_cache._reset_for_tests()


def _seed(model_id: str, *, price_in=None, price_out=None):
    catalog_cache._replace_snapshot({
        **catalog_cache._snapshot,
        model_id: {
            "id": model_id, "label": model_id, "status": "live", "enabled": True,
            "fail_count": 0, "price_in": price_in, "price_out": price_out,
            "context_window": None, "latency_ms": None, "supports_tools": None,
            "reasoning": None, "request_extras": None, "min_max_tokens": None,
        },
    })


class TestGetPricing:
    def test_static_table_used_when_no_catalog_entry(self):
        assert get_pricing(LLAMA) == config.MODEL_PRICING[LLAMA]

    def test_unpriced_unknown_model_gets_default_rate_not_zero(self):
        assert get_pricing("nobody/made-this-up") == {
            "input": config.DEFAULT_MODEL_PRICE_IN,
            "output": config.DEFAULT_MODEL_PRICE_OUT,
        }
        assert config.DEFAULT_MODEL_PRICE_IN > 0
        assert config.DEFAULT_MODEL_PRICE_OUT > 0

    def test_catalog_override_beats_static_table(self):
        _seed(LLAMA, price_in=9.0, price_out=18.0)
        assert get_pricing(LLAMA) == {"input": 9.0, "output": 18.0}

    def test_catalog_entry_with_only_one_price_set_falls_back_to_static(self):
        # Admin PATCH validation requires the pair together, but pricing()
        # itself should still degrade safely if that ever isn't true.
        _seed(LLAMA, price_in=9.0, price_out=None)
        assert get_pricing(LLAMA) == config.MODEL_PRICING[LLAMA]

    def test_catalog_entry_for_unpriced_model_used_over_default(self):
        _seed("nobody/made-this-up", price_in=1.0, price_out=2.0)
        assert get_pricing("nobody/made-this-up") == {"input": 1.0, "output": 2.0}


def test_no_model_pricing_get_call_sites_outside_pricing_module():
    """grep lock: MODEL_PRICING.get(...) must not appear as real code anywhere
    except llm/catalog/pricing.py (which mentions it only in its own
    docstring, explaining what it replaces) — every billing call site must go
    through get_pricing()."""
    backend_dir = os.path.join(os.path.dirname(__file__), "..")
    result = subprocess.run(
        ["grep", "-rn", "MODEL_PRICING.get(", backend_dir,
         "--include=*.py", "--exclude-dir=__pycache__", "--exclude-dir=tests"],
        capture_output=True, text=True,
    )
    hits = [
        line for line in result.stdout.splitlines()
        if line.strip() and "llm/catalog/pricing.py" not in line
    ]
    assert hits == [], f"MODEL_PRICING.get(...) found outside pricing.py: {hits}"
