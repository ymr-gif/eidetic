"""Runnable harness — re-tunes the connector-intent / closing-intent embedding-cosine thresholds
against the LIVE embedder. NOT a pytest test (no `test_` prefix; not collected by `pytest.ini`).

Usage (from `backend/`, needs a reachable NIM embedder — this repo's local dev venv cannot reach
`llm_client.client`; run inside the API container, e.g. `docker exec -w /app/backend
docker-api-1 python tests/latch/retune_thresholds.py`):

    python tests/latch/retune_thresholds.py [--cache PATH] [--force-embed]

What it does:
  1. Embeds (ONCE each, cached to --cache so repeat runs / iteration cost nothing further) every
     anchor phrase in `llm.tools.connector_intent.INTENT_PHRASES` + `llm.closing_intent.
     _CLOSING_PHRASES`, and every eval line in `tests/{drive,calendar,gmail,closing}_intent_eval.
     jsonl` — via `llm.embeddings.embed(text, input_type="query")`, the exact call production makes.
  2. Seeds the REAL module-level anchor caches (`connector_intent._anchors`, `closing_intent.
     _anchors`) from the cached vectors, then scores every eval line with the REAL `intent_score` /
     `closing_score` functions — no reimplementation of the cosine math.
  3. Sweeps thresholds (`threshold_sweep.sweep_thresholds`, 0.01 steps) and prints a
     threshold/precision/recall/FP table per connector + closing, selecting the lowest threshold
     with zero false positives.
  4. Measures cross-talk: each connector's positives scored against the OTHER connectors' anchors,
     to inform `FLOOR_THRESHOLD`.
  5. For closing, applies the REAL tier-2 operating-point gate (`llm.router._looks_like_request`
     veto + `_TIER2_MIN_TOKENS`/`_TIER2_MAX_TOKENS` band) before scoring — matching what production
     actually evaluates, not a raw unconditional sweep.

Caching: a JSON file (default under the OS temp dir — never the repo) keyed by exact text ->
embedding vector. Delete it or pass --force-embed to re-embed everything (e.g. after an embedder
change); otherwise this script never re-embeds a text it has already cached — see the Phase 6
HANDOFF budget note (~160 eval lines + ~90 anchor phrases, embedded once).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile

os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _BACKEND_ROOT)

import httpx  # noqa: E402

import config  # noqa: E402
import llm.client as llm_client  # noqa: E402
from llm.embeddings import embed  # noqa: E402
import llm.tools.connector_intent as ci  # noqa: E402
import llm.closing_intent as cl  # noqa: E402
from llm.router import _looks_like_request  # noqa: E402
from threshold_sweep import sweep_thresholds, print_sweep_table, cross_talk_summary  # noqa: E402

CONNECTORS = ["drive", "calendar", "gmail"]
TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_FILES = {
    "drive": os.path.join(TESTS_DIR, "drive_intent_eval.jsonl"),
    "calendar": os.path.join(TESTS_DIR, "calendar_intent_eval.jsonl"),
    "gmail": os.path.join(TESTS_DIR, "gmail_intent_eval.jsonl"),
    "closing": os.path.join(TESTS_DIR, "closing_intent_eval.jsonl"),
}
DEFAULT_CACHE = os.path.join(tempfile.gettempdir(), "eidetic_threshold_retune_cache.json")


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def _cached_embed(text: str, cache: dict[str, list[float] | None]) -> list[float] | None:
    if text in cache:
        return cache[text]
    v = await embed(text, input_type="query")
    cache[text] = v
    return v


async def _embed_all(cache: dict) -> None:
    """Embed every anchor phrase + eval line not already in `cache` (mutated in place)."""
    for phrases in ci.INTENT_PHRASES.values():
        for p in phrases:
            await _cached_embed(p, cache)
    for p in cl._CLOSING_PHRASES:
        await _cached_embed(p, cache)
    for path in EVAL_FILES.values():
        for row in _load_jsonl(path):
            await _cached_embed(row["text"], cache)


def _seed_anchors(cache: dict) -> None:
    """Point the REAL production anchor caches at our captured vectors — no further embed calls."""
    for connector, phrases in ci.INTENT_PHRASES.items():
        ci._anchors[connector] = [ci._normalize(cache[p]) for p in phrases]
    cl._anchors = [cl._normalize(cache[p]) for p in cl._CLOSING_PHRASES]


async def _score_connector(connector: str, cache: dict) -> list[tuple[str, float, bool]]:
    rows = _load_jsonl(EVAL_FILES[connector])
    out = []
    for r in rows:
        s = await ci.intent_score(connector, cache[r["text"]])
        out.append((r["text"], s, r["label"] == connector))
    return out


async def _score_closing(cache: dict) -> tuple[list[tuple[str, float, bool]], int]:
    """Tier-2 operating point: only veto-clear, in-band rows are eligible to score as `closing` —
    mirrors exactly what `_build_stream_context`/`generate_stream` evaluate in production. Returns
    (eligible_scored_rows, total_positives_across_ALL_rows) — an ineligible positive can never
    become a true positive at tier 2 (it falls through to tier 3 in production), so it is dropped
    from the swept distribution but still counted in the recall denominator."""
    rows = _load_jsonl(EVAL_FILES["closing"])
    eligible_scored = []
    total_pos = 0
    for r in rows:
        is_pos = r["label"] == "closing"
        total_pos += is_pos
        veto = _looks_like_request(r["text"])
        band = cl._TIER2_MIN_TOKENS <= len(r["text"].split()) <= cl._TIER2_MAX_TOKENS
        if not veto and band:
            s = await cl.closing_score(cache[r["text"]])
            eligible_scored.append((r["text"], s, is_pos))
    return eligible_scored, total_pos


async def _cross_talk(cache: dict) -> dict[tuple[str, str], float]:
    out = {}
    for c in CONNECTORS:
        pos_rows = [r for r in _load_jsonl(EVAL_FILES[c]) if r["label"] == c]
        for other in CONNECTORS:
            if other == c:
                continue
            scores = [await ci.intent_score(other, cache[r["text"]]) for r in pos_rows]
            out[(c, other)] = max(scores)
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=DEFAULT_CACHE, help=f"embedding cache JSON path (default: {DEFAULT_CACHE})")
    ap.add_argument("--force-embed", action="store_true", help="ignore any existing cache and re-embed everything")
    args = ap.parse_args()

    cache: dict = {}
    if os.path.exists(args.cache) and not args.force_embed:
        with open(args.cache) as f:
            cache = json.load(f)
        print(f"loaded {len(cache)} cached embeddings from {args.cache}")

    missing = sum(
        1 for phrases in ci.INTENT_PHRASES.values() for p in phrases if p not in cache
    ) + sum(1 for p in cl._CLOSING_PHRASES if p not in cache) + sum(
        1 for path in EVAL_FILES.values() for row in _load_jsonl(path) if row["text"] not in cache
    )
    if missing:
        print(f"embedding {missing} new lines live (model={config.MODEL_EMBEDDING})...")
        llm_client.client = httpx.AsyncClient(timeout=30)
        await _embed_all(cache)
        await llm_client.client.aclose()
        with open(args.cache, "w") as f:
            json.dump(cache, f)
        print(f"cache now has {len(cache)} entries, written to {args.cache}")
    else:
        print("cache already covers every anchor + eval line — no network calls needed")

    _seed_anchors(cache)

    print("=" * 90)
    print(f"CONNECTOR SWEEPS ({config.MODEL_EMBEDDING}, input_type=query)")
    print("=" * 90)
    for c in CONNECTORS:
        scored = await _score_connector(c, cache)
        print_sweep_table(c, sweep_thresholds(scored))

    print("\n" + "=" * 90)
    print("CLOSING SWEEP (tier-2 operating point: real veto + real token-band applied)")
    print("=" * 90)
    closing_scored, closing_total_pos = await _score_closing(cache)
    print_sweep_table("closing", sweep_thresholds(closing_scored, total_pos=closing_total_pos))

    print("\n" + "=" * 90)
    print("CROSS-TALK (sets FLOOR_THRESHOLD)")
    print("=" * 90)
    cross_talk_summary(await _cross_talk(cache))


if __name__ == "__main__":
    asyncio.run(main())
