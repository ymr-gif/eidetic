"""Pure threshold-sweep math for `retune_thresholds.py` — no network, no embedder.

Kept as its own module (>200-line convention) so the sweep/selection logic can be unit-exercised
independently of the embedding/caching orchestration in `retune_thresholds.py`. Operates on already-
scored rows: `(text, score, is_positive)` triples. The caller is responsible for producing `score`
via the REAL production scorer (`llm.tools.connector_intent.intent_score` /
`llm.closing_intent.closing_score`) so the sweep measures exactly what production would decide.
"""
from __future__ import annotations

import math

Scored = tuple[str, float, bool]


def sweep_thresholds(scored: list[Scored], *, step: float = 0.01, total_pos: int | None = None) -> dict:
    """Sweep candidate thresholds in `step` increments across the observed score range.

    Returns {"table": [(thr, precision, recall, fp), ...], "best_zero_fp": (thr, precision, recall,
    fp) | None, "total_pos": int, "total_neg": int}. `best_zero_fp` is the LOWEST threshold with zero
    false positives — the selection rule specified for this re-tune (precision-biased: pick the most
    permissive threshold that still makes zero mistakes on the negatives).

    `scored` should contain only rows that are actually eligible to be scored in production (e.g.
    for closing, rows that pass the veto/token-band gate — an ineligible row can never become a true
    positive no matter the threshold, so it does not belong in the swept score distribution). Pass
    `total_pos` explicitly when the true positive count includes rows excluded from `scored` this
    way, so recall is computed against the REAL total rather than just the eligible subset.
    """
    total_neg = sum(1 for _, _, p in scored if not p)
    if total_pos is None:
        total_pos = sum(1 for _, _, p in scored if p)
    if not scored:
        return {"table": [], "best_zero_fp": None, "total_pos": total_pos, "total_neg": 0}

    lo = min(s for _, s, _ in scored)
    hi = max(s for _, s, _ in scored)
    steps = []
    t = math.floor(lo / step) * step
    while t <= hi + step + 1e-9:
        steps.append(round(t, 2))
        t += step

    table = []
    best = None
    for t in steps:
        tp = sum(1 for _, s, p in scored if p and s >= t)
        fp = sum(1 for _, s, p in scored if not p and s >= t)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / total_pos if total_pos else 0.0
        table.append((t, precision, recall, fp))
        if fp == 0 and best is None:
            best = (t, precision, recall, fp)

    return {"table": table, "best_zero_fp": best, "total_pos": total_pos, "total_neg": total_neg}


def print_sweep_table(label: str, result: dict) -> None:
    print(f"\n--- {label} --- (positives={result['total_pos']} negatives={result['total_neg']})")
    print(f"{'thr':>6} {'prec':>6} {'recall':>7} {'FP':>4}")
    best = result["best_zero_fp"]
    for t, p, r, fp in result["table"]:
        marker = "  <== zero-FP pick" if best and t == best[0] else ""
        print(f"{t:6.2f} {p:6.3f} {r:7.3f} {fp:4d}{marker}")
    if best:
        print(f"BEST (lowest zero-FP threshold): thr={best[0]:.2f} precision={best[1]:.3f} "
              f"recall={best[2]:.3f} FP={best[3]}")
    else:
        print("BEST: NO threshold in the swept range reaches zero FP — see module docstring "
              "'best trade-off' guidance; do not silently ship a threshold with unmeasured FP risk.")


def cross_talk_summary(cross_scores: dict[tuple[str, str], float]) -> float:
    """`cross_scores` maps (connector, other_connector) -> max cosine of `connector`'s positives
    against `other_connector`'s anchors. Prints a sorted table and returns the overall max."""
    print("\n--- cross-talk (each connector's positives vs OTHER connectors' anchors) ---")
    max_overall = 0.0
    for (c, other), score in sorted(cross_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>8} positives vs {other:>8} anchors: max={score:.3f}")
        max_overall = max(max_overall, score)
    print(f"\nMAX cross-talk score overall: {max_overall:.3f}")
    return max_overall
