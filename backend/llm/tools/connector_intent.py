"""Per-connector intent latch signals (Q3 Task B + calendar/gmail generalization).

Generalizes the Drive intent latch to every capability-gated OAuth connector
(drive, calendar, gmail). Each connector's tool schemas are withheld from the
model until genuine intent for THAT connector appears in the session — an
embedding cosine of the turn's `query_emb` against a per-connector centroid
derived at boot from example phrases. Fixes the capability-only over-fire where
the tool-eager 70B called connector tools on greetings/chit-chat (e.g.
`drive_list_files` / `calendar_list_events` / `calendar_search_events {"query":"hi"}`).

NOT a keyword match — the cosine generalizes past the exact example words. The
phrase lists are example sentences for centroid derivation only.

Mechanism (resolved per turn in `llm/service/stream.py`): score the reused `query_emb` by
NEAREST-EXAMPLE — the MAX cosine over `get_anchors(connector)` (every phrase's embedding, not a
single mean); `>= max(INTENT_THRESHOLDS[connector], FLOOR_THRESHOLD)` → latch (Redis
`{connector}_latched:{conv_id}`, latch-then-serve same turn, sticky 1h). The gate for each
connector is `ctx.{connector}_active AND ctx.{connector}_latched`.

NOTE (2026-07-01): scoring changed from mean-centroid cosine to nearest-example MAX. Max runs
HIGHER than mean, so a mean-tuned threshold under-fires — nearest-example needs its own tuning.

Embedder-specific: centroids auto-regenerate at boot under any embedder, but the THRESHOLDS below
are tuned to the LIVE embedder's score geometry and will be WRONG after any embedder swap.
**Re-tuned 2026-09-20 for `nvidia/nemotron-3-embed-1b` (2048-d)** — see `INTENT_THRESHOLDS`/
`FLOOR_THRESHOLD` comments below for the measured numbers. nemotron-3-embed-1b's cosine range runs
noticeably LOWER/TIGHTER than the retired `nv-embedqa-e5-v5` (1024-d) — genuine positives cluster
~0.2–0.9 vs. e5's ~0.6–1.0 — so the old 0.60/0.60/0.65 + 0.70 floor do NOT port; a straight
"recalibrate the old numbers" pass would have left every connector permanently unreachable (e5's
floor alone exceeds nemotron's strongest drive positive). On the eventual bge-large-en-v1.5
(homeserver) swap, re-run `tests/{connector}_intent_eval.jsonl` and re-tune again — do not port
these nemotron numbers either. See backend/CLAUDE.md → LLM_BACKEND invariant.

Cross-connector talk: connector requests share a "check my X / find my Y" possessive
structure, so e.g. a gmail request scores high on the drive & calendar centroids too
(measured: gmail turns latch drive 13/20 @0.60). To stop one request latching multiple
connectors, the latch flip in `generate_stream` is **single-winner** — among active,
not-yet-latched connectors it latches only the top-scoring one (the correct connector
is the argmax ~always: gmail turns score gmail ~0.75 vs drive/cal ~0.60). This shrinks a
cross-talk/task-imperative false latch from all connectors to one.

Known ceiling (per connector): a single-centroid cosine separates greetings/chit-chat
cleanly from connector requests, but connector-vs-connector-vs-task-imperative bands
overlap — thresholds are precision-biased ("fail toward fewer tools"). The global
`FLOOR_THRESHOLD` (see its own comment for the current value) rejects weak winners that
only "won" because every connector scored low — a confident wrong latch is worse than
a humble abstention. The latch is sticky 1h, so a false latch poisons that one connector
for the session until TTL. See BUGS.md residual.
"""

from __future__ import annotations

import asyncio
import logging
import math

from llm.embeddings import embed as embed_text

logger = logging.getLogger("connector_intent")

# Example intent sentences per connector — the ONLY place example vocabulary lives.
# Centroid = normalized mean of these embeddings, derived at boot, never hardcoded.
# Each anchored on the connector's nouns (files/drive · schedule/calendar/events ·
# email/inbox/messages); structure varied to avoid clustering on one verb.
INTENT_PHRASES: dict[str, list[str]] = {
    "drive": [
        "what files do I have in my drive",
        "show me the documents in my google drive",
        "open the document I have about the budget",
        "find a file in my drive",
        "search my google drive for a document",
        "list the files in my drive",
        "do I have a document about this in my files",
        "pull up a file from my drive",
        "look in my drive for a spreadsheet",
        "read one of my documents",
        "what is saved in my google drive",
        "find my file about the project",
        "open a folder in my drive",
        "get a document from my files",
        "which documents are in my drive",
        "check my google drive for a file",
        "retrieve a file from my documents",
        "where is my document saved in drive",
        # terse anchors — genuine short requests carry the connector noun (file/doc/spreadsheet)
        "open my document",
        "read that file",
        "grab the spreadsheet",
        "pull up my doc",
        "which file has this",
    ],
    "calendar": [
        "what is on my calendar today",
        "do I have any meetings this week",
        "show me my schedule",
        "when is my next appointment",
        "add an event to my calendar",
        "schedule a meeting for tomorrow",
        "what events do I have on friday",
        "am I free this afternoon",
        "put a reminder on my calendar",
        "check my calendar for next monday",
        "do I have anything booked tomorrow",
        "create a calendar event for the call",
        "what is my agenda for today",
        "move my three pm meeting",
        "cancel the event on thursday",
        "list my upcoming events",
        "when am I meeting with the team",
        "block off time on my calendar",
        # terse anchors — carry the connector noun (meeting/calendar/agenda/schedule)
        "when is my meeting",
        "what's my agenda",
        "put this on my calendar",
        "do I have anything scheduled",
        "any meetings on my calendar",
    ],
    "gmail": [
        "check my email",
        "do I have any new messages",
        "show me my latest emails",
        "search my inbox for the invoice",
        "did I get an email from john",
        "what is in my inbox",
        "read my most recent email",
        "find the email about the meeting",
        "do I have any unread messages",
        "look up that email from support",
        "what emails did I get today",
        "show me messages from my boss",
        "find emails about the project",
        "open the email with the receipt",
        "check for new mail",
        "did anyone email me about the order",
        "search my gmail for the contract",
        "read the email from the bank",
        # terse anchors — carry the connector noun (email/inbox/mail/message)
        "show my inbox",
        "did I get any email",
        "open my messages",
        "who emailed me",
        "look through my mail",
    ],
}

# Cosine thresholds separating connector-intent from non-intent turns. RE-TUNED 2026-09-20 for
# `nvidia/nemotron-3-embed-1b` (2048-d) against `tests/{connector}_intent_eval.jsonl` (20 positives
# + 20 easy negatives each), via `tests/latch/retune_thresholds.py`. Selection rule: the LOWEST
# threshold (0.01 steps) that yields ZERO false positives on that connector's 20 negatives —
# precision-biased, same philosophy as the original e5 tuning. Measured (embed each eval line once
# with `llm/embeddings.embed(text, input_type="query")`, score with the real `intent_score`):
INTENT_THRESHOLDS: dict[str, float] = {
    "drive": 0.21,     # precision 1.00, recall 20/20 (1.00), 0 FP — clean separation
    "calendar": 0.29,  # precision 1.00, recall 20/20 (1.00), 0 FP — clean separation
    "gmail": 0.44,     # precision 1.00, recall 19/20 (0.95), 0 FP — misses one weak positive
                       # ("find that message about the refund", scores 0.316)
}

# Global floor: a connector must clear BOTH its per-connector threshold AND this floor to latch.
# Prevents latching on a weak winner that only "won" because every connector scored low — a
# confident wrong latch is worse than a humble abstention. Already-latched connectors (sticky TTL)
# unaffected.
#
# RE-TUNED 2026-09-20 for nemotron-3-embed-1b. Cross-talk was measured (each connector's 20
# positives scored against the OTHER TWO connectors' anchors, via `tests/latch/retune_thresholds.py`)
# to inform this floor per the standard methodology: MAX off-target score = 0.695 (a "look in my
# drive for the invoice" drive-positive scores 0.695 against the GMAIL anchors — "invoice" is an
# explicit gmail anchor noun — higher than its OWN drive score of 0.545). Setting FLOOR at/above that
# literal max (~0.70, mirroring the old floor's numeric style) was REJECTED: nemotron's cosine range
# runs far tighter than e5's did, and drive's single strongest positive in the eval set only scores
# 0.643 — a 0.70 floor would make the drive connector's latch STRUCTURALLY UNREACHABLE (0/20 recall),
# and would cut calendar to 4/20 (0.20) and gmail to 10/20 (0.50). That is not "precision-biased," it
# is "disabled," so it was not shipped. Chosen instead: 0.30, set from the negative/noise ceiling
# measured across all three connectors' own eval sets (generic non-connector chit-chat tops out at
# 0.206 drive / 0.288 calendar / 0.262 gmail-excluding-cross-connector-items) — high enough to reject
# near-zero-signal winners, low enough to only mildly cost recall: drive 20/20→18/20 (loses its two
# weakest positives, ~0.22/0.30), calendar unaffected (still 20/20, its own threshold already clears
# 0.30), gmail unaffected (own threshold 0.44 > floor). The 0.695 cross-talk case above is a genuine
# anchor-vocabulary-overlap problem (shared nouns like "invoice") that NO floor value can fix without
# gutting recall — it needs a margin-based single-winner rule or disjoint anchor vocabulary, both out
# of scope for a threshold-only re-tune; flagged in BUGS.md as a residual.
FLOOR_THRESHOLD = 0.30

_anchors: dict[str, list[list[float]] | None] = {}
_locks: dict[str, asyncio.Lock] = {c: asyncio.Lock() for c in INTENT_PHRASES}


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


async def _embed_phrase(text: str, *, retries: int = 6) -> list[float] | None:
    # Sequential + retry: the NIM embedder rate-limits (429) a concurrent burst,
    # which would silently drop phrases and make the centroid non-deterministic —
    # and the tuned threshold assumes the FULL phrase set. One-time boot cost.
    for attempt in range(retries):
        v = await embed_text(text, input_type="query")
        if v:
            return v
        await asyncio.sleep(1.0 * (attempt + 1))
    return None


async def _build_anchors(connector: str) -> list[list[float]] | None:
    # NEAREST-EXAMPLE model: keep every phrase's normalized embedding, not a single mean.
    # A single mean-centroid punishes short/terse genuine requests ("read that file") — they
    # sit far from the average of long descriptive phrases and score low, overlapping vague
    # connector-adjacent turns. Scoring against the MAX over individual phrases lets a terse
    # request match a nearby example instead of the diluted average.
    # input_type MUST match the request-time query embed: helpers.py embeds the user message
    # as "query"; the e5 embedder is ASYMMETRIC (query/passage occupy different subspaces).
    phrases = INTENT_PHRASES[connector]
    vecs = []
    for p in phrases:
        v = await _embed_phrase(p)
        if v:
            vecs.append(_normalize(list(v)))
    if len(vecs) < len(phrases):
        # Partial set → the anchor cloud is incomplete and the tuned threshold assumes the
        # FULL set. Refuse rather than ship a mistuned latch; the lazy path retries (self-heal).
        logger.warning("[connector_intent] %s anchors incomplete (%d/%d) — leaving unbuilt for retry",
                       connector, len(vecs), len(phrases))
        return None
    logger.info("[connector_intent] %s anchors built from %d phrases (dim=%d)",
                connector, len(vecs), len(vecs[0]))
    return vecs


async def get_anchors(connector: str) -> list[list[float]] | None:
    """Boot-derived normalized phrase embeddings for `connector`, built once per process
    (lazy + locked). None if the embedder was unreachable at build time → callers treat as
    "no signal" (score 0.0). The next call retries, so a transient outage self-heals."""
    if _anchors.get(connector) is None:
        async with _locks[connector]:
            if _anchors.get(connector) is None:
                _anchors[connector] = await _build_anchors(connector)
    return _anchors.get(connector)


async def warm_centroids() -> None:
    """Optional startup warm of all connector anchor sets so the first connector-active
    request doesn't pay the phrase-embedding cost. Safe to fire-and-forget from lifespan;
    failures are swallowed (the lazy path retries). (Name kept for the lifespan import.)"""
    for connector in INTENT_PHRASES:
        try:
            await get_anchors(connector)
        except Exception:
            logger.warning("[connector_intent] warm %s failed; will build lazily", connector, exc_info=True)


async def intent_score(connector: str, query_emb) -> float:
    """Nearest-example intent signal: the MAX cosine of the query embedding against any of
    `connector`'s phrase embeddings.

    `query_emb` is None/empty (no embed this turn) → 0.0, failing toward NOT latching. Anchors
    unbuildable (embedder down at boot) → 0.0 likewise.
    """
    if not query_emb:
        logger.debug("[intent_score] %s query_emb=%s → 0.0", connector, type(query_emb).__name__ if query_emb is not None else "None")
        return 0.0
    anchors = await get_anchors(connector)
    if not anchors:
        return 0.0
    q = _normalize(list(query_emb))
    return max(float(sum(a * b for a, b in zip(q, anc))) for anc in anchors)
