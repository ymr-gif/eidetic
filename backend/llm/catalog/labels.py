"""Human labels + chat-candidate filtering for raw NIM model ids.

Pure string logic — no I/O, no config reads — so both the scanner and any
caller can use it without pulling in DB/Redis dependencies.
"""

# Substrings that mark an id as a non-chat model (embedder, reranker, safety
# classifier, ...). Matched case-insensitively against the whole id (including
# the vendor prefix) so e.g. "nvidia/nv-embedqa-e5-v5" is excluded via "embed".
_EXCLUDE_SUBSTRINGS = (
    "embed", "rerank", "retriev", "reward", "safety", "guard",
    "parse", "ocr", "asr", "tts", "vlm-embed",
)


def is_chat_candidate(model_id: str) -> bool:
    """True unless `model_id` looks like a non-chat model."""
    low = model_id.lower()
    return not any(term in low for term in _EXCLUDE_SUBSTRINGS)


def derive_label(model_id: str) -> str:
    """Best-effort human label from a raw id.

    "z-ai/glm-5.3-flash" -> "Glm 5.3 Flash"; "openai/gpt-oss-20b" -> "Gpt Oss 20b".
    Falls back to the raw id if nothing splits out (never returns empty)."""
    name = model_id.split("/")[-1]
    parts = [p for p in name.replace("_", "-").split("-") if p]
    if not parts:
        return model_id
    words = [p.upper() if len(p) <= 3 and any(c.isdigit() for c in p) else p.capitalize() for p in parts]
    return " ".join(words)
