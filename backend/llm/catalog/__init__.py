"""Live NIM model catalog (HANDOFF Phase 3, 2026-09-20).

  labels.py  — derive_label() / is_chat_candidate() for a raw NIM model id
  store.py   — model_catalog row CRUD (upsert scan results, admin edits)
  cache.py   — in-process snapshot (ensure_fresh/publish/is_available/get_entry)
  pricing.py — get_pricing(): catalog override -> MODEL_PRICING -> default
  scanner.py — run_scan(): GET /v1/models, probe, upsert, publish

See backend/CLAUDE.md "Graphify" + HANDOFF.md Phase 3 for the full design.
"""
