# Configuration

See `.env.example` for all variables. Commonly changed:

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | NIM API key (required) |
| `JWT_SECRET_KEY` | — | JWT signing secret (required) |
| `DATABASE_URL` | — | PostgreSQL via pgBouncer |
| `REDIS_URL` | — | Redis connection string |
| `NEO4J_URI` | — | Bolt URI; omit to disable graph memory |
| `REQUIRE_INVITE` | `false` | Gate registration behind invite tokens |
| `REQUEST_TIMEOUT` | `30` | NIM request timeout (seconds) |
| `MAX_CONCURRENT_REQUESTS` | `10` | Max parallel NIM requests (cap 50) |
| `MODEL_LLAMA` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Override general model |
| `MODEL_CODER` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Override coder model |
| `MODEL_REASONING` | `nvidia/nemotron-3-super-120b-a12b` | Override reasoning model |
| `MODEL_EMBEDDING` | `nvidia/nemotron-3-embed-1b` | Changing this triggers a full re-embed (now 2048d) |
| `BACKUP_SCHEDULE` | `0 2 * * *` | Cron for automated DB backup |
| `LLM_BACKEND` | `nim` | `nim` \| `homeserver` — flip to local llama.cpp stack |
| `INTEGRATION_SECRET` | — | Fernet key (44-char base64url) for connector credentials; OAuth endpoints 503 without it |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | — | Shared Google OAuth app (Drive + Calendar + Gmail) |
| `NOTION_CLIENT_ID` / `NOTION_CLIENT_SECRET` | — | Notion OAuth app |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | — | GitHub OAuth app |
| `IMAGE_OCR_ENABLED` | `false` | CPU PaddleOCR for images + scanned PDFs |
| `VOICE_ENABLED` | `false` | Enable `POST /api/transcribe` voice input |

## Models

NVIDIA retires and adds models on its hosted endpoint within days, so treat the table below as a
snapshot (2026-09-25), not a guarantee. The live answer is the model catalog: a scheduled job probes
every listed chat model every six hours and records what actually responds, and admins enable which of
those users can pick (`ADMIN -> MODELS` in the UI, or `GET /api/admin/models`).

| Role | Model | Env var |
|---|---|---|
| General | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `MODEL_LLAMA` |
| Coder | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `MODEL_CODER` |
| Reasoning | `nvidia/nemotron-3-super-120b-a12b` | `MODEL_REASONING` |
| Embedding | `nvidia/nemotron-3-embed-1b` (2048d) | `MODEL_EMBEDDING` |

Model selection priority: `per-request override > conversation lock > keyword router`

> Setting `LLM_BACKEND=homeserver` (live-toggleable via `/admin/env`) repoints inference to a local OpenAI-compatible endpoint, collapses to a single Mixtral model, sizes context to 32k, swaps the embedder to `bge-large-en-v1.5` (1024-d, no re-embed), and drops the NIM-key requirement.
