# Roadmap

What is planned for Eidetic. What already works is described in the [README](README.md) and in
[docs/features.md](docs/features.md); this file lists only what is left. Nothing here has a date.

## Planned

- **Self-hosted inference.** NVIDIA NIM is a test backend. The target is a llama.cpp server on
  the home server, reached through the same OpenAI-compatible calls. The switch is in the code
  (`LLM_BACKEND=homeserver`), the llama.cpp services sit in the Compose file behind the
  `homeserver` profile, and `docker/docker-compose.homeserver.cpu.yml` runs a small model on CPU
  to check that tool calls work through llama.cpp. Left: running the full app against llama.cpp
  on the home server.
- **Connectors.** Google Drive, Calendar and Gmail are built end to end and ship switched off
  (`ENABLED_CONNECTOR_TYPES`). Notion and GitHub have OAuth and ingest code but no agent tools
  yet. Outlook and CalDAV are not started.
- **Voice input.** The endpoint and the mic button exist behind `VOICE_ENABLED`, but the only
  transcriber is a stub that returns placeholder text. Left: a real speech-to-text backend
  (`ASR_BACKEND=faster_whisper` is the name reserved for it). Text-to-speech is not started.

## Later

- **Image search beyond OCR.** Images are found through their OCR text (`IMAGE_OCR_ENABLED`).
  Image embeddings and entity extraction from images are not started.
- **More than one host.** Compose can run several API and worker containers
  (`--scale api=N`, `--scale arq-worker=N`); the scheduler has to stay a single container. A
  Redis lock for memory writes is available for that case (`MEMORY_LOCK_BACKEND=redis`). There
  is no multi-host or Kubernetes setup.

## Removed

Both were built and later taken out. The migrations that added and dropped them are still in
`backend/alembic/versions/`.

- **Workspaces.** Collapsed into plain sessions.
- **Agent canvas.** A Neo4j-backed node graph the agent could edit, with its tools.
