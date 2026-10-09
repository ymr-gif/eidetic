# API reference

A hand-written overview. A running backend serves the generated OpenAPI docs at
`http://localhost:8000/docs`, which is the complete and current list.

All endpoints require `Authorization: Bearer <token>` unless noted.

## Auth
```
POST /auth/token              login (form: username, password)
POST /auth/register           register (json: username, password, invite_token?)
POST /auth/me/api-key         generate API key (returned once, stored as hash)
DELETE /auth/me/api-key       revoke API key
```

## Chat
```
POST /chat/stream             SSE streaming (json: message, conversation_id?, model_override?, file_ids?, image_b64?)
POST /chat                    non-streaming (stateless; cost-capped + spend metered)
POST /v1/chat/completions     OpenAI-compatible, streaming + non-streaming (cost-capped + spend metered)
```

**SSE event types:** `token` · `tool_call` · `tool_result` · `status` · `ask_user` · `confirm_write_memory` · `confirm_calendar_write` · `rotated` · `preamble_discard` · `error` · `done`

`done` payload: `model` · `cache_hit` · `fallback_used` · `web_searched` · `url_fetched` · `total_tokens` · `prompt_tokens` · `completion_tokens` · `cost_usd` · `query_type` · `src_count` · `intent` · `grounding` · `activity[]` · `provenance[]` · `conversation_id` · `last_session?`

## Conversations
```
GET    /conversations                 list; ?q= full-text search
GET    /conversations/{id}/messages   message history with activity traces
PATCH  /conversations/{id}            update title or locked_model
DELETE /conversations/{id}
GET    /conversations/{id}/export     markdown export
```

## Files
```
POST /files/upload            multipart upload
GET  /files                   list with chunk status
GET  /files/{id}/content      raw content
PUT  /files/{id}/content      overwrite (saves version)
GET  /files/{id}/versions     version history
POST /files/{id}/versions/{version_id}/restore   restore version
DELETE /files/{id}
```

## Memory & Graph
```
GET  /memory                          compressed memory + conflict count
POST /memory/write                    confirm a memory write (from agent)
GET  /memory/conflicts                list unresolved conflicts
POST /memory/conflicts/{id}/resolve   resolve: keep_a | keep_b | merge | discard_both
GET  /graph/stats
GET  /graph/sample?limit=&entity_type=
DELETE /graph/entities/{name}
POST /graph/prune
```

## Integrations & Notifications
> Paths below are as served on `:8000`; the frontend reaches them via its `/api/*` proxy.
```
GET    /integrations                          list connected sources
POST   /integrations                          create a source
GET/PATCH/DELETE /integrations/{id}           manage a source
POST   /integrations/{id}/sync                trigger a sync
GET    /integrations/oauth/start              begin OAuth (?connector_type=)
GET    /integrations/oauth/callback           OAuth redirect target (no JWT)
POST   /integrations/calendar/execute         run a confirmed calendar write
GET/PATCH /api/notifications/preferences      per-channel notification prefs
POST   /api/notifications/push/subscribe      register a web-push subscription
GET    /api/notifications/vapid-public-key    VAPID public key
POST   /api/transcribe                        voice → text (VOICE_ENABLED)
POST   /auth/me/onboarding-complete           mark onboarding done
```

## Other
```
GET  /search?q=&scope=                unified search across all sources
GET  /usage                           aggregate token + cost stats
GET  /export/full                     ZIP: conversations + files + memory + graph
GET  /goals                           user goals
GET  /scheduled-prompts               automation schedules
POST /webhooks/{user_token}           receive external event (public — no auth header needed)
GET  /auth/me/webhook-token           retrieve webhook token
POST /auth/me/webhook-token           generate / regenerate webhook token
DELETE /auth/me/webhook-token         revoke webhook token
PATCH /auth/me/email                  set / update / clear email address for digest delivery
GET  /system/hardware                 CPU / RAM / GPU / disk / uptime
GET  /health
```

## Admin (role: admin)
```
GET    /admin/users
PATCH  /admin/users/{id}/active         toggle is_active (disabled users 401 everywhere)
PATCH  /admin/users/{id}/cost-limit     set / clear the rolling-window cap
GET    /admin/users/{id}/usage
GET    /admin/audit-log
GET    /admin/env
GET/PUT /admin/env/{key}                read / write one var (live setattr + .env write)
POST   /admin/env/reload                importlib.reload(config)
POST   /admin/re-embed
POST   /admin/memory/reset              soft|hard, dry_run, confirm "RESET <user_id>"
GET    /admin/memory/versions?user_id=
POST   /admin/memory/restore            confirm "RESTORE <user_id>" — reversible rollback
```
