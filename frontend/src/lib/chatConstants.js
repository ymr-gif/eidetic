// Role → model id. Model ids churn every few days, so this is a STATIC PRE-FETCH FALLBACK
// ONLY, used for first paint before GET /api/models resolves, or if that fetch fails outright.
// The source of truth at runtime is the live catalog — useModelCatalog.js's `roleModels` /
// `roleKeyForId` / `labelFor`. Every consumer should read through the hook, never this object,
// except the hook itself (which falls back to these values while `models` is still empty).
// Last synced to the live roles 2026-09-25 (nemotron-3-nano-omni EOL recovery swap: llama AND
// coder both route to the same nano-omni id now; reasoning was unaffected).
export const MODEL_KEYS = {
  llama:     'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  coder:     'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  reasoning: 'nvidia/nemotron-3-super-120b-a12b',
}
export const MODEL_LABELS = {
  [MODEL_KEYS.llama]:     'Nemotron 3 Nano Omni',
  [MODEL_KEYS.reasoning]: 'Nemotron 3 Super',
  // legacy/dead ids — kept so labels still resolve on messages persisted before a role's id was
  // last swapped (the catalog no longer returns these, so useModelCatalog.labelFor falls back here)
  'meta/llama-3.1-8b-instruct':              'LLaMA 3.1 8B (retired)',
  'deepseek-ai/deepseek-v4-flash':           'DeepSeek V4 (retired)',
  'openai/gpt-oss-120b':                     'GPT-OSS 120B (retired)',
  'meta/llama-3.3-70b-instruct':             'LLaMA 3.3 70B (retired)',
  'nvidia/llama-3.3-nemotron-super-49b-v1':  'Nemotron 49B (retired)',
  'openai/gpt-oss-20b':                      'GPT-OSS 20B (retired)',
  'deepseek-ai/deepseek-v4-flash-0731':      'DeepSeek V4 Flash (retired)',
}
// Role → short tag ('fast'/'code'/'reasoning') shown next to a model label. Keyed by ROLE, not
// by model id: two roles can share one live id (see MODEL_KEYS above), so an id-keyed map can't
// disambiguate which role actually produced a given reply. Resolve a raw model id to its role
// via useModelCatalog's roleKeyForId(id) first, then look up ROLE_SUBLABELS[role].
export const ROLE_SUBLABELS = { llama: 'fast', coder: 'code', reasoning: 'reasoning' }
// Default compare set fallback only (pre-fetch/offline) — mirrors useModelCatalog's
// `defaultCompareIds`. De-duplicated by id: llama and coder currently share one id, so this is
// 2 entries, not 3. Actual compare picks live in useModelParams.compareModels (null = this
// default) and the authoritative id/label pairs come from the `compare_start` SSE event.
export const COMPARE_MODELS = [...new Set(Object.values(MODEL_KEYS))]

export const SECTION_COLORS = {
  USER:'#4f9cf0', STACK:'#55d67c', PROJECT:'#f2a33c', CORRECTIONS:'#e5534b', PATTERNS:'#4f9cf0',
  GOALS:'#55d67c', ARCH:'#f2a33c', STATUS:'#55d67c', PENDING:'#f2a33c',
}
