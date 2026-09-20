// Role → live model id (2026-09-20 NIM EOL recovery swap). These are STATIC fallbacks only —
// the source of truth is the live catalog served by GET /api/models (see useModelCatalog.js).
// Use catalog.labelFor(id) for display; MODEL_LABELS below is what labelFor falls back to when
// the catalog hasn't loaded yet or a message persisted an id the catalog no longer lists.
export const MODEL_KEYS = {
  llama:     'openai/gpt-oss-20b',
  coder:     'deepseek-ai/deepseek-v4-flash-0731',
  reasoning: 'nvidia/nemotron-3-super-120b-a12b',
}
export const MODEL_LABELS = {
  [MODEL_KEYS.llama]:     'GPT-OSS 20B',
  [MODEL_KEYS.coder]:     'DeepSeek V4 Flash',
  [MODEL_KEYS.reasoning]: 'Nemotron 3 Super',
  // legacy/dead ids — kept so labels still resolve on messages persisted before the 2026-09-19 EOL
  'meta/llama-3.1-8b-instruct':              'LLaMA 3.1 8B (retired)',
  'deepseek-ai/deepseek-v4-flash':           'DeepSeek V4 (retired)',
  'openai/gpt-oss-120b':                     'GPT-OSS 120B (retired)',
  'meta/llama-3.3-70b-instruct':             'LLaMA 3.3 70B (retired)',
  'nvidia/llama-3.3-nemotron-super-49b-v1':  'Nemotron 49B (retired)',
}
// Reverse lookup: a raw model id that happens to be one of the 3 role ids -> its role key
// ('llama'/'coder'/'reasoning'), else null. Used to normalize a catalog id picked via the
// command palette back onto the matching role pill instead of spawning a duplicate "extra" pill.
export function roleKeyForModelId(id) {
  return Object.keys(MODEL_KEYS).find(k => MODEL_KEYS[k] === id) || null
}
export const MODEL_SUBLABELS = {
  [MODEL_KEYS.llama]:     'fast',
  [MODEL_KEYS.coder]:     'code',
  [MODEL_KEYS.reasoning]: 'reasoning',
}
// Default compare set only — actual compare picks live in useModelParams.compareModels
// (null = this default) and the authoritative id/label pairs come from the `compare_start` SSE event.
export const COMPARE_MODELS = Object.values(MODEL_KEYS)

export const SECTION_COLORS = {
  USER:'#4f9cf0', STACK:'#55d67c', PROJECT:'#f2a33c', CORRECTIONS:'#e5534b', PATTERNS:'#4f9cf0',
  GOALS:'#55d67c', ARCH:'#f2a33c', STATUS:'#55d67c', PENDING:'#f2a33c',
}
