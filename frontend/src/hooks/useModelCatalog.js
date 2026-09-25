import { useState, useCallback, useEffect, useRef } from 'react'
import { MODEL_KEYS, MODEL_LABELS } from '../lib/chatConstants.js'

// Throttle for the window-focus refresh — GET /api/models is cheap (catalog_cache.ensure_fresh()
// is a 15s in-process guard server-side) but no need to hit it on every tab switch.
const FOCUS_REFRESH_MIN_MS = 5 * 60 * 1000

// Fixed role precedence used everywhere two roles might share one live model id (they do today:
// llama and coder both route to the same nano-omni id). Whichever consumer needs a single answer
// (which pill to highlight, which row to show in a dedup'd list) always prefers the earliest role
// here — llama first, then coder, then reasoning — so the choice is deterministic and consistent
// across the toolbar, command palette and compare picker.
const ROLE_ORDER = ['llama', 'coder', 'reasoning']

// Live model catalog — GET /api/models (role models + every other currently-available catalog
// model, per backend/HANDOFF.md Phase 3). Source of truth for pill/palette/lock labels and for
// role->id / id->role mapping; falls back to the static MODEL_KEYS/MODEL_LABELS map
// (chatConstants.js) before the first fetch resolves, or for an id the catalog no longer lists
// (e.g. a retired id on an old persisted message).
export default function useModelCatalog(token) {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const lastFetchRef = useRef(0)

  const authHeaders = { 'Authorization': `Bearer ${token}` }

  const load = useCallback(async () => {
    if (!token) return
    setLoading(true)
    try {
      const r = await fetch('/api/models', { headers: authHeaders })
      if (r.ok) {
        const data = await r.json()
        setModels(Array.isArray(data.models) ? data.models : [])
        setError(null)
      } else {
        setError(r.status)
      }
    } catch {
      setError('network')
    } finally {
      setLoading(false)
      lastFetchRef.current = Date.now()
    }
  }, [token])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    function onFocus() {
      if (Date.now() - lastFetchRef.current >= FOCUS_REFRESH_MIN_MS) load()
    }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [load])

  // role -> {id, label}. Live catalog entry wins when present; otherwise falls back to the
  // static MODEL_KEYS/MODEL_LABELS pair for that role, so this is always fully populated even
  // before the first fetch resolves.
  const roleModels = {}
  for (const role of ROLE_ORDER) {
    const entry = models.find(m => m.role === role)
    roleModels[role] = entry
      ? { id: entry.id, label: entry.label || MODEL_LABELS[entry.id] || entry.id }
      : { id: MODEL_KEYS[role], label: MODEL_LABELS[MODEL_KEYS[role]] || MODEL_KEYS[role] }
  }

  // role -> whether its id is shared with another role ({llama:true, coder:true, reasoning:false}
  // today). Role-pill UI (ModelToolbar, SettingsModal) uses this to decide whether two pills
  // would otherwise show identical text and need the role sub-label appended to disambiguate.
  const roleCollides = {}
  for (const role of ROLE_ORDER) {
    roleCollides[role] = ROLE_ORDER.some(other => other !== role && roleModels[other].id === roleModels[role].id)
  }

  // id -> every role that currently owns it (e.g. nano-omni id -> ['llama','coder']).
  const rolesById = {}
  for (const role of ROLE_ORDER) {
    const id = roleModels[role].id
    ;(rolesById[id] = rolesById[id] || []).push(role)
  }

  // Flat, de-duplicated view of roleModels for UI that lists "the role models" as rows (compare
  // picker) — one row per unique id, even when two roles share it, since there's only one
  // response to compare either way. `roles` carries every role that owns the id (for a combined
  // tag like "llama/coder"); `role` keeps the single first-owner for back-compat.
  const roleModelList = []
  const seenRoleIds = new Set()
  for (const role of ROLE_ORDER) {
    const m = roleModels[role]
    if (seenRoleIds.has(m.id)) continue
    seenRoleIds.add(m.id)
    roleModelList.push({ id: m.id, label: m.label, role, roles: rolesById[m.id] })
  }
  const defaultCompareIds = roleModelList.map(m => m.id)

  // id -> role key ('llama'/'coder'/'reasoning'), or null if id isn't a current role id.
  // Deterministic when two roles share an id: earliest role in ROLE_ORDER wins, so a catalog id
  // picked via the command palette always resolves to ONE role pill (never both, never neither).
  function roleKeyForId(id) {
    if (!id) return null
    for (const role of ROLE_ORDER) {
      if (roleModels[role].id === id) return role
    }
    return null
  }

  function labelFor(id) {
    if (!id) return id
    // The 3 role ids always use our curated label ("Nemotron 3 Nano Omni") over the backend's
    // auto-derived catalog label ("Gpt Oss 20B") — admin-set catalog labels still win for every
    // other (non-role) id.
    const role = roleKeyForId(id)
    if (role) return roleModels[role].label
    const entry = models.find(m => m.id === id)
    return entry?.label || MODEL_LABELS[id] || id
  }

  function entryFor(id) {
    return models.find(m => m.id === id) || null
  }

  const catalogModels = models.filter(m => !m.role)

  return {
    models, roleModels, roleModelList, roleCollides, defaultCompareIds, catalogModels,
    roleKeyForId, labelFor, entryFor,
    loading, error, refresh: load,
  }
}
