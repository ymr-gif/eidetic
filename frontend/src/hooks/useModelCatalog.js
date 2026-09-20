import { useState, useCallback, useEffect, useRef } from 'react'
import { MODEL_LABELS, roleKeyForModelId } from '../lib/chatConstants.js'

// Throttle for the window-focus refresh — GET /api/models is cheap (catalog_cache.ensure_fresh()
// is a 15s in-process guard server-side) but no need to hit it on every tab switch.
const FOCUS_REFRESH_MIN_MS = 5 * 60 * 1000

// Live model catalog — GET /api/models (role models + every other currently-available catalog
// model, per backend/HANDOFF.md Phase 3). Source of truth for pill/palette/lock labels; falls
// back to the static MODEL_LABELS map (chatConstants.js) before the first fetch resolves or for
// an id the catalog no longer lists (e.g. a retired id on an old persisted message).
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

  function labelFor(id) {
    if (!id) return id
    // The 3 role ids always use our curated MODEL_LABELS ("GPT-OSS 20B") over the backend's
    // auto-derived catalog label ("Gpt Oss 20B") — admin-set catalog labels still win for every
    // other (non-role) id.
    if (roleKeyForModelId(id)) return MODEL_LABELS[id] || id
    const entry = models.find(m => m.id === id)
    return entry?.label || MODEL_LABELS[id] || id
  }

  function entryFor(id) {
    return models.find(m => m.id === id) || null
  }

  const roleModels = models.filter(m => m.role)
  const catalogModels = models.filter(m => !m.role)

  return { models, roleModels, catalogModels, loading, error, labelFor, entryFor, refresh: load }
}
