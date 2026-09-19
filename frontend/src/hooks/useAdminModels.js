import { useState, useCallback, useEffect } from 'react'

// Admin model catalog — GET/PATCH /api/admin/models, POST /api/admin/models/rescan
// (backend/HANDOFF.md Phase 3: admin-only, require_role("admin")). Same fetch-on-open
// pattern as useAdmin.js's invite list.
export default function useAdminModels(token) {
  const [modelsOpen, setModelsOpen] = useState(false)
  const [rows, setRows] = useState([])
  const [scanMeta, setScanMeta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [rescanning, setRescanning] = useState(false)
  const [rescanMsg, setRescanMsg] = useState('')
  const [savingId, setSavingId] = useState(null)

  const authHeaders = { 'Authorization': `Bearer ${token}` }

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const qs = new URLSearchParams()
      if (query.trim()) qs.set('q', query.trim())
      if (statusFilter) qs.set('status', statusFilter)
      const r = await fetch(`/api/admin/models?${qs}`, { headers: authHeaders })
      if (r.ok) {
        const data = await r.json()
        setRows(data.models || [])
        setScanMeta(data.scan_meta || null)
        setError('')
      } else {
        setError(`Failed to load models (${r.status})`)
      }
    } catch {
      setError('Failed to load models')
    } finally {
      setLoading(false)
    }
  }, [token, query, statusFilter])

  useEffect(() => { if (modelsOpen) load() }, [modelsOpen])
  // debounce search/filter while the pane stays open
  useEffect(() => {
    if (!modelsOpen) return
    const tid = setTimeout(load, 300)
    return () => clearTimeout(tid)
  }, [query, statusFilter])

  async function patchModel(id, patch) {
    setSavingId(id)
    try {
      const r = await fetch(`/api/admin/models/${encodeURIComponent(id)}`, {
        method: 'PATCH', headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      })
      if (r.ok) {
        const updated = await r.json()
        setRows(prev => prev.map(row => row.id === id ? updated : row))
        return { ok: true }
      }
      const data = await r.json().catch(() => ({}))
      const msg = r.status === 409
        ? 'Cannot disable a role model — swap it in .env first'
        : (data.detail || `Update failed (${r.status})`)
      return { ok: false, error: msg }
    } catch {
      return { ok: false, error: 'Request failed' }
    } finally {
      setSavingId(null)
    }
  }

  async function rescan() {
    setRescanning(true)
    setRescanMsg('')
    try {
      const r = await fetch('/api/admin/models/rescan', { method: 'POST', headers: authHeaders })
      if (r.status === 202) { setRescanMsg('Scan queued…'); setTimeout(load, 4000) }
      else if (r.status === 409) setRescanMsg('Scan already running')
      else if (r.status === 400) setRescanMsg('Rescan unavailable — home-server mode')
      else setRescanMsg(`Rescan failed (${r.status})`)
    } catch {
      setRescanMsg('Rescan request failed')
    } finally {
      setRescanning(false)
    }
  }

  return {
    modelsOpen, setModelsOpen,
    rows, scanMeta, loading, error,
    query, setQuery, statusFilter, setStatusFilter,
    rescanning, rescanMsg, rescan,
    patchModel, savingId,
    reload: load,
  }
}
