import { useState } from 'react'

// 422 model_unavailable's detail shape from the backend (resolve_model_strict) — a short code,
// not prose — so it gets a friendlier inline message here instead of being shown raw.
function friendlyLockError(status, detail) {
  if (status === 422 && (detail === 'model_unavailable' || detail?.error === 'model_unavailable')) {
    return 'That model is no longer available — pick another or use Auto.'
  }
  return (typeof detail === 'string' && detail) || `Failed to save (${status})`
}

export default function useSettings(token, activeConvId, setConversations) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [editSysPrompt, setEditSysPrompt] = useState('')
  const [editLockModel, setEditLockModel] = useState('')
  const [convSysPrompt, setConvSysPrompt] = useState('')
  const [convLockModel, setConvLockModel] = useState('')
  const [convLockModelAvailable, setConvLockModelAvailable] = useState(true)
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsError, setSettingsError] = useState('')

  const authHeaders = { 'Authorization': `Bearer ${token}` }

  async function saveSettings() {
    if (!activeConvId) return
    setSettingsSaving(true); setSettingsError('')
    try {
      const r = await fetch(`/api/conversations/${activeConvId}`, {
        method: 'PATCH', headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify({ system_prompt: editSysPrompt || null, locked_model: editLockModel || null }),
      })
      if (!r.ok) {
        const data = await r.json().catch(() => ({}))
        setSettingsError(friendlyLockError(r.status, data.detail))
        return
      }
      const data = await r.json()
      setConvSysPrompt(data.system_prompt); setConvLockModel(data.locked_model)
      setConvLockModelAvailable(data.locked_model_available ?? true)
      setConversations(prev => prev.map(c => c.id === activeConvId ? { ...c, system_prompt: data.system_prompt, locked_model: data.locked_model, locked_model_available: data.locked_model_available } : c))
      setSettingsOpen(false)
    } catch {
      setSettingsError('Network error — request failed')
    } finally {
      setSettingsSaving(false)
    }
  }

  // Quick-lock from the command palette — locks to `modelId` without touching the system prompt.
  async function lockModel(modelId) {
    if (!activeConvId) return { ok: false, error: 'No active conversation' }
    try {
      const r = await fetch(`/api/conversations/${activeConvId}`, {
        method: 'PATCH', headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify({ locked_model: modelId }),
      })
      if (!r.ok) {
        const data = await r.json().catch(() => ({}))
        return { ok: false, error: friendlyLockError(r.status, data.detail) }
      }
      const data = await r.json()
      setConvLockModel(data.locked_model); setConvLockModelAvailable(data.locked_model_available ?? true)
      setEditLockModel(data.locked_model || '')
      setConversations(prev => prev.map(c => c.id === activeConvId ? { ...c, locked_model: data.locked_model, locked_model_available: data.locked_model_available } : c))
      return { ok: true }
    } catch {
      return { ok: false, error: 'Network error — request failed' }
    }
  }

  return {
    settingsOpen, setSettingsOpen,
    editSysPrompt, setEditSysPrompt,
    editLockModel, setEditLockModel,
    convSysPrompt,
    convLockModel,
    convLockModelAvailable, setConvLockModelAvailable,
    settingsSaving,
    settingsError,
    saveSettings,
    lockModel,
  }
}
