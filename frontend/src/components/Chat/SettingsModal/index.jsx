import { useMemo, useState } from 'react'
import s, { FG4, FG3, AMBER, ALERT, MONO } from '../../../lib/chatStyles.js'
import { MODEL_KEYS } from '../../../lib/chatConstants.js'
import { usePanelProps } from '../PanelPropsContext.js'

const NOTIF_LABELS = {
  email_digest: 'Email Digest',
  email_scheduled: 'Email Scheduled',
  email_insights: 'Email Insights',
  push_enabled: 'Push',
}

const ROLE_KEYS = ['llama', 'coder', 'reasoning']

export default function SettingsModal() {
  const p = usePanelProps()
  const { settingsOpen, setSettingsOpen, editSysPrompt, setEditSysPrompt, editLockModel, setEditLockModel, saveSettings, settingsSaving, settingsError, convLockModel, convLockModelAvailable } = p.settings
  const { prefs, prefsLoading, vapidPublicKey, pushSupported, togglePref } = p.notificationPrefs || {}
  const catalog = p.catalog
  const [modelFilter, setModelFilter] = useState('')
  const isRolePick = editLockModel === '' || ROLE_KEYS.includes(editLockModel)

  const filteredCatalogModels = useMemo(() => {
    const needle = modelFilter.trim().toLowerCase()
    const list = catalog?.catalogModels || []
    if (!needle) return list.slice(0, 20)
    return list.filter(m => m.id.toLowerCase().includes(needle) || (m.label || '').toLowerCase().includes(needle)).slice(0, 20)
  }, [modelFilter, catalog?.catalogModels])

  if (!settingsOpen) return null
  return (
    <div style={s.settingsModal} onClick={e => e.stopPropagation()}>
      <div style={s.settingsHeader}>
        <span style={s.settingsTitle}>Settings</span>
        <button onClick={() => setSettingsOpen(false)} style={s.closeBtn}>✕</button>
      </div>
      <div style={s.settingsBody}>
        <div style={{ ...s.editLabel, marginTop:0 }}>System Prompt</div>
        <textarea value={editSysPrompt} onChange={e => setEditSysPrompt(e.target.value)}
          rows={5} style={s.editArea} placeholder="You are a helpful assistant specializing in…" />
        <div style={{ ...s.editLabel, marginTop:'1rem', display:'flex', alignItems:'center', gap:'0.5rem' }}>
          Model Lock
          {convLockModel && convLockModelAvailable === false && (
            <span style={{ ...s.statusBadge, color:ALERT, borderColor:ALERT }}>unavailable — using Auto</span>
          )}
        </div>
        <div style={{ ...s.modelPills, marginTop:'0.4rem', flexWrap:'wrap' }}>
          {[['', 'Auto (route)'], ...ROLE_KEYS.map(k => [k, catalog?.labelFor(MODEL_KEYS[k]) || k])].map(([key, label]) => (
            <button key={key} onClick={() => setEditLockModel(key)}
              style={{ ...s.pill, ...(editLockModel === key ? s.pillActive : {}) }}>
              {label}
            </button>
          ))}
        </div>

        <div style={{ marginTop:'0.6rem' }}>
          <input value={modelFilter} onChange={e => setModelFilter(e.target.value)}
            placeholder="Search other live models…" style={s.sideSearchInput} />
          {modelFilter.trim() && (
            <div style={{ maxHeight:'140px', overflowY:'auto', border:`1px solid #2a4160`, borderRadius:'3px', marginTop:'0.3rem' }}>
              {filteredCatalogModels.length === 0 && (
                <div style={{ padding:'0.4rem 0.6rem', fontSize:'12px', color:FG4 }}>No matches</div>
              )}
              {filteredCatalogModels.map(m => (
                <div key={m.id} onClick={() => { setEditLockModel(m.id); setModelFilter('') }}
                  style={{ padding:'0.4rem 0.6rem', fontSize:'13px', color: editLockModel === m.id ? AMBER : FG3, cursor:'pointer' }}>
                  {m.label || m.id}
                </div>
              ))}
            </div>
          )}
          {!isRolePick && editLockModel && (
            <div style={{ display:'flex', alignItems:'center', gap:'0.4rem', marginTop:'0.4rem' }}>
              <span style={{ ...s.pill, ...s.pillActive }}>{catalog?.labelFor(editLockModel) || editLockModel}</span>
              <span style={s.chipX} onClick={() => setEditLockModel('')} title="Clear">✕</span>
            </div>
          )}
        </div>

        {editLockModel && <div style={{ fontSize:'14px', color:FG4, marginTop:'0.4rem' }}>All messages in this conversation will use {catalog?.labelFor(isRolePick ? MODEL_KEYS[editLockModel] : editLockModel) || editLockModel}.</div>}
        {settingsError && <div style={{ fontSize:'13px', color:ALERT, marginTop:'0.5rem' }}>{settingsError}</div>}

        <div style={s.notifSection}>
          <div style={s.notifLabel}>Notifications</div>
          {prefsLoading ? (
            <div style={{ fontFamily:'inherit', fontSize:'15px', color:FG4 }}>Loading preferences...</div>
          ) : prefs ? (
            Object.keys(NOTIF_LABELS).map(key => {
              const isPush = key === 'push_enabled'
              const disabled = isPush && (!vapidPublicKey || !pushSupported)
              const val = prefs[key]
              return (
                <div key={key} style={s.notifRow}>
                  <span style={s.notifText}>{NOTIF_LABELS[key]}</span>
                  <button
                    onClick={() => togglePref(key)}
                    disabled={disabled}
                    style={{
                      padding:'0.2rem 0.55rem', border:'1px solid', borderRadius:'3px',
                      borderColor: val ? AMBER : (disabled ? '#1d2a3a' : '#2a4160'),
                      background: val ? 'rgba(242,163,60,0.10)' : 'none',
                      color: val ? AMBER : (disabled ? '#33465e' : '#8ba3bd'),
                      cursor: disabled ? 'default' : 'pointer',
                      fontFamily:MONO, fontSize:'9px',
                      letterSpacing:'0.08em', textTransform:'uppercase',
                    }}>
                    {val ? 'ON' : 'OFF'}
                  </button>
                </div>
              )
            })
          ) : (
            <div style={{ fontFamily:'inherit', fontSize:'13px', color:ALERT }}>Failed to load preferences</div>
          )}
        </div>
      </div>
      <div style={s.settingsFooter}>
        <button onClick={() => setSettingsOpen(false)} style={s.cancelBtn}>Cancel</button>
        <button onClick={saveSettings} disabled={settingsSaving} style={s.saveBtn}>
          {settingsSaving ? 'Saving...' : 'Save'}
        </button>
      </div>
    </div>
  )
}
