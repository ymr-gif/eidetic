import { useEffect, useState } from 'react'
import s, { NOMINAL, ALERT, AMBER, FG4, INFOBLUE } from '../../../lib/chatStyles.js'

const STATUS_COLOR = {
  live: NOMINAL,
  not_found: ALERT,
  gone: ALERT,
  delisted: ALERT,
  timeout: AMBER,
  error: AMBER,
}

export default function CatalogRow({ row, onSave, saving }) {
  const [priceIn, setPriceIn] = useState(row.price_in ?? '')
  const [priceOut, setPriceOut] = useState(row.price_out ?? '')
  const [contextWindow, setContextWindow] = useState(row.context_window ?? '')
  const [err, setErr] = useState('')

  // resync local edit buffers after an external row update (e.g. this row's own successful save)
  useEffect(() => {
    setPriceIn(row.price_in ?? ''); setPriceOut(row.price_out ?? ''); setContextWindow(row.context_window ?? '')
  }, [row.price_in, row.price_out, row.context_window])

  async function commitPrice() {
    const inVal = priceIn === '' ? null : Number(priceIn)
    const outVal = priceOut === '' ? null : Number(priceOut)
    if (inVal === (row.price_in ?? null) && outVal === (row.price_out ?? null)) return
    // backend 400s if only one of the pair is given — always send both together.
    const res = await onSave(row.id, { price_in: inVal, price_out: outVal })
    setErr(res.ok ? '' : res.error)
  }

  async function commitContextWindow() {
    const val = contextWindow === '' ? null : Number(contextWindow)
    if (val === (row.context_window ?? null)) return
    const res = await onSave(row.id, { context_window: val })
    setErr(res.ok ? '' : res.error)
  }

  async function toggleEnabled() {
    const res = await onSave(row.id, { enabled: !row.enabled })
    setErr(res.ok ? '' : res.error)
  }

  const statusColor = STATUS_COLOR[row.status] || FG4

  return (
    <div style={s.modelRow}>
      <div style={s.modelRowTop}>
        <div style={s.modelIdentity}>
          <span style={s.modelLabelText}>{row.label || row.id}</span>
          <span style={s.modelIdText}>{row.id}</span>
        </div>
        <div style={s.modelBadgeRow}>
          {row.role && <span style={{ ...s.statusBadge, color: INFOBLUE, borderColor: INFOBLUE }}>{row.role}</span>}
          <span style={{ ...s.statusBadge, color: statusColor, borderColor: statusColor }}>{row.status}</span>
          <button
            onClick={toggleEnabled}
            disabled={saving || row.is_role_model}
            title={row.is_role_model ? 'Role models stay enabled — swap the role in .env first' : (row.enabled ? 'Disable' : 'Enable')}
            style={row.enabled ? s.toggleOn : s.toggleOff}>
            {row.enabled ? 'ENABLED' : 'DISABLED'}
          </button>
        </div>
      </div>

      <div style={s.modelMetaRow}>
        {row.latency_ms != null && <span>{row.latency_ms}ms</span>}
        {row.fail_count > 0 && <span style={{ color: AMBER }}>{row.fail_count} fail</span>}
        {row.reasoning && <span style={{ color: AMBER }}>reasoning leak risk</span>}
      </div>

      <div style={s.modelEditRow}>
        <div style={s.modelEditField}>
          <span style={s.modelEditLabel}>IN $/1M</span>
          <input type="number" step="0.01" min="0" max="100" value={priceIn}
            onChange={e => setPriceIn(e.target.value)} onBlur={commitPrice} style={s.numInput} />
        </div>
        <div style={s.modelEditField}>
          <span style={s.modelEditLabel}>OUT $/1M</span>
          <input type="number" step="0.01" min="0" max="100" value={priceOut}
            onChange={e => setPriceOut(e.target.value)} onBlur={commitPrice} style={s.numInput} />
        </div>
        <div style={s.modelEditField}>
          <span style={s.modelEditLabel}>CTX</span>
          <input type="number" step="1024" min="0" value={contextWindow}
            onChange={e => setContextWindow(e.target.value)} onBlur={commitContextWindow} style={{ ...s.numInput, width:'72px' }} />
        </div>
      </div>
      {err && <div style={{ fontSize:'11px', color:ALERT, marginTop:'0.3rem' }}>{err}</div>}
    </div>
  )
}
