import s, { ALERT } from '../../../lib/chatStyles.js'
import { fmtDate } from '../../../lib/chatUtils.js'
import { usePanelProps } from '../PanelPropsContext.js'
import CatalogRow from './CatalogRow.jsx'

const STATUS_OPTIONS = ['', 'live', 'timeout', 'error', 'not_found', 'gone', 'delisted']

export default function ModelCatalogPanel() {
  const p = usePanelProps()
  const { rows, scanMeta, loading, error, query, setQuery, statusFilter, setStatusFilter, rescanning, rescanMsg, rescan, patchModel, savingId, reload } = p.adminModels

  return (
    <div style={s.dockPane}>
      <div style={s.modelsHdr}>
        <div style={s.modelsTitleRow}>
          <span style={s.modelsTitle}>◇ Model Catalog</span>
          <button onClick={reload} style={s.refreshBtn} disabled={loading}>{loading ? '…' : '↻'}</button>
        </div>
        <div style={s.modelsFilterRow}>
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search id or label…"
            style={{ ...s.sideSearchInput, flex:1 }} />
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)} style={s.modelsSelect}>
            {STATUS_OPTIONS.map(opt => <option key={opt} value={opt}>{opt || 'all status'}</option>)}
          </select>
        </div>
        <div style={s.modelsScanRow}>
          <button onClick={rescan} disabled={rescanning} style={s.actionBtn}>
            {rescanning ? 'Queuing…' : '↺ Rescan'}
          </button>
          <span style={s.modelsScanMeta}>
            {rescanMsg || (scanMeta
              ? `last scan ${fmtDate(scanMeta.timestamp)} · ${scanMeta.trigger || ''}`
              : 'no scan recorded yet')}
          </span>
        </div>
      </div>
      <div style={s.modelsBody}>
        {error && <div style={{ fontSize:'13px', color:ALERT, marginBottom:'0.5rem' }}>{error}</div>}
        {loading && rows.length === 0 && <p style={s.emptyMem}>Loading…</p>}
        {!loading && rows.length === 0 && <p style={s.emptyMem}>No models match.</p>}
        {rows.map(row => (
          <CatalogRow key={row.id} row={row} onSave={patchModel} saving={savingId === row.id} />
        ))}
      </div>
    </div>
  )
}
