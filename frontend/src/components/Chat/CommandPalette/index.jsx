import { useEffect, useMemo, useRef, useState } from 'react'
import s, { AMBER, NOMINAL, INFOBLUE, TRACK, FG3, FG4 } from '../../../lib/chatStyles.js'
import { MODEL_KEYS, roleKeyForModelId } from '../../../lib/chatConstants.js'
import { usePanelProps } from '../PanelPropsContext.js'

const SRC_COLORS = { files: AMBER, conversations: INFOBLUE, memory: NOMINAL, graph: TRACK, action: FG3 }

// Ctrl+K palette: unified /api/search across everything + static actions.
// Absorbs the old SearchPanel.
export default function CommandPalette({ open, onClose, openDock, onLogout, onToast }) {
  const p = usePanelProps()
  const { search, conv, modelParams, settings, catalog, selectConv } = p
  const [sel, setSel] = useState(0)
  const inputRef = useRef(null)
  const q = search.searchQuery

  // Live catalog matches — id or label substring, 2+ chars (short queries would swamp the list
  // with near-every model). Each match contributes a "select for next turn" action and, with an
  // open conversation, a "lock this conversation" action.
  const modelActions = useMemo(() => {
    if (q.trim().length < 2) return []
    const needle = q.trim().toLowerCase()
    const matches = (catalog?.models || [])
      .filter(m => m.id.toLowerCase().includes(needle) || (m.label || '').toLowerCase().includes(needle))
      .slice(0, 6)
    return matches.flatMap(m => {
      const label = m.label || m.id
      // A role model's raw id resolves to the SAME pill as its role key — select the role key so
      // ModelToolbar highlights the existing role pill instead of adding a duplicate "extra" pill.
      const acts = [{ id: `model-${m.id}`, label: `Model: ${label}`, run: () => modelParams.setSelectedModel(roleKeyForModelId(m.id) || m.id) }]
      if (conv.activeConvId) {
        acts.push({
          id: `lock-${m.id}`,
          label: `Lock conversation to ${label}`,
          run: async () => {
            const res = await settings.lockModel(m.id)
            if (!res.ok && onToast) onToast(res.error || 'Failed to lock model')
          },
        })
      }
      return acts
    })
  }, [q, catalog?.models, conv.activeConvId])

  const actions = useMemo(() => {
    const a = [
      { id: 'new', label: 'New session', run: () => conv.newChat() },
      { id: 'mind', label: 'Open dock · Mind (memory / goals / insights)', run: () => openDock('mind', 'memory') },
      { id: 'files', label: 'Open dock · Files', run: () => openDock('files', 'files') },
      { id: 'ops', label: 'Open dock · Ops (usage / log / automations)', run: () => openDock('ops', 'usage') },
      { id: 'compare', label: `${modelParams.compareMode ? 'Disable' : 'Enable'} compare mode`, run: () => modelParams.setCompareMode(!modelParams.compareMode) },
      { id: 'auto', label: 'Model → Auto routing', run: () => modelParams.setSelectedModel('auto') },
      { id: 'llama', label: `Model → ${catalog?.labelFor(MODEL_KEYS.llama)}`, run: () => modelParams.setSelectedModel('llama') },
      { id: 'coder', label: `Model → ${catalog?.labelFor(MODEL_KEYS.coder)}`, run: () => modelParams.setSelectedModel('coder') },
      { id: 'reasoning', label: `Model → ${catalog?.labelFor(MODEL_KEYS.reasoning)}`, run: () => modelParams.setSelectedModel('reasoning') },
      { id: 'logout', label: 'Log out', run: onLogout },
    ]
    if (conv.activeConvId) {
      const c = conv.conversations.find(x => x.id === conv.activeConvId)
      a.splice(1, 0, { id: 'export', label: 'Export this conversation (markdown)', run: () => conv.exportConv(conv.activeConvId, c?.title || 'conversation') })
    }
    const staticActions = !q.trim() ? a : a.filter(x => x.label.toLowerCase().includes(q.toLowerCase()))
    return [...staticActions, ...modelActions]
  }, [q, conv.activeConvId, conv.conversations, modelParams.compareMode, modelActions, catalog?.models])

  const results = search.searchResults?.results || []
  const rows = useMemo(() => [
    ...results.map(r => ({ kind: 'result', ...r })),
    ...actions.map(a => ({ kind: 'action', source: 'action', title: a.label, run: a.run, id: a.id })),
  ], [results, actions])

  useEffect(() => { setSel(0) }, [q, open])
  useEffect(() => {
    if (open) { inputRef.current?.focus(); search.clearSearch() }
  }, [open])
  // window-level Escape — the input handler misses it when focus is elsewhere
  useEffect(() => {
    if (!open) return
    function onEsc(e) { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onEsc)
    return () => window.removeEventListener('keydown', onEsc)
  }, [open])

  function runRow(row) {
    if (!row) return
    if (row.kind === 'action') row.run()
    else if (row.source === 'conversations') selectConv(row.id)
    else if (row.source === 'files') p.files.viewFile(row.id)
    else if (row.source === 'memory') openDock('mind', 'memory')
    else if (row.source === 'graph') openDock('mind', 'memory')
    onClose()
  }

  function onKey(e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel(i => Math.min(i + 1, rows.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel(i => Math.max(i - 1, 0)) }
    else if (e.key === 'Enter') { e.preventDefault(); runRow(rows[sel]) }
    else if (e.key === 'Escape') { e.preventDefault(); onClose() }
  }

  if (!open) return null
  return (
    <div style={s.paletteOverlay} onClick={onClose}>
      <div style={s.palette} onClick={e => e.stopPropagation()}>
        <input ref={inputRef} value={q} onChange={e => search.setSearchQuery(e.target.value)}
          onKeyDown={onKey} placeholder="Search conversations, files, memory — or type a command…"
          style={s.paletteInput} />
        <div style={s.paletteBody}>
          {search.searchLoading && <div style={{ ...s.paletteGroup }}>Searching…</div>}
          {results.length > 0 && <div style={s.paletteGroup}>Results</div>}
          {rows.map((row, i) => (
            <div key={`${row.kind}-${row.source}-${row.id ?? i}`}
              onMouseEnter={() => setSel(i)} onClick={() => runRow(row)}
              style={{ ...s.paletteRow, ...(i === sel ? s.paletteRowOn : {}) }}>
              {row.kind === 'action' && rows[i - 1]?.kind !== 'action' && null}
              <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {row.title}
                {row.snippet && <span style={{ color: FG4, marginLeft: '0.5rem', fontSize: '12px' }}>{row.snippet.slice(0, 60)}</span>}
              </span>
              {row.kind === 'result' && typeof row.score === 'number' && <span style={s.paletteScore}>{row.score.toFixed(2)}</span>}
              <span style={{ ...s.paletteSrc, color: SRC_COLORS[row.source] || FG3 }}>
                {row.source}{row.media_type === 'image' ? ' · img' : ''}
              </span>
            </div>
          ))}
          {rows.length === 0 && <div style={s.paletteGroup}>No matches</div>}
        </div>
        <div style={s.paletteHint}>
          <span>↑↓ navigate</span><span>↵ open</span><span>esc close</span>
        </div>
      </div>
    </div>
  )
}
