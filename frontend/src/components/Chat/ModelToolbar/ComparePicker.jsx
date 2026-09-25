import { useEffect, useMemo, useState } from 'react'
import s, { FG3, FG4, AMBER, LINE, LINE2, INSET, MONO } from '../../../lib/chatStyles.js'
import { COMPARE_MODELS } from '../../../lib/chatConstants.js'

const MAX_COMPARE = 4

// Searchable checklist popover for the compare pill — max 4 models, defaults to the 3 role
// models (compareModels=null in useModelParams). Confirm writes the pick + flips compareMode on;
// Cancel closes without changing anything.
export default function ComparePicker({ catalog, compareModels, setCompareModels, setCompareMode, onClose }) {
  const [filter, setFilter] = useState('')
  const [picked, setPicked] = useState(() => {
    if (compareModels && compareModels.length) return compareModels
    return (catalog?.defaultCompareIds && catalog.defaultCompareIds.length) ? catalog.defaultCompareIds : COMPARE_MODELS
  })

  // Escape closes (window-level, same pattern as CommandPalette's own Escape listener).
  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const options = useMemo(() => {
    // roleModelList is already de-duplicated by id — llama and coder currently share one live
    // id, and this must show ONE row for it, not two, or the same checkbox would appear twice.
    const all = [...(catalog?.roleModelList || []), ...(catalog?.catalogModels || [])]
    const needle = filter.trim().toLowerCase()
    const list = needle
      ? all.filter(m => m.id.toLowerCase().includes(needle) || (m.label || '').toLowerCase().includes(needle))
      : all
    return list.slice(0, 24)
  }, [filter, catalog?.roleModelList, catalog?.catalogModels])

  function toggle(id) {
    setPicked(prev => {
      if (prev.includes(id)) return prev.filter(x => x !== id)
      if (prev.length >= MAX_COMPARE) return prev
      return [...prev, id]
    })
  }

  function confirm() {
    if (picked.length < 2) return
    setCompareModels(picked)
    setCompareMode(true)
    onClose()
  }

  return (
    <>
      {/* Invisible full-screen click-catcher — same outside-close mechanism as CommandPalette's
          paletteOverlay, just transparent since this is a small anchored popover, not a modal. */}
      <div style={{ position:'fixed', inset:0, zIndex:14 }} onClick={onClose} />
      <div onClick={e => e.stopPropagation()} style={{ position:'absolute', bottom:'calc(100% + 6px)', right:0, width:'280px', maxHeight:'320px', display:'flex', flexDirection:'column', background:INSET, border:`1px solid ${LINE2}`, borderRadius:'6px', boxShadow:'0 8px 28px rgba(0,0,0,0.45)', zIndex:15 }}>
        <div style={{ padding:'0.5rem 0.6rem', borderBottom:`1px solid ${LINE}` }}>
          <input value={filter} onChange={e => setFilter(e.target.value)} autoFocus
            placeholder="Search models to compare…" style={s.sideSearchInput} />
        </div>
        <div style={{ flex:1, overflowY:'auto', padding:'0.35rem' }}>
          {options.map(m => {
            const checked = picked.includes(m.id)
            const disabled = !checked && picked.length >= MAX_COMPARE
            return (
              <div key={m.id} onClick={() => !disabled && toggle(m.id)}
                style={{ display:'flex', alignItems:'center', gap:'0.5rem', padding:'0.35rem 0.5rem', borderRadius:'3px', cursor: disabled ? 'default' : 'pointer', opacity: disabled ? 0.4 : 1, fontSize:'12.5px', color: checked ? AMBER : FG3 }}>
                <span style={{ width:'12px', height:'12px', flexShrink:0, borderRadius:'2px', border:`1px solid ${checked ? AMBER : LINE2}`, background: checked ? AMBER : 'transparent' }} />
                <span style={{ flex:1, minWidth:0, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{m.label || m.id}</span>
                {/* roleModelList rows are de-duplicated by id, so a shared id (llama+coder) is
                    ONE row — tag it with every role that owns the id, not just the first. */}
                {m.roles?.length > 0 && <span style={{ fontFamily:MONO, fontSize:'8px', color:FG4, textTransform:'uppercase' }}>{m.roles.join('/')}</span>}
              </div>
            )
          })}
          {options.length === 0 && <div style={{ padding:'0.5rem', fontSize:'12px', color:FG4 }}>No matches</div>}
        </div>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'0.45rem 0.6rem', borderTop:`1px solid ${LINE}` }}>
          <span style={{ fontFamily:MONO, fontSize:'9px', color:FG4 }}>{picked.length}/{MAX_COMPARE}</span>
          <div style={{ display:'flex', gap:'0.4rem' }}>
            <button onClick={onClose} style={s.cancelBtn}>Cancel</button>
            <button onClick={confirm} disabled={picked.length < 2} style={{ ...s.saveBtn, opacity: picked.length < 2 ? 0.5 : 1 }}>Compare</button>
          </div>
        </div>
      </div>
    </>
  )
}
