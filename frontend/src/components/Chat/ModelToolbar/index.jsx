import { useState } from 'react'
import s, { NOMINAL } from '../../../lib/chatStyles.js'
import ParamSlider from '../../ParamSlider.jsx'
import ComparePicker from './ComparePicker.jsx'
import { ROLE_SUBLABELS } from '../../../lib/chatConstants.js'

const ROLE_KEYS = ['llama', 'coder', 'reasoning']
// Cap on the model-label portion of a role pill when it needs a disambiguating role suffix
// (two roles sharing one id) — keeps a single pill from ballooning if a future catalog label is
// long, while the " · fast"/" · code"/" · reasoning" suffix that actually disambiguates always
// stays fully visible.
const PILL_LABEL_MAX = '150px'

export default function ModelToolbar({
  selectedModel, setSelectedModel,
  compareMode, setCompareMode,
  compareModels, setCompareModels,
  paramsOpen, setParamsOpen,
  tempEnabled, setTempEnabled, temperature, setTemperature,
  tokensEnabled, setTokensEnabled, maxTokens, setMaxTokens,
  topPEnabled, setTopPEnabled, topP, setTopP,
  attachedFiles, detachFile,
  input, setInput, loading,
  send,
  voice,
  catalog,
}) {
  const [comparePickerOpen, setComparePickerOpen] = useState(false)
  const catalogSelected = selectedModel !== 'auto' && !ROLE_KEYS.includes(selectedModel)

  return (
    <div>
      <div style={s.toolbarWrap}>
        <div style={s.toolbar}>
          <div style={s.modelPills}>
            <button onClick={() => setSelectedModel('auto')}
              style={{ ...s.pill, ...(selectedModel === 'auto' ? s.pillActive : {}) }}>
              Auto
            </button>
            {ROLE_KEYS.map(k => {
              const label = catalog?.roleModels?.[k]?.label || k
              // Two roles can share one live id (llama+coder both on nano-omni today) — when
              // that happens their pills would otherwise show identical text with no way to
              // tell them apart except highlight position. Append the role's short tag to
              // disambiguate; leave the label alone when every role has a distinct id.
              const suffix = catalog?.roleCollides?.[k] ? ROLE_SUBLABELS[k] : null
              return (
                <button key={k} onClick={() => setSelectedModel(k)}
                  style={{ ...s.pill, ...(selectedModel === k ? s.pillActive : {}) }}>
                  <span style={suffix
                    ? { display:'inline-block', maxWidth:PILL_LABEL_MAX, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', verticalAlign:'bottom' }
                    : undefined}>{label}</span>
                  {suffix && <span> · {suffix}</span>}
                </button>
              )
            })}
            {catalogSelected && (
              <span style={{ ...s.pill, ...s.pillActive, display:'inline-flex', alignItems:'center', gap:'0.35rem' }}>
                {catalog?.labelFor(selectedModel) || selectedModel}
                <span style={s.chipX} onClick={() => setSelectedModel('auto')} title="Clear model pick">✕</span>
              </span>
            )}
          </div>
          <div style={{ ...s.toolRight, position:'relative' }}>
            <button onClick={() => { if (compareMode) { setCompareMode(false); setComparePickerOpen(false) } else { setComparePickerOpen(o => !o) } }}
              style={{ ...s.pill, ...(compareMode ? s.pillCompare : {}) }}
              title={compareMode ? 'Turn off compare mode' : 'Pick models to compare side by side'}>
              ⊞ Compare{compareMode ? ` (${(compareModels || []).length || 3})` : ''}
            </button>
            {comparePickerOpen && (
              <ComparePicker
                catalog={catalog}
                compareModels={compareModels}
                setCompareModels={setCompareModels}
                setCompareMode={setCompareMode}
                onClose={() => setComparePickerOpen(false)}
              />
            )}
            <button onClick={() => setParamsOpen(!paramsOpen)}
              style={{ ...s.pill, ...(paramsOpen ? s.pillActive : {}) }}
              title="Temperature / max tokens / top-p">
              ⚙
            </button>
          </div>
        </div>

        {paramsOpen && (
          <div style={s.paramsBar}>
            <ParamSlider label="Temp" enabled={tempEnabled} onToggle={setTempEnabled}
              value={temperature} onChange={setTemperature} min={0} max={2} step={0.05}
              fmt={v => v.toFixed(2)} />
            <ParamSlider label="Tokens" enabled={tokensEnabled} onToggle={setTokensEnabled}
              value={maxTokens} onChange={setMaxTokens} min={256} max={4096} step={256}
              fmt={v => v} />
            <ParamSlider label="Top-p" enabled={topPEnabled} onToggle={setTopPEnabled}
              value={topP} onChange={setTopP} min={0} max={1} step={0.05}
              fmt={v => v.toFixed(2)} />
          </div>
        )}
      </div>

      {attachedFiles.length > 0 && (
        <div style={s.fileChipsRow}>
          {attachedFiles.map(f => (
            <span key={f.id} style={s.fileChip}>
              📄 {f.filename.length > 24 ? f.filename.slice(0,22)+'…' : f.filename}
              <span style={s.chipX} onClick={() => detachFile(f.id)} title="Detach">✕</span>
            </span>
          ))}
        </div>
      )}

      <form onSubmit={send} style={s.bar}>
        {voice && (voice.voiceAvailable === true || voice.voiceAvailable === null) && (
          <button type="button"
            onClick={voice.recording ? voice.stopRecording : voice.startRecording}
            disabled={voice.transcribing}
            style={{ ...s.micBtn, ...(voice.recording ? s.micRec : {}) }}>
            {voice.transcribing ? 'BUSY' : voice.recording ? 'REC' : 'MIC'}
          </button>
        )}
        <input value={input} onChange={e => setInput(e.target.value)} placeholder={compareMode ? 'Compare prompt across all models…' : 'Ask anything…'} disabled={loading} style={s.input} />
        <button type="submit" disabled={loading || !input.trim()} style={{ ...s.send, ...(compareMode ? { background:'rgba(85,214,124,0.10)', color:NOMINAL, border:`1px solid ${NOMINAL}` } : {}) }}>
          {loading ? '…' : compareMode ? '⊞' : 'Transmit'}
        </button>
      </form>
    </div>
  )
}
