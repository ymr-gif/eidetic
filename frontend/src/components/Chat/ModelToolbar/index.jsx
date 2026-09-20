import { useState } from 'react'
import s, { NOMINAL } from '../../../lib/chatStyles.js'
import ParamSlider from '../../ParamSlider.jsx'
import ComparePicker from './ComparePicker.jsx'
import { MODEL_KEYS } from '../../../lib/chatConstants.js'

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
  const catalogSelected = selectedModel !== 'auto' && !['llama', 'coder', 'reasoning'].includes(selectedModel)

  return (
    <div>
      <div style={s.toolbarWrap}>
        <div style={s.toolbar}>
          <div style={s.modelPills}>
            {[['auto', 'Auto'], ['llama', catalog?.labelFor(MODEL_KEYS.llama)], ['coder', catalog?.labelFor(MODEL_KEYS.coder)], ['reasoning', catalog?.labelFor(MODEL_KEYS.reasoning)]].map(([key, label]) => (
              <button key={key} onClick={() => setSelectedModel(key)}
                style={{ ...s.pill, ...(selectedModel === key ? s.pillActive : {}) }}>
                {label}
              </button>
            ))}
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
