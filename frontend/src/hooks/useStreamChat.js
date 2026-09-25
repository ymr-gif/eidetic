import { COMPARE_MODELS } from '../lib/chatConstants.js'

export default function useStreamChat({ token, conv, modelParams, mem, insights, catalog, onLogout, onCalendarWrite, onTtft, onLinkState, onModelNotice }) {
  const authHeaders = { 'Authorization': `Bearer ${token}` }

  function buildBody(text) {
    const body = { message: text, conversation_id: conv.activeConvId }
    if (modelParams.selectedModel !== 'auto') body.model_override = modelParams.selectedModel
    if (modelParams.compareMode) {
      body.compare = true
      if (modelParams.compareModels && modelParams.compareModels.length) body.compare_models = modelParams.compareModels
    }
    if (modelParams.tempEnabled)   body.temperature = modelParams.temperature
    if (modelParams.tokensEnabled) body.max_tokens  = modelParams.maxTokens
    if (modelParams.topPEnabled)   body.top_p       = modelParams.topP
    return body
  }

  async function send(e) {
    e.preventDefault()
    const text = conv.input.trim(); if (!text || conv.loading) return
    const isCompare = modelParams.compareMode
    const userId = conv.nextId.current++, aiId = conv.nextId.current++
    conv.setInput(''); conv.setLoading(true); conv.setProactive(null); conv.setPendingWriteFact(null); if (onCalendarWrite) onCalendarWrite(null); conv.setLastSession('')

    if (isCompare) {
      // Best-effort initial order so the layout isn't empty before `compare_start` lands (it's
      // the very first SSE event, so this is usually overwritten within one network round trip).
      // Prefer the live catalog's de-duplicated default (falls back to the static list before
      // the catalog fetch resolves).
      const liveDefault = (catalog?.defaultCompareIds && catalog.defaultCompareIds.length) ? catalog.defaultCompareIds : COMPARE_MODELS
      const initialOrder = (modelParams.compareModels && modelParams.compareModels.length) ? modelParams.compareModels : liveDefault
      conv.setMessages(prev => [...prev,
        { id: userId, role: 'user', text, streaming: false },
        { id: aiId, role: 'compare', compareOrder: initialOrder, compareLabels: {},
          responses: Object.fromEntries(initialOrder.map(m => [m, { text: '', streaming: true }])) },
      ])
    } else {
      conv.setMessages(prev => [...prev,
        { id: userId, role: 'user', text, streaming: false },
        { id: aiId, role: 'ai', text: '', model: null, streaming: true },
      ])
    }

    const t0 = performance.now()
    let ttftReported = false
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST', headers: { ...authHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify(buildBody(text)),
      })
      if (res.status === 401) { onLogout(); return }
      if (!res.ok) {
        // 422 here is a pre-stream rejection (e.g. compare_models: too_many_compare_models /
        // model_unavailable on the first bad id) — surface the backend's detail instead of a flat message.
        const data = await res.json().catch(() => ({}))
        const detail = typeof data.detail === 'string' ? data.detail : data.detail?.error
        const text = detail === 'model_unavailable' ? 'That model is no longer available.'
          : detail === 'too_many_compare_models' ? 'Too many models selected to compare (max 4).'
          : detail || `Request failed (${res.status})`
        conv.setMessages(prev => prev.map(m => m.id === aiId ? { ...m, role: 'err', text, streaming: false } : m))
        if (onLinkState) onLinkState(true)
        return
      }

      const reader = res.body.getReader(), decoder = new TextDecoder(); let buffer = ''
      while (true) {
        const { done, value } = await reader.read(); if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n'); buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim(); if (!raw) continue
          try {
            const event = JSON.parse(raw)
            if (event.type === 'compare_start') {
              const order = (event.models || []).map(mm => mm.id)
              const labels = Object.fromEntries((event.models || []).map(mm => [mm.id, mm.label]))
              conv.setMessages(prev => prev.map(m => m.id === aiId
                ? { ...m, compareOrder: order, compareLabels: labels,
                    responses: Object.fromEntries(order.map(id => [id, { text: '', streaming: true }])) }
                : m))
            }
            if (event.type === 'token' && !ttftReported) {
              ttftReported = true
              if (onTtft) onTtft(performance.now() - t0)
              if (onLinkState) onLinkState(false)
            }
            if (event.type === 'token') {
              if (isCompare) {
                const model = event.model
                conv.setMessages(prev => prev.map(m => m.id === aiId
                  ? { ...m, responses: { ...m.responses, [model]: { ...m.responses[model], text: (m.responses[model]?.text||'') + event.content } } }
                  : m))
              } else {
                conv.setMessages(prev => prev.map(m => m.id === aiId ? { ...m, text: m.text + event.content } : m))
              }
            } else if (event.type === 'preamble_discard') {
              // streamed tokens were pre-tool preamble — clear them; real answer follows
              if (isCompare) {
                const model = event.model
                conv.setMessages(prev => prev.map(m => m.id === aiId
                  ? { ...m, responses: { ...m.responses, [model]: { ...m.responses[model], text: '' } } }
                  : m))
              } else {
                conv.setMessages(prev => prev.map(m => m.id === aiId ? { ...m, text: '' } : m))
              }
            } else if (event.type === 'done') {
              if (isCompare) {
                conv.setMessages(prev => prev.map(m => m.id === aiId
                  ? { ...m, responses: Object.fromEntries(Object.entries(m.responses).map(([k,v]) => [k, { ...v, streaming: false }])) }
                  : m))
              } else {
                conv.setMessages(prev => prev.map(m => m.id === aiId ? { ...m, model: event.model, streaming: false, promptTokens: event.prompt_tokens, completionTokens: event.completion_tokens, totalTokens: event.total_tokens, costUsd: event.cost_usd, provenance: event.provenance || [], queryType: event.query_type || '', srcCount: event.src_count ?? 0, webSearched: event.web_searched ?? false, urlFetched: event.url_fetched ?? false, grounding: event.grounding || null, intent: event.intent || '', activity: event.activity || [] } : m))
                mem.setMemTick(t => t + 1)
                setTimeout(() => { if (mem.memTab === 'graph') insights.loadGraphStats() }, 2000)
                // The locked-model-unavailable notice rides the activity trace (no dedicated SSE
                // event) and stays hidden behind the grounding gauge's click-to-expand timeline —
                // surface it directly too, since it can fire on turns with no retrieval at all.
                const modelNotice = (event.activity || []).find(a => a.stage === 'model' && a.level === 'error')
                if (modelNotice && onModelNotice) onModelNotice(modelNotice.detail)
              }
              if (event.last_session) conv.setLastSession(event.last_session)
              const cid = event.conversation_id
              if (cid) {
                conv.setActiveConvId(cid)
                conv.setConversations(prev => {
                  const exists = prev.find(c => c.id === cid)
                  if (exists) return [{ ...exists, updated_at: new Date().toISOString() }, ...prev.filter(c => c.id !== cid)]
                  return [{ id: cid, title: text.slice(0, 60), updated_at: new Date().toISOString(), memory_enabled: true, system_prompt: '', locked_model: '' }, ...prev]
                })
              }
            } else if (event.type === 'tool_call') {
              conv.setMessages(prev => prev.map(m => m.id === aiId
                ? { ...m, toolCalls: [...(m.toolCalls || []), { name: event.name, args: event.args, result: null }] }
                : m))
            } else if (event.type === 'tool_result') {
              conv.setMessages(prev => prev.map(m => m.id === aiId
                ? { ...m, toolCalls: (m.toolCalls || []).map((tc, i) =>
                    i === (m.toolCalls.length - 1) ? { ...tc, result: event.content } : tc
                  )}
                : m))
            } else if (event.type === 'ask_user') {
              conv.setMessages(prev => prev.map(m => m.id === aiId ? { ...m, askUser: event.question } : m))
            } else if (event.type === 'confirm_write_memory') {
              conv.setPendingWriteFact(event.fact)
            } else if (event.type === 'confirm_calendar_write') {
              if (onCalendarWrite) onCalendarWrite({ op: event.op, args: event.args, summary: event.summary })
            } else if (event.type === 'proactive') {
              conv.setProactive(event.content)
            } else if (event.type === 'error') {
              conv.setMessages(prev => prev.map(m => m.id === aiId ? { ...m, role: 'err', text: event.message || 'Error', streaming: false } : m))
            }
          } catch { /* ignore */ }
        }
      }
    } catch (err) {
      conv.setMessages(prev => prev.map(m => m.id === aiId ? { ...m, role: 'err', text: `Network error: ${err.message}`, streaming: false } : m))
      if (onLinkState) onLinkState(true)
    } finally { conv.setLoading(false) }
  }

  return { send, buildBody }
}
