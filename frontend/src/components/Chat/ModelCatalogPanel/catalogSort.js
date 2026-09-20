// Default-order sort for the catalog list: useful rows first. The catalog holds
// ~67 rows, mostly not_found — plain id-order buries every live/enabled model
// under a wall of dead ones. Priority groups (enabled rows always first, since
// role models must stay enabled regardless of scan status):
//   0 enabled · 1 live · 2 timeout/error (transient) · 3 not_found/gone/delisted
// Within a group: latency ascending (null latency sorts last), then id.
const TRANSIENT = new Set(['timeout', 'error'])

function rowRank(row) {
  if (row.enabled) return 0
  if (row.status === 'live') return 1
  if (TRANSIENT.has(row.status)) return 2
  return 3
}

export function sortCatalogRows(rows) {
  return [...rows].sort((a, b) => {
    const ra = rowRank(a), rb = rowRank(b)
    if (ra !== rb) return ra - rb
    const la = a.latency_ms ?? Infinity, lb = b.latency_ms ?? Infinity
    if (la !== lb) return la - lb
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
  })
}

// "last scan" line: relative time (title carries the absolute ISO timestamp),
// trigger, and the useful summary counts. scan_meta is null until the first
// scan ever runs.
function relTime(iso) {
  const diffS = Math.round((Date.now() - new Date(iso).getTime()) / 1000)
  if (diffS < 5) return 'just now'
  if (diffS < 60) return `${diffS}s ago`
  const diffM = Math.round(diffS / 60)
  if (diffM < 60) return `${diffM}m ago`
  const diffH = Math.round(diffM / 60)
  if (diffH < 24) return `${diffH}h ago`
  return `${Math.round(diffH / 24)}d ago`
}

export function describeScan(scanMeta) {
  if (!scanMeta) return { text: 'no scan recorded yet', title: '' }
  const { ran_at, trigger, scanned, live } = scanMeta
  const parts = [`last scan ${ran_at ? relTime(ran_at) : 'unknown time'}`]
  if (trigger) parts.push(trigger)
  if (scanned != null) parts.push(`${scanned} scanned · ${live || 0} live`)
  return {
    text: parts.join(' · '),
    title: ran_at ? new Date(ran_at).toLocaleString() : '',
  }
}
