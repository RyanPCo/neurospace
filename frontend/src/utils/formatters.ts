export function formatConfidence(v: number | null): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(1)}%`
}

export function formatLoss(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toFixed(4)
}

export function formatAcc(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(1)}%`
}

export function formatDatetime(s: string | null): string {
  if (!s) return '—'
  return new Date(s).toLocaleString()
}

export function classColor(cls: string | null): string {
  if (cls === 'malignant') return '#ef4444'
  if (cls === 'benign') return '#22c55e'
  return '#94a3b8'
}
