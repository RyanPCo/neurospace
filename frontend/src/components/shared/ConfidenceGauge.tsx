import { formatConfidence } from '../../utils/formatters'

interface Props {
  confidence: number | null
  predicted_class: string | null
}

export function ConfidenceGauge({ confidence, predicted_class }: Props) {
  const pct = confidence != null ? confidence * 100 : 0
  const classColors: Record<string, string> = {
    glioma: 'bg-red-500',
    meningioma: 'bg-orange-500',
    pituitary: 'bg-blue-500',
    notumor: 'bg-green-500',
  }
  const color = classColors[predicted_class ?? ''] ?? 'bg-slate-500'

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>Confidence</span>
        <span className="font-mono">{formatConfidence(confidence)}</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
