interface Props {
  score: number
  max?: number
}

export function ImportanceBar({ score, max = 1 }: Props) {
  const pct = max > 0 ? Math.min((score / max) * 100, 100) : 0
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-xs text-gray-500">
        <span>Importance</span>
        <span className="font-mono">{score.toFixed(3)}</span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-amber-500 rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
