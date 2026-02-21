interface Props {
  value: number
  onChange: (v: number) => void
  label?: string
}

export function OpacitySlider({ value, onChange, label = 'Opacity' }: Props) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-400 w-16">{label}</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 accent-brand-500"
      />
      <span className="text-xs text-gray-400 w-10 text-right font-mono">
        {Math.round(value * 100)}%
      </span>
    </div>
  )
}
