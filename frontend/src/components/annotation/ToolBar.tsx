import { useAnnotationStore, Tool, LabelClass } from '../../store/annotationStore'

const TOOLS: { value: Tool; label: string; icon: string }[] = [
  { value: 'polygon', label: 'Polygon', icon: '⬡' },
  { value: 'brush', label: 'Brush', icon: '✏️' },
  { value: 'eraser', label: 'Eraser', icon: '⊘' },
]

const CLASSES: { value: LabelClass; label: string; color: string }[] = [
  { value: 'malignant', label: 'Malignant', color: 'bg-red-600' },
  { value: 'benign', label: 'Benign', color: 'bg-green-600' },
]

export function ToolBar() {
  const { activeTool, activeClass, brushRadius, setTool, setClass, setBrushRadius } = useAnnotationStore()

  return (
    <div className="flex flex-wrap items-center gap-3 py-2">
      {/* Tool selector */}
      <div className="flex gap-1">
        {TOOLS.map(t => (
          <button
            key={t.value}
            onClick={() => setTool(t.value)}
            title={t.label}
            className={`px-2.5 py-1.5 rounded text-sm transition-colors ${
              activeTool === t.value ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* Class selector */}
      <div className="flex gap-1">
        {CLASSES.map(c => (
          <button
            key={c.value}
            onClick={() => setClass(c.value)}
            className={`px-2.5 py-1.5 rounded text-sm transition-colors ${
              activeClass === c.value
                ? `${c.color} text-white`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>

      {/* Brush radius (only for brush tool) */}
      {activeTool === 'brush' && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Radius:</span>
          <input
            type="range"
            min={0.01}
            max={0.1}
            step={0.005}
            value={brushRadius}
            onChange={e => setBrushRadius(Number(e.target.value))}
            className="accent-brand-500 w-24"
          />
          <span className="text-xs text-gray-400 font-mono">{(brushRadius * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}
