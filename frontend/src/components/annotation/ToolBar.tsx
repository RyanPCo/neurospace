import { useAnnotationStore, Tool, LabelClass } from '../../store/annotationStore'

const TOOLS: { value: Tool; label: string; icon: string }[] = [
  { value: 'polygon', label: 'Polygon', icon: '⬡' },
  { value: 'brush',   label: 'Brush',   icon: '✏' },
]

const CLASSES: { value: LabelClass; label: string; activeClass: string; description?: string }[] = [
  { value: 'glioma',       label: 'Glioma',       activeClass: 'bg-red-600 text-white border-red-600' },
  { value: 'meningioma',   label: 'Meningioma',   activeClass: 'bg-orange-500 text-white border-orange-500' },
  { value: 'pituitary',    label: 'Pituitary',    activeClass: 'bg-blue-600 text-white border-blue-600' },
  { value: 'notumor',      label: 'No Tumor',     activeClass: 'bg-green-600 text-white border-green-600' },
  {
    value: 'gradcam_focus',
    label: 'Focus Here',
    activeClass: 'bg-amber-500 text-black border-amber-500',
    description: 'Constrained: model must attend here without shifting elsewhere',
  },
]

export function ToolBar() {
  const { activeTool, activeClass, brushRadius, setTool, setClass, setBrushRadius } = useAnnotationStore()

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex gap-1 bg-gray-800 p-1 rounded-lg">
        {TOOLS.map(t => (
          <button key={t.value} onClick={() => setTool(t.value)} aria-label={t.label}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              activeTool === t.value
                ? 'bg-gray-700 text-white shadow'
                : 'text-gray-400 hover:text-gray-200'
            }`}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      <div className="flex gap-1 bg-gray-800 p-1 rounded-lg flex-wrap">
        {CLASSES.map(c => (
          <button key={c.value} onClick={() => setClass(c.value)}
            title={c.description}
            className={`px-3 py-1.5 rounded-md text-sm font-medium border transition-colors ${
              activeClass === c.value
                ? c.activeClass
                : 'text-gray-400 border-transparent hover:text-gray-200'
            }`}>
            {c.value === 'gradcam_focus' ? '⚡ ' : ''}{c.label}
          </button>
        ))}
      </div>

      {activeTool === 'brush' && (
        <div className="flex items-center gap-2 bg-gray-800 px-3 py-1.5 rounded-lg">
          <span className="text-xs text-gray-500">Size</span>
          <input type="range" min={0.01} max={0.1} step={0.005} value={brushRadius}
            onChange={e => setBrushRadius(Number(e.target.value))}
            className="accent-brand-500 w-20" aria-label="Brush radius" />
          <span className="text-xs text-gray-400 font-mono w-8">{(brushRadius * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}
