import { useAnnotationStore } from '../../store/annotationStore'
import { ClassBadge } from '../shared/ClassBadge'

export function AnnotationList() {
  const { annotations, deleteAnnotation } = useAnnotationStore()

  if (annotations.length === 0) {
    return <div className="text-xs text-gray-500 italic">No annotations yet.</div>
  }

  return (
    <div className="space-y-1.5">
      {annotations.map(ann => (
        <div
          key={ann.id}
          className="flex items-center gap-2 bg-gray-800 rounded-lg px-3 py-2"
        >
          <ClassBadge cls={ann.label_class} size="sm" />
          <span className="text-xs text-gray-400 flex-1">{ann.geometry_type}</span>
          <button
            onClick={() => deleteAnnotation(ann.id)}
            className="text-gray-600 hover:text-red-400 transition-colors text-xs"
            title="Delete"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}
