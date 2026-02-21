import { KernelSummary } from '../../types'
import { kernelsApi } from '../../api/kernels'
import { ImportanceBar } from './ImportanceBar'
import { ClassBadge } from '../shared/ClassBadge'

interface Props {
  kernel: KernelSummary
  selected: boolean
  maxImportance: number
  onClick: () => void
  onDelete: () => void
  onReclassify: (cls: string) => void
}

export function KernelCard({ kernel, selected, maxImportance, onClick, onDelete, onReclassify }: Props) {
  return (
    <div
      onClick={onClick}
      className={`cursor-pointer rounded-lg border transition-all bg-gray-900 ${
        kernel.is_deleted ? 'opacity-40 border-gray-800' :
        selected ? 'border-brand-500 ring-1 ring-brand-500/30' : 'border-gray-800 hover:border-gray-600'
      }`}
    >
      <div className="aspect-square bg-gray-800 rounded-t-lg overflow-hidden">
        <img
          src={kernelsApi.imageUrl(kernel.id)}
          alt={`Kernel ${kernel.id}`}
          className="w-full h-full object-cover"
          loading="lazy"
        />
      </div>
      <div className="p-2 space-y-1.5">
        <ImportanceBar score={kernel.importance_score} max={maxImportance} />
        <div className="flex items-center justify-between gap-1">
          {kernel.assigned_class && <ClassBadge cls={kernel.assigned_class} size="sm" />}
          {kernel.is_deleted && <span className="text-xs text-gray-600 italic">deleted</span>}
        </div>
        <div className="text-xs text-gray-600 truncate">{kernel.layer_name} #{kernel.filter_index}</div>
        {/* Action buttons */}
        <div className="flex gap-1" onClick={e => e.stopPropagation()}>
          <button
            onClick={() => onReclassify(kernel.assigned_class === 'malignant' ? 'benign' : 'malignant')}
            className="flex-1 text-xs py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-400 transition-colors"
          >
            Reclassify
          </button>
          <button
            onClick={onDelete}
            disabled={kernel.is_deleted}
            className="flex-1 text-xs py-0.5 rounded bg-gray-800 hover:bg-red-900 text-gray-400 hover:text-red-300 disabled:opacity-40 transition-colors"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}
