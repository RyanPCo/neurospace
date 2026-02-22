import { useState } from 'react'
import { KernelSummary } from '../../types'
import { kernelsApi } from '../../api/kernels'
import { ImportanceBar } from './ImportanceBar'
import { ClassBadge } from '../shared/ClassBadge'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { LoadingSpinner } from '../shared/LoadingSpinner'

interface Props {
  kernel: KernelSummary
  selected: boolean
  maxImportance: number
  actioning: boolean
  onClick: () => void
  onDelete: () => void
  onReclassify: (cls: string) => void
}

export function KernelCard({ kernel, selected, maxImportance, actioning, onClick, onDelete, onReclassify }: Props) {
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [confirmReclassify, setConfirmReclassify] = useState(false)
  const [targetClass, setTargetClass] = useState(kernel.assigned_class ?? 'glioma')
  const classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

  return (
    <>
      <div
        onClick={onClick}
        className={`cursor-pointer rounded-xl border transition-all bg-gray-900 group ${
          kernel.is_deleted
            ? 'opacity-40 border-gray-800 pointer-events-none'
            : selected
            ? 'border-brand-500 ring-2 ring-brand-500/20 shadow-lg shadow-brand-500/10'
            : 'border-gray-800 hover:border-gray-600 hover:shadow-md'
        }`}
      >
        {/* Filter image */}
        <div className="aspect-square bg-gray-800 rounded-t-xl overflow-hidden relative">
          <img
            src={kernelsApi.imageUrl(kernel.id)}
            alt={`Kernel ${kernel.id}`}
            className="w-full h-full object-cover transition-transform group-hover:scale-105 duration-200"
            loading="lazy"
          />
          {kernel.is_deleted && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60">
              <span className="text-xs text-gray-400 font-medium">Deleted</span>
            </div>
          )}
          {actioning && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <LoadingSpinner size="sm" />
            </div>
          )}
        </div>

        {/* Info */}
        <div className="p-2.5 space-y-2">
          <ImportanceBar score={kernel.importance_score} max={maxImportance} />
          {kernel.assigned_class && <ClassBadge cls={kernel.assigned_class} size="sm" />}
          <div className="text-[10px] text-gray-600 truncate font-mono">
            {kernel.layer_name} <span className="text-gray-500">#{kernel.filter_index}</span>
          </div>

          {/* Actions */}
          <div className="flex gap-1 pt-0.5" onClick={e => e.stopPropagation()}>
            <button
              onClick={() => {
                setTargetClass(kernel.assigned_class ?? 'glioma')
                setConfirmReclassify(true)
              }}
              disabled={actioning}
              className="flex-1 text-xs py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 disabled:opacity-40 transition-colors"
            >
              Reclassify
            </button>
            <button
              onClick={() => setConfirmDelete(true)}
              disabled={actioning || kernel.is_deleted}
              className="flex-1 text-xs py-1 rounded-lg bg-gray-800 hover:bg-red-900/60 text-gray-400 hover:text-red-300 disabled:opacity-40 transition-colors"
            >
              Delete
            </button>
          </div>
        </div>
      </div>

      <ConfirmDialog
        open={confirmDelete}
        title="Delete kernel?"
        description={`Mark kernel ${kernel.id} as deleted. It will be zeroed out on the next retrain. This cannot be undone.`}
        confirmLabel="Delete"
        danger
        onConfirm={() => { setConfirmDelete(false); onDelete() }}
        onCancel={() => setConfirmDelete(false)}
      />

      <ConfirmDialog
        open={confirmReclassify}
        title="Reclassify kernel?"
        description={`Change assigned class from "${kernel.assigned_class ?? 'unset'}".`}
        confirmLabel="Reclassify"
        onConfirm={() => { setConfirmReclassify(false); onReclassify(targetClass) }}
        onCancel={() => setConfirmReclassify(false)}
      >
        <div className="grid grid-cols-2 gap-2">
          {classes.map(cls => (
            <button
              key={cls}
              onClick={() => setTargetClass(cls)}
              className={`px-2 py-1 text-xs rounded-md border transition-colors ${
                targetClass === cls
                  ? 'border-brand-500 bg-brand-600/20 text-brand-300'
                  : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-500'
              }`}
            >
              {cls}
            </button>
          ))}
        </div>
      </ConfirmDialog>
    </>
  )
}
