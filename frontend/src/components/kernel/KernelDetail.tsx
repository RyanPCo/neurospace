import { useEffect, useState } from 'react'
import { useKernelStore } from '../../store/kernelStore'
import { kernelsApi } from '../../api/kernels'
import { ImportanceBar } from './ImportanceBar'
import { ClassBadge } from '../shared/ClassBadge'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { imagesApi } from '../../api/images'
import { formatDatetime } from '../../utils/formatters'

export function KernelDetail() {
  const { selectedKernel, activations, activationsLoading, fetchActivations, selectKernel, updateNotes } = useKernelStore()
  const [notes, setNotes] = useState('')
  const [notesDirty, setNotesDirty] = useState(false)

  useEffect(() => {
    if (selectedKernel) {
      fetchActivations(selectedKernel.id)
      setNotes(selectedKernel.doctor_notes ?? '')
      setNotesDirty(false)
    }
  }, [selectedKernel?.id])

  if (!selectedKernel) return null

  const handleSaveNotes = () => {
    updateNotes(selectedKernel.id, notes)
    setNotesDirty(false)
  }

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-gray-900 border-l border-gray-800 shadow-2xl z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800 flex-shrink-0">
        <div>
          <div className="font-semibold text-sm text-gray-100 font-mono">{selectedKernel.id}</div>
          <div className="text-xs text-gray-500 mt-0.5">
            {selectedKernel.layer_name} · filter #{selectedKernel.filter_index}
          </div>
        </div>
        <button onClick={() => selectKernel(null)}
          className="text-gray-500 hover:text-gray-200 transition-colors text-xl w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-800">
          ✕
        </button>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Filter visualization */}
        <div>
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Filter Pattern</div>
          <img src={kernelsApi.imageUrl(selectedKernel.id)} alt="filter"
            className="w-full rounded-xl bg-gray-800 border border-gray-700" />
        </div>

        {/* Stats */}
        <div className="space-y-3">
          <ImportanceBar score={selectedKernel.importance_score} />
          <div className="flex items-center gap-2 flex-wrap">
            {selectedKernel.assigned_class
              ? <ClassBadge cls={selectedKernel.assigned_class} />
              : <span className="text-xs text-gray-600 italic">No class assigned</span>}
            {selectedKernel.is_deleted && (
              <span className="text-xs bg-red-900/40 text-red-400 border border-red-800 px-2 py-0.5 rounded-full">deleted</span>
            )}
          </div>
          {selectedKernel.last_scored_at && (
            <div className="text-xs text-gray-600">
              Last scored: {formatDatetime(selectedKernel.last_scored_at)}
            </div>
          )}
        </div>

        {/* Notes */}
        <div>
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Doctor Notes</div>
          <textarea
            value={notes}
            onChange={e => { setNotes(e.target.value); setNotesDirty(true) }}
            placeholder="Add clinical observations about this kernel…"
            rows={3}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-600 resize-none focus:outline-none focus:border-brand-500 transition-colors"
          />
          {notesDirty && (
            <button onClick={handleSaveNotes}
              className="mt-2 px-3 py-1.5 text-xs bg-brand-600 text-white rounded-lg hover:bg-brand-700 transition-colors">
              Save Notes
            </button>
          )}
        </div>

        {/* Top activating images */}
        <div>
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Top Activating Images
          </div>
          {activationsLoading ? (
            <div className="flex justify-center py-8"><LoadingSpinner /></div>
          ) : activations && activations.top_images.length > 0 ? (
            <div className="grid grid-cols-3 gap-1.5">
              {activations.top_images.map((ti, i) => (
                <div key={i} className="relative aspect-square rounded-lg overflow-hidden bg-gray-800 group">
                  <img src={imagesApi.fileUrl(ti.image_id)} alt="activating"
                    className="w-full h-full object-cover" />
                  <img src={`data:image/png;base64,${ti.activation_map_b64}`} alt="activation"
                    className="absolute inset-0 w-full h-full object-cover mix-blend-multiply opacity-75 group-hover:opacity-90 transition-opacity" />
                  <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-1">
                    <span className="text-white text-[9px] font-mono">{ti.max_activation.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-gray-600 italic py-4 text-center">
              No activation data yet. Run <code className="bg-gray-800 px-1 rounded">make kernels</code> to compute.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
