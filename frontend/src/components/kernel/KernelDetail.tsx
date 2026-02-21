import { useEffect } from 'react'
import { useKernelStore } from '../../store/kernelStore'
import { kernelsApi } from '../../api/kernels'
import { ImportanceBar } from './ImportanceBar'
import { ClassBadge } from '../shared/ClassBadge'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { imagesApi } from '../../api/images'

export function KernelDetail() {
  const { selectedKernel, activations, activationsLoading, fetchActivations, selectKernel } = useKernelStore()

  useEffect(() => {
    if (selectedKernel) {
      fetchActivations(selectedKernel.id)
    }
  }, [selectedKernel?.id])

  if (!selectedKernel) return null

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-gray-900 border-l border-gray-800 shadow-2xl z-50 overflow-y-auto">
      <div className="flex items-center justify-between p-4 border-b border-gray-800 sticky top-0 bg-gray-900">
        <div>
          <div className="font-semibold text-sm">{selectedKernel.id}</div>
          <div className="text-xs text-gray-500">{selectedKernel.layer_name} · filter #{selectedKernel.filter_index}</div>
        </div>
        <button
          onClick={() => selectKernel(null)}
          className="text-gray-500 hover:text-gray-200 text-xl leading-none"
        >
          ✕
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* Large filter viz */}
        <div>
          <div className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">Filter Pattern</div>
          <img
            src={kernelsApi.imageUrl(selectedKernel.id)}
            alt="filter"
            className="w-full rounded-lg bg-gray-800"
          />
        </div>

        {/* Stats */}
        <div className="space-y-2">
          <ImportanceBar score={selectedKernel.importance_score} />
          {selectedKernel.assigned_class && <ClassBadge cls={selectedKernel.assigned_class} />}
        </div>

        {/* Top activating images */}
        <div>
          <div className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">
            Top Activating Images
          </div>
          {activationsLoading ? (
            <div className="flex justify-center py-6"><LoadingSpinner /></div>
          ) : activations && activations.top_images.length > 0 ? (
            <div className="grid grid-cols-3 gap-2">
              {activations.top_images.map((ti, i) => (
                <div key={i} className="relative aspect-square rounded overflow-hidden bg-gray-800">
                  <img
                    src={imagesApi.fileUrl(ti.image_id)}
                    alt="activating"
                    className="w-full h-full object-cover"
                  />
                  <img
                    src={`data:image/png;base64,${ti.activation_map_b64}`}
                    alt="activation"
                    className="absolute inset-0 w-full h-full object-cover mix-blend-multiply opacity-70"
                  />
                  <div className="absolute bottom-0 right-0 bg-black/70 text-white text-[9px] px-1 rounded-tl">
                    {ti.max_activation.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-gray-500 italic">No activation data available.</div>
          )}
        </div>
      </div>
    </div>
  )
}
