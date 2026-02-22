import { useKernelStore } from '../../store/kernelStore'
import { KernelCard } from './KernelCard'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { usePagination } from '../../hooks/usePagination'

export function KernelGallery() {
  const { kernels, total, page, pageSize, loading, selectedKernel, actioningId, setPage, selectKernel, deleteKernel, reclassifyKernel } = useKernelStore()
  const { totalPages, hasNext, hasPrev } = usePagination(total, page, pageSize)
  const maxImportance = Math.max(...kernels.map(k => k.importance_score), 1)

  if (loading && kernels.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4 text-gray-500">
        <LoadingSpinner size="lg" />
        <span className="text-sm">Loading kernels…</span>
      </div>
    )
  }

  if (kernels.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-3 text-gray-500">
        <div className="text-4xl">🧩</div>
        <div className="text-sm font-medium text-gray-400">No kernels available</div>
        <div className="text-xs text-gray-600">Run <code className="bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">make kernels</code> to pre-compute filter visualizations</div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-3">
        {kernels.map(k => (
          <KernelCard
            key={k.id}
            kernel={k}
            selected={selectedKernel?.id === k.id}
            maxImportance={maxImportance}
            actioning={actioningId === k.id}
            onClick={() => selectKernel(selectedKernel?.id === k.id ? null : k)}
            onDelete={() => deleteKernel(k.id)}
            onReclassify={cls => reclassifyKernel(k.id, cls)}
          />
        ))}
      </div>

      <div className="flex items-center justify-between text-sm text-gray-500 border-t border-gray-800 pt-3">
        <span>{total.toLocaleString()} kernels · page {page} of {totalPages || 1}</span>
        <div className="flex gap-2">
          <button disabled={!hasPrev} onClick={() => setPage(page - 1)}
            className="px-3 py-1.5 rounded-lg bg-gray-800 disabled:opacity-40 hover:bg-gray-700 transition-colors text-gray-300">
            ← Prev
          </button>
          <button disabled={!hasNext} onClick={() => setPage(page + 1)}
            className="px-3 py-1.5 rounded-lg bg-gray-800 disabled:opacity-40 hover:bg-gray-700 transition-colors text-gray-300">
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
