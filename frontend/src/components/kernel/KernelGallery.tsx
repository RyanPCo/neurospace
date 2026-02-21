import { useKernelStore } from '../../store/kernelStore'
import { KernelCard } from './KernelCard'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { usePagination } from '../../hooks/usePagination'

export function KernelGallery() {
  const { kernels, total, page, pageSize, loading, selectedKernel, setPage, selectKernel, deleteKernel, reclassifyKernel } = useKernelStore()
  const { totalPages, hasNext, hasPrev } = usePagination(total, page, pageSize)

  const maxImportance = Math.max(...kernels.map(k => k.importance_score), 1)

  if (loading && kernels.length === 0) {
    return <div className="flex justify-center py-16"><LoadingSpinner /></div>
  }

  if (kernels.length === 0) {
    return (
      <div className="text-center text-gray-500 py-16 text-sm">
        No kernels found. Run <code className="bg-gray-800 px-1 rounded">make kernels</code> to pre-compute.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 xl:grid-cols-6 2xl:grid-cols-8 gap-3">
        {kernels.map(k => (
          <KernelCard
            key={k.id}
            kernel={k}
            selected={selectedKernel?.id === k.id}
            maxImportance={maxImportance}
            onClick={() => selectKernel(selectedKernel?.id === k.id ? null : k)}
            onDelete={() => deleteKernel(k.id)}
            onReclassify={cls => reclassifyKernel(k.id, cls)}
          />
        ))}
      </div>
      {/* Pagination */}
      <div className="flex items-center justify-between text-sm text-gray-400 border-t border-gray-800 pt-3">
        <span>{total} kernels · page {page}/{totalPages || 1}</span>
        <div className="flex gap-2">
          <button
            disabled={!hasPrev}
            onClick={() => setPage(page - 1)}
            className="px-3 py-1 rounded bg-gray-800 disabled:opacity-40 hover:bg-gray-700"
          >
            Prev
          </button>
          <button
            disabled={!hasNext}
            onClick={() => setPage(page + 1)}
            className="px-3 py-1 rounded bg-gray-800 disabled:opacity-40 hover:bg-gray-700"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  )
}
