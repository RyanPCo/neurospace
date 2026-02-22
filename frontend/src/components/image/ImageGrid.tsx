import { useImageStore } from '../../store/imageStore'
import { ImageCard } from './ImageCard'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { usePagination } from '../../hooks/usePagination'

export function ImageGrid() {
  const { images, total, page, pageSize, loading, selectedImageId, filters, setPage, selectImage, clearFilters } = useImageStore()
  const { totalPages, hasNext, hasPrev } = usePagination(total, page, pageSize)
  const hasFilters = Object.values(filters).some(Boolean)

  if (loading && images.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4 text-gray-500">
        <LoadingSpinner size="lg" />
        <span className="text-sm">Loading images…</span>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3 flex-1 overflow-auto content-start pb-1">
        {images.map(img => (
          <ImageCard key={img.id} image={img}
            selected={img.id === selectedImageId}
            onClick={() => selectImage(img.id === selectedImageId ? null : img.id)} />
        ))}

        {images.length === 0 && !loading && (
          <div className="col-span-full flex flex-col items-center justify-center py-20 gap-3 text-gray-500">
            <div className="text-4xl">🔬</div>
            {hasFilters ? (
              <>
                <div className="text-sm font-medium text-gray-400">No images match these filters</div>
                <button onClick={clearFilters}
                  className="text-xs text-brand-400 hover:text-brand-300 underline transition-colors">
                  Clear all filters
                </button>
              </>
            ) : (
              <>
                <div className="text-sm font-medium text-gray-400">No images indexed yet</div>
                <div className="text-xs text-gray-600">
                  Run <code className="bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">make index</code> to import the dataset
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Pagination */}
      {total > 0 && (
        <div className="flex items-center justify-between border-t border-gray-800 pt-3 flex-shrink-0">
          <span className="text-xs text-gray-500">
            {total.toLocaleString()} images · page {page} of {totalPages || 1}
            {hasFilters && <button onClick={clearFilters} className="ml-3 text-brand-400 hover:text-brand-300 transition-colors">Clear filters</button>}
          </span>
          <div className="flex gap-2">
            <button disabled={!hasPrev} onClick={() => setPage(page - 1)}
              className="px-3 py-1.5 rounded-lg bg-gray-800 disabled:opacity-40 hover:bg-gray-700 transition-colors text-sm text-gray-300">
              ← Prev
            </button>
            <button disabled={!hasNext} onClick={() => setPage(page + 1)}
              className="px-3 py-1.5 rounded-lg bg-gray-800 disabled:opacity-40 hover:bg-gray-700 transition-colors text-sm text-gray-300">
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
