import { useImageStore } from '../../store/imageStore'
import { ImageCard } from './ImageCard'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { usePagination } from '../../hooks/usePagination'

export function ImageGrid() {
  const { images, total, page, pageSize, loading, selectedImageId, setPage, selectImage } = useImageStore()
  const { totalPages, hasNext, hasPrev } = usePagination(total, page, pageSize)

  if (loading && images.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="grid grid-cols-3 xl:grid-cols-4 gap-3 flex-1 overflow-auto content-start pb-2">
        {images.map(img => (
          <ImageCard
            key={img.id}
            image={img}
            selected={img.id === selectedImageId}
            onClick={() => selectImage(img.id === selectedImageId ? null : img.id)}
          />
        ))}
        {images.length === 0 && !loading && (
          <div className="col-span-4 text-center text-gray-500 py-16 text-sm">
            No images found. Run <code className="bg-gray-800 px-1 rounded">make index</code> to index the dataset.
          </div>
        )}
      </div>
      {/* Pagination */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-800 text-sm text-gray-400 flex-shrink-0">
        <span>{total} images · page {page}/{totalPages || 1}</span>
        <div className="flex gap-2">
          <button
            disabled={!hasPrev}
            onClick={() => setPage(page - 1)}
            className="px-3 py-1 rounded bg-gray-800 disabled:opacity-40 hover:bg-gray-700 transition-colors"
          >
            Prev
          </button>
          <button
            disabled={!hasNext}
            onClick={() => setPage(page + 1)}
            className="px-3 py-1 rounded bg-gray-800 disabled:opacity-40 hover:bg-gray-700 transition-colors"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  )
}
