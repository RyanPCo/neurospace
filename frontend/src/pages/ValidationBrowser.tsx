import { useEffect } from 'react'
import { useImageStore } from '../store/imageStore'
import { useAnnotationStore } from '../store/annotationStore'
import { ImageGrid } from '../components/image/ImageGrid'
import { MagnificationFilter } from '../components/image/MagnificationFilter'
import { GradCAMOverlay } from '../components/image/GradCAMOverlay'
import { ConfidenceGauge } from '../components/shared/ConfidenceGauge'
import { ClassBadge } from '../components/shared/ClassBadge'
import { AnnotationCanvas } from '../components/annotation/AnnotationCanvas'
import { LoadingSpinner } from '../components/shared/LoadingSpinner'
import { useGradCAM } from '../hooks/useGradCAM'
import { imagesApi } from '../api/images'

export function ValidationBrowser() {
  const { images, selectedImageId, selectedImage, loading, fetchImages } = useImageStore()
  const { fetchAnnotations } = useAnnotationStore()
  const { gradcam, gradcamLoading } = useGradCAM(selectedImageId)

  useEffect(() => {
    fetchImages()
  }, [])

  useEffect(() => {
    if (selectedImageId) {
      fetchAnnotations(selectedImageId)
    }
  }, [selectedImageId])

  return (
    <div className="flex gap-4 h-full">
      {/* Left: image browser */}
      <div className="flex-1 flex flex-col gap-3 min-w-0">
        <div className="flex items-center gap-4 flex-wrap">
          <h1 className="font-bold text-lg text-gray-100">Validation Browser</h1>
          <MagnificationFilter />
          <ClassFilter />
        </div>
        <ImageGrid />
      </div>

      {/* Right: detail panel */}
      {selectedImageId && (
        <div className="w-80 xl:w-96 flex flex-col gap-4 overflow-y-auto flex-shrink-0">
          {selectedImage ? (
            <>
              {/* Prediction info */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <ClassBadge cls={selectedImage.predicted_class} />
                  <span className="text-xs text-gray-500">GT: {selectedImage.ground_truth}</span>
                </div>
                <ConfidenceGauge
                  confidence={selectedImage.confidence}
                  predicted_class={selectedImage.predicted_class}
                />
                {selectedImage.subtype_predicted && (
                  <div className="text-xs text-gray-500">
                    Subtype: <span className="text-gray-300">{selectedImage.subtype_predicted}</span>
                  </div>
                )}
              </div>

              {/* GradCAM */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                <GradCAMOverlay
                  imageId={selectedImageId}
                  gradcam={gradcam}
                  loading={gradcamLoading}
                />
              </div>

              {/* Annotation canvas */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
                  Annotate
                </div>
                <AnnotationCanvas
                  imageId={selectedImageId}
                  width={320}
                  height={210}
                />
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-40">
              <LoadingSpinner />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ClassFilter() {
  const { filters, setFilters } = useImageStore()
  const CLASSES = ['malignant', 'benign']
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500">Class:</span>
      <div className="flex gap-1">
        <button
          onClick={() => setFilters({ predicted_class: undefined })}
          className={`px-2.5 py-1 rounded text-xs ${!filters.predicted_class ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
        >
          All
        </button>
        {CLASSES.map(c => (
          <button
            key={c}
            onClick={() => setFilters({ predicted_class: filters.predicted_class === c ? undefined : c })}
            className={`px-2.5 py-1 rounded text-xs capitalize ${filters.predicted_class === c ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
          >
            {c}
          </button>
        ))}
      </div>
    </div>
  )
}
