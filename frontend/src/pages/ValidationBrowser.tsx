import { useEffect } from 'react'
import { useImageStore } from '../store/imageStore'
import { useAnnotationStore } from '../store/annotationStore'
import { ImageGrid } from '../components/image/ImageGrid'
import { GradCAMOverlay } from '../components/image/GradCAMOverlay'
import { ConfidenceGauge } from '../components/shared/ConfidenceGauge'
import { ClassBadge } from '../components/shared/ClassBadge'
import { AnnotationCanvas } from '../components/annotation/AnnotationCanvas'
import { LoadingSpinner } from '../components/shared/LoadingSpinner'
import { useGradCAM } from '../hooks/useGradCAM'

const TUMOR_CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
const CLASS_COLORS: Record<string, string> = {
  glioma:     'bg-red-600 text-white',
  meningioma: 'bg-orange-500 text-white',
  pituitary:  'bg-blue-600 text-white',
  notumor:    'bg-green-600 text-white',
}
const CLASS_LABELS: Record<string, string> = { notumor: 'No Tumor' }

export function ValidationBrowser() {
  const {
    images,
    selectedImageId,
    selectedImage,
    loading,
    retraining,
    fetchImages,
    fetchGradCAM,
    retrainFromAnnotation,
  } = useImageStore()
  const { fetchAnnotations } = useAnnotationStore()
  const { gradcam, gradcamLoading } = useGradCAM(selectedImageId)

  useEffect(() => { fetchImages() }, [])
  useEffect(() => {
    if (selectedImageId) fetchAnnotations(selectedImageId)
  }, [selectedImageId])

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key !== 'cancerscope:annotation-updated' || !e.newValue || !selectedImageId) return
      try {
        const payload = JSON.parse(e.newValue) as { imageId?: string }
        if (payload.imageId === selectedImageId) {
          fetchAnnotations(selectedImageId)
          fetchGradCAM(selectedImageId)
        }
      } catch {
        // ignore malformed sync payload
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [selectedImageId, fetchAnnotations, fetchGradCAM])

  return (
    <div className="flex gap-4 h-full">
      {/* Left: image browser */}
      <div className="flex-1 flex flex-col gap-3 min-w-0">
        <div className="flex items-center gap-4 flex-wrap">
          <h1 className="font-bold text-lg text-gray-100">Brain Tumor MRI Browser</h1>
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
              </div>

              {/* GradCAM */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                <GradCAMOverlay
                  imageId={selectedImageId}
                  modelVersion={selectedImage?.model_version}
                  gradcam={gradcam}
                  loading={gradcamLoading}
                />
              </div>

              {/* Annotation canvas */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
                    Annotate
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => void retrainFromAnnotation(selectedImageId)}
                      disabled={retraining}
                      className={`text-xs px-2 py-1 rounded border ${
                        retraining
                          ? 'border-gray-700 text-gray-500 cursor-not-allowed'
                          : 'border-emerald-700 text-emerald-300 hover:bg-emerald-900/30'
                      }`}
                    >
                      {retraining ? 'Retraining...' : 'Retrain'}
                    </button>
                    <button
                      onClick={() => window.open(`/annotate/${selectedImageId}`, '_blank', 'width=1280,height=900')}
                      className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-300 hover:bg-gray-800"
                    >
                      Pop Out
                    </button>
                  </div>
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
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500 font-medium">Class:</span>
      <div className="flex gap-1 bg-gray-800 p-1 rounded-lg flex-wrap">
        <button
          onClick={() => setFilters({ predicted_class: undefined })}
          className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
            !filters.predicted_class ? 'bg-gray-700 text-white shadow' : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          All
        </button>
        {TUMOR_CLASSES.map(c => (
          <button
            key={c}
            onClick={() => setFilters({ predicted_class: filters.predicted_class === c ? undefined : c })}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
              filters.predicted_class === c
                ? CLASS_COLORS[c]
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {CLASS_LABELS[c] ?? c.charAt(0).toUpperCase() + c.slice(1)}
          </button>
        ))}
      </div>
    </div>
  )
}
