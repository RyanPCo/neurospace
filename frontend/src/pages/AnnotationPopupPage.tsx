import { useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { AnnotationCanvas } from '../components/annotation/AnnotationCanvas'
import { useImageStore } from '../store/imageStore'

export function AnnotationPopupPage() {
  const { imageId } = useParams<{ imageId: string }>()
  const { selectedImage, selectImage, retrainFromAnnotation, retraining } = useImageStore()

  useEffect(() => {
    if (imageId) {
      void selectImage(imageId)
    }
  }, [imageId, selectImage])

  if (!imageId) {
    return (
      <div className="h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="text-sm text-gray-400">Missing image id.</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6">
      <div className="max-w-6xl mx-auto space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold">Annotation Window</h1>
            <p className="text-xs text-gray-400 mt-1">
              Changes sync back to the main Validation Browser and trigger Grad-CAM refresh.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => void retrainFromAnnotation(imageId)}
              disabled={retraining}
              className={`text-xs px-2 py-1 rounded border ${
                retraining
                  ? 'border-gray-700 text-gray-500 cursor-not-allowed'
                  : 'border-emerald-700 text-emerald-300 hover:bg-emerald-900/30'
              }`}
            >
              {retraining ? 'Retraining...' : 'Retrain'}
            </button>
            <Link to="/" className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-300 hover:bg-gray-800">
              Back
            </Link>
          </div>
        </div>

        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <div className="text-xs text-gray-500 mb-3">
            Image: {selectedImage?.filename ?? imageId}
          </div>
          <AnnotationCanvas imageId={imageId} width={1100} height={720} />
        </div>
      </div>
    </div>
  )
}
