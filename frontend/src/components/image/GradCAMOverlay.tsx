import { useState } from 'react'
import { OpacitySlider } from '../shared/OpacitySlider'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import type { GradCAMResult } from '../../types'
import { imagesApi } from '../../api/images'

interface Props {
  imageId: string
  gradcam: GradCAMResult | null
  loading: boolean
}

export function GradCAMOverlay({ imageId, gradcam, loading }: Props) {
  const [opacity, setOpacity] = useState(0.6)

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Grad-CAM</div>

      {loading ? (
        <div className="flex items-center justify-center h-40">
          <LoadingSpinner />
        </div>
      ) : gradcam ? (
        <div className="space-y-3">
          <div className="relative rounded-lg overflow-hidden bg-gray-800">
            <img
              src={imagesApi.fileUrl(imageId)}
              alt="tissue"
              className="w-full block"
            />
            <img
              src={`data:image/png;base64,${gradcam.overlay_b64}`}
              alt="GradCAM overlay"
              className="absolute inset-0 w-full h-full object-cover mix-blend-multiply"
              style={{ opacity }}
            />
          </div>
          <OpacitySlider value={opacity} onChange={setOpacity} label="Heatmap" />
          <div className="text-xs text-gray-500">
            Top activated kernels: {gradcam.top_kernel_indices.slice(0, 5).join(', ')}
          </div>
        </div>
      ) : (
        <div className="text-sm text-gray-500 py-6 text-center">
          No GradCAM available. Ensure model is trained.
        </div>
      )}
    </div>
  )
}
