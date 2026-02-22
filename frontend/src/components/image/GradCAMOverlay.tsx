import { useState } from 'react'
import { OpacitySlider } from '../shared/OpacitySlider'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import type { GradCAMResult } from '../../types'
import { imagesApi } from '../../api/images'

interface Props {
  imageId: string
  modelVersion?: string | null
  gradcam: GradCAMResult | null
  loading: boolean
}

export function GradCAMOverlay({ imageId, modelVersion, gradcam, loading }: Props) {
  const [opacity, setOpacity] = useState(0.7)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Grad-CAM</span>
        {modelVersion && (
          <span className="text-xs text-gray-600 font-mono">{modelVersion}</span>
        )}
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center h-40 gap-3 text-gray-500">
          <LoadingSpinner />
          <span className="text-xs">Computing activation map…</span>
        </div>
      ) : gradcam ? (
        <div className="space-y-3">
          <div className="relative rounded-xl overflow-hidden bg-gray-800 shadow-inner">
            <img src={imagesApi.fileUrl(imageId)} alt="tissue" className="w-full block" />
            <img
              src={`data:image/png;base64,${gradcam.overlay_b64}`}
              alt="GradCAM heatmap"
              className="absolute inset-0 w-full h-full object-cover"
              style={{ opacity, mixBlendMode: 'multiply' }}
            />
          </div>
          <OpacitySlider value={opacity} onChange={setOpacity} label="Heatmap" />
          <div className="flex items-center gap-1.5 text-xs text-gray-600">
            <span>Top kernels:</span>
            <span className="font-mono text-gray-500">
              {gradcam.top_kernel_indices.slice(0, 5).join(', ')}
            </span>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-8 gap-2 text-gray-600 text-center">
          <div className="text-2xl opacity-40">🌡</div>
          <div className="text-xs">
            GradCAM unavailable.<br />
            Ensure the model is trained and loaded.
          </div>
        </div>
      )}
    </div>
  )
}
