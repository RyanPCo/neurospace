import { useEffect } from 'react'
import { useImageStore } from '../store/imageStore'

export function useGradCAM(imageId: string | null) {
  const { gradcam, gradcamLoading, fetchGradCAM, clearGradCAM } = useImageStore()

  useEffect(() => {
    if (imageId) {
      fetchGradCAM(imageId)
    } else {
      clearGradCAM()
    }
    return () => clearGradCAM()
  }, [imageId])

  return { gradcam, gradcamLoading }
}
