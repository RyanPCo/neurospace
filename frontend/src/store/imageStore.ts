import { create } from 'zustand'
import type { AnnotationRetrainResult, ImageSummary, ImageDetail, GradCAMResult } from '../types'
import { imagesApi } from '../api/images'
import { toast } from '../components/shared/Toast'

interface ImageFilters {
  predicted_class?: string
  split?: string
}

interface ImageStore {
  images: ImageSummary[]
  total: number
  page: number
  pageSize: number
  filters: ImageFilters
  selectedImageId: string | null
  selectedImage: ImageDetail | null
  gradcam: GradCAMResult | null
  loading: boolean
  gradcamLoading: boolean
  retraining: boolean

  fetchImages: () => Promise<void>
  setPage: (page: number) => void
  setFilters: (f: Partial<ImageFilters>) => void
  clearFilters: () => void
  selectImage: (id: string | null) => Promise<void>
  fetchGradCAM: (id: string) => Promise<void>
  retrainFromAnnotation: (id: string) => Promise<AnnotationRetrainResult | null>
  clearGradCAM: () => void
}

export const useImageStore = create<ImageStore>((set, get) => ({
  images: [],
  total: 0,
  page: 1,
  pageSize: 50,
  filters: {},
  selectedImageId: null,
  selectedImage: null,
  gradcam: null,
  loading: false,
  gradcamLoading: false,
  retraining: false,

  fetchImages: async () => {
    const { page, pageSize, filters } = get()
    set({ loading: true })
    try {
      const data = await imagesApi.list({ page, page_size: pageSize, ...filters })
      set({ images: data.items, total: data.total, loading: false })
    } catch (e: any) {
      set({ loading: false })
      toast.error(`Failed to load images: ${e.message}`)
    }
  },

  setPage: (page) => { set({ page }); get().fetchImages() },

  setFilters: (f) => {
    set({ filters: { ...get().filters, ...f }, page: 1 })
    get().fetchImages()
  },

  clearFilters: () => {
    set({ filters: {}, page: 1 })
    get().fetchImages()
  },

  selectImage: async (id) => {
    if (!id) { set({ selectedImageId: null, selectedImage: null, gradcam: null }); return }
    set({ selectedImageId: id, selectedImage: null })
    try {
      const detail = await imagesApi.detail(id)
      set({ selectedImage: detail })
    } catch (e: any) {
      toast.error(`Failed to load image details: ${e.message}`)
    }
  },

  fetchGradCAM: async (id) => {
    set({ gradcamLoading: true, gradcam: null })
    try {
      const result = await imagesApi.gradcam(id)
      set({ gradcam: result, gradcamLoading: false })
    } catch (e: any) {
      set({ gradcamLoading: false })
      const detail = e?.response?.data?.detail || e?.message || 'Unknown GradCAM error'
      toast.error(`GradCAM failed: ${detail}`)
    }
  },

  retrainFromAnnotation: async (id) => {
    set({ retraining: true })
    try {
      const result = await imagesApi.retrainFromAnnotation(id, {})
      set({ retraining: false })
      toast.success('Retraining completed; model reloaded.')
      await get().fetchGradCAM(id)
      return result
    } catch (e: any) {
      set({ retraining: false })
      const detail = e?.response?.data?.detail || e?.message || 'Unknown retraining error'
      toast.error(`Retraining failed: ${detail}`)
      return null
    }
  },

  clearGradCAM: () => set({ gradcam: null }),
}))
