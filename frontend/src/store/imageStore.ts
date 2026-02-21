import { create } from 'zustand'
import type { ImageSummary, ImageDetail, GradCAMResult } from '../types'
import { imagesApi } from '../api/images'

interface ImageFilters {
  magnification?: string
  subtype?: string
  predicted_class?: string
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
  error: string | null

  fetchImages: () => Promise<void>
  setPage: (page: number) => void
  setFilters: (f: Partial<ImageFilters>) => void
  selectImage: (id: string | null) => Promise<void>
  fetchGradCAM: (id: string) => Promise<void>
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
  error: null,

  fetchImages: async () => {
    const { page, pageSize, filters } = get()
    set({ loading: true, error: null })
    try {
      const data = await imagesApi.list({ page, page_size: pageSize, ...filters })
      set({ images: data.items, total: data.total, loading: false })
    } catch (e: any) {
      set({ error: e.message, loading: false })
    }
  },

  setPage: (page) => {
    set({ page })
    get().fetchImages()
  },

  setFilters: (f) => {
    set({ filters: { ...get().filters, ...f }, page: 1 })
    get().fetchImages()
  },

  selectImage: async (id) => {
    if (!id) {
      set({ selectedImageId: null, selectedImage: null, gradcam: null })
      return
    }
    set({ selectedImageId: id, selectedImage: null })
    try {
      const detail = await imagesApi.detail(id)
      set({ selectedImage: detail })
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  fetchGradCAM: async (id) => {
    set({ gradcamLoading: true, gradcam: null })
    try {
      const result = await imagesApi.gradcam(id)
      set({ gradcam: result, gradcamLoading: false })
    } catch (e: any) {
      set({ error: e.message, gradcamLoading: false })
    }
  },

  clearGradCAM: () => set({ gradcam: null }),
}))
