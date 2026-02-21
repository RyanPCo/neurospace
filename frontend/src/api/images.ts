import client from './client'
import type { PaginatedImages, ImageDetail, GradCAMResult } from '../types'

interface ImageFilters {
  page?: number
  page_size?: number
  magnification?: string
  subtype?: string
  predicted_class?: string
  split?: string
}

export const imagesApi = {
  list: (filters: ImageFilters = {}) =>
    client.get<PaginatedImages>('/images', { params: filters }).then(r => r.data),

  detail: (id: string) =>
    client.get<ImageDetail>(`/images/${id}`).then(r => r.data),

  gradcam: (id: string) =>
    client.get<GradCAMResult>(`/images/${id}/gradcam`).then(r => r.data),

  fileUrl: (id: string) => `/api/images/${id}/file`,
}
