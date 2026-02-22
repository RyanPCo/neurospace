import client from './client'
import type {
  AnnotationRetrainRequest,
  AnnotationRetrainResult,
  PaginatedImages,
  ImageDetail,
  GradCAMResult,
} from '../types'

interface ImageFilters {
  page?: number
  page_size?: number
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

  retrainFromAnnotation: (id: string, body: AnnotationRetrainRequest = {}) =>
    client.post<AnnotationRetrainResult>(`/images/${id}/retrain-from-annotation`, body).then(r => r.data),

  fileUrl: (id: string) => `/api/images/${id}/file`,
}
