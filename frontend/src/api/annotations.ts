import client from './client'
import type { Annotation, AnnotationCreate } from '../types'

export const annotationsApi = {
  create: (data: AnnotationCreate) =>
    client.post<Annotation>('/annotations', data).then(r => r.data),

  list: (image_id: string) =>
    client.get<Annotation[]>('/annotations', { params: { image_id } }).then(r => r.data),

  delete: (id: number) =>
    client.delete(`/annotations/${id}`).then(r => r.data),
}
