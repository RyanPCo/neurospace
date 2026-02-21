import client from './client'
import type { PaginatedKernels, KernelSummary, KernelActivations } from '../types'

interface KernelFilters {
  page?: number
  page_size?: number
  layer_name?: string
  assigned_class?: string
  sort_by?: string
}

export const kernelsApi = {
  list: (filters: KernelFilters = {}) =>
    client.get<PaginatedKernels>('/kernels', { params: filters }).then(r => r.data),

  imageUrl: (id: string) => `/api/kernels/${id}/image`,

  activations: (id: string) =>
    client.get<KernelActivations>(`/kernels/${id}/activations`).then(r => r.data),

  update: (id: string, data: { assigned_class?: string; doctor_notes?: string }) =>
    client.put<KernelSummary>(`/kernels/${id}`, data).then(r => r.data),

  delete: (id: string) =>
    client.delete(`/kernels/${id}`).then(r => r.data),

  batchUpdate: (kernel_ids: string[], action: string, assigned_class?: string) =>
    client.post('/kernels/batch', { kernel_ids, action, assigned_class }).then(r => r.data),
}
