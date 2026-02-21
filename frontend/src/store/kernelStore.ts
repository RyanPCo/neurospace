import { create } from 'zustand'
import type { KernelSummary, KernelActivations } from '../types'
import { kernelsApi } from '../api/kernels'

interface KernelStore {
  kernels: KernelSummary[]
  total: number
  page: number
  pageSize: number
  selectedLayer: string
  selectedKernel: KernelSummary | null
  activations: KernelActivations | null
  loading: boolean
  activationsLoading: boolean
  error: string | null

  fetchKernels: () => Promise<void>
  setPage: (page: number) => void
  setLayer: (layer: string) => void
  selectKernel: (k: KernelSummary | null) => void
  fetchActivations: (id: string) => Promise<void>
  deleteKernel: (id: string) => Promise<void>
  reclassifyKernel: (id: string, cls: string) => Promise<void>
}

export const useKernelStore = create<KernelStore>((set, get) => ({
  kernels: [],
  total: 0,
  page: 1,
  pageSize: 100,
  selectedLayer: '',
  selectedKernel: null,
  activations: null,
  loading: false,
  activationsLoading: false,
  error: null,

  fetchKernels: async () => {
    const { page, pageSize, selectedLayer } = get()
    set({ loading: true, error: null })
    try {
      const data = await kernelsApi.list({
        page, page_size: pageSize,
        layer_name: selectedLayer || undefined,
        sort_by: 'importance',
      })
      set({ kernels: data.items, total: data.total, loading: false })
    } catch (e: any) {
      set({ error: e.message, loading: false })
    }
  },

  setPage: (page) => {
    set({ page })
    get().fetchKernels()
  },

  setLayer: (layer) => {
    set({ selectedLayer: layer, page: 1 })
    get().fetchKernels()
  },

  selectKernel: (k) => set({ selectedKernel: k, activations: null }),

  fetchActivations: async (id) => {
    set({ activationsLoading: true })
    try {
      const data = await kernelsApi.activations(id)
      set({ activations: data, activationsLoading: false })
    } catch (e: any) {
      set({ error: e.message, activationsLoading: false })
    }
  },

  deleteKernel: async (id) => {
    await kernelsApi.delete(id)
    set(state => ({
      kernels: state.kernels.map(k => k.id === id ? { ...k, is_deleted: true } : k),
    }))
  },

  reclassifyKernel: async (id, cls) => {
    const updated = await kernelsApi.update(id, { assigned_class: cls })
    set(state => ({
      kernels: state.kernels.map(k => k.id === id ? updated : k),
    }))
  },
}))
