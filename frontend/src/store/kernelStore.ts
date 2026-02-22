import { create } from 'zustand'
import type { KernelSummary, KernelActivations } from '../types'
import { kernelsApi } from '../api/kernels'
import { toast } from '../components/shared/Toast'

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
  actioningId: string | null  // kernel id currently being acted on

  fetchKernels: () => Promise<void>
  setPage: (page: number) => void
  setLayer: (layer: string) => void
  selectKernel: (k: KernelSummary | null) => void
  fetchActivations: (id: string) => Promise<void>
  deleteKernel: (id: string) => Promise<void>
  reclassifyKernel: (id: string, cls: string) => Promise<void>
  updateNotes: (id: string, notes: string) => Promise<void>
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
  actioningId: null,

  fetchKernels: async () => {
    const { page, pageSize, selectedLayer } = get()
    set({ loading: true })
    try {
      const data = await kernelsApi.list({
        page, page_size: pageSize,
        layer_name: selectedLayer || undefined,
        sort_by: 'importance',
      })
      set({ kernels: data.items, total: data.total, loading: false })
    } catch (e: any) {
      set({ loading: false })
      toast.error(`Failed to load kernels: ${e.message}`)
    }
  },

  setPage: (page) => { set({ page }); get().fetchKernels() },
  setLayer: (layer) => { set({ selectedLayer: layer, page: 1 }); get().fetchKernels() },
  selectKernel: (k) => set({ selectedKernel: k, activations: null }),

  fetchActivations: async (id) => {
    set({ activationsLoading: true })
    try {
      const data = await kernelsApi.activations(id)
      set({ activations: data, activationsLoading: false })
    } catch (e: any) {
      set({ activationsLoading: false })
      toast.error(`Failed to load activations: ${e.message}`)
    }
  },

  deleteKernel: async (id) => {
    set({ actioningId: id })
    try {
      await kernelsApi.delete(id)
      set(state => ({
        kernels: state.kernels.map(k => k.id === id ? { ...k, is_deleted: true } : k),
        actioningId: null,
        selectedKernel: state.selectedKernel?.id === id ? null : state.selectedKernel,
      }))
      toast.success('Kernel marked as deleted — will be zeroed on next retrain')
    } catch (e: any) {
      set({ actioningId: null })
      toast.error(`Failed to delete kernel: ${e.message}`)
    }
  },

  reclassifyKernel: async (id, cls) => {
    set({ actioningId: id })
    try {
      const updated = await kernelsApi.update(id, { assigned_class: cls })
      set(state => ({
        kernels: state.kernels.map(k => k.id === id ? updated : k),
        selectedKernel: state.selectedKernel?.id === id ? updated : state.selectedKernel,
        actioningId: null,
      }))
      toast.success(`Kernel reclassified as ${cls}`)
    } catch (e: any) {
      set({ actioningId: null })
      toast.error(`Failed to reclassify kernel: ${e.message}`)
    }
  },

  updateNotes: async (id, notes) => {
    try {
      const updated = await kernelsApi.update(id, { doctor_notes: notes })
      set(state => ({
        kernels: state.kernels.map(k => k.id === id ? updated : k),
        selectedKernel: state.selectedKernel?.id === id ? updated : state.selectedKernel,
      }))
    } catch (e: any) {
      toast.error(`Failed to save notes: ${e.message}`)
    }
  },
}))
