import { create } from 'zustand'
import type { TrainingRun, TrainingEpoch, TrainingConfig, WSMessage } from '../types'
import { trainingApi } from '../api/training'

interface EpochData {
  epoch: number
  train_loss: number | null
  val_loss: number | null
  train_acc: number | null
  val_acc: number | null
}

interface TrainingStore {
  status: string
  currentRunId: string | null
  epochHistory: EpochData[]
  history: TrainingRun[]
  config: TrainingConfig
  loading: boolean
  error: string | null

  setConfig: (cfg: Partial<TrainingConfig>) => void
  startTraining: () => Promise<void>
  stopTraining: () => Promise<void>
  fetchHistory: () => Promise<void>
  fetchStatus: () => Promise<void>
  handleWSMessage: (msg: WSMessage) => void
}

const defaultConfig: TrainingConfig = {
  num_epochs: 10,
  learning_rate: 1e-4,
  weight_decay: 1e-4,
  batch_size: 32,
  annotation_weight: 0.3,
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  status: 'idle',
  currentRunId: null,
  epochHistory: [],
  history: [],
  config: defaultConfig,
  loading: false,
  error: null,

  setConfig: (cfg) => set(s => ({ config: { ...s.config, ...cfg } })),

  startTraining: async () => {
    const { config } = get()
    set({ loading: true, error: null, epochHistory: [] })
    try {
      const data = await trainingApi.start(config)
      set({ currentRunId: data.run_id, status: 'running', loading: false })
    } catch (e: any) {
      set({ error: e.message, loading: false })
    }
  },

  stopTraining: async () => {
    await trainingApi.stop()
    set({ status: 'stopping' })
  },

  fetchHistory: async () => {
    set({ loading: true })
    try {
      const data = await trainingApi.history()
      set({ history: data, loading: false })
    } catch (e: any) {
      set({ error: e.message, loading: false })
    }
  },

  fetchStatus: async () => {
    try {
      const data = await trainingApi.status()
      set({ status: data.status, currentRunId: data.run_id })
    } catch {}
  },

  handleWSMessage: (msg: WSMessage) => {
    if (msg.type === 'epoch_end') {
      const entry: EpochData = {
        epoch: msg.epoch!,
        train_loss: msg.train_loss ?? null,
        val_loss: msg.val_loss ?? null,
        train_acc: msg.train_acc ?? null,
        val_acc: msg.val_acc ?? null,
      }
      set(s => ({ epochHistory: [...s.epochHistory, entry] }))
    } else if (msg.type === 'completed') {
      set({ status: 'idle' })
      get().fetchHistory()
    } else if (msg.type === 'error') {
      set({ status: 'error', error: msg.message })
    }
  },
}))
