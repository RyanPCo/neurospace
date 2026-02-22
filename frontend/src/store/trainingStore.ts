import { create } from 'zustand'
import type { TrainingRun, TrainingConfig, WSMessage } from '../types'
import { trainingApi } from '../api/training'
import { toast } from '../components/shared/Toast'

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
  currentEpoch: number | null
  totalEpochs: number | null
  latestMetrics: Record<string, number | null> | null
  epochHistory: EpochData[]
  history: TrainingRun[]
  config: TrainingConfig
  loading: boolean

  setConfig: (cfg: Partial<TrainingConfig>) => void
  startTraining: () => Promise<void>
  stopTraining: () => Promise<void>
  fetchHistory: () => Promise<void>
  fetchStatus: () => Promise<void>
  handleWSMessage: (msg: WSMessage) => void
}

const defaultConfig: TrainingConfig = {
  num_epochs: 10, learning_rate: 1e-4, weight_decay: 1e-4, batch_size: 32, spatial_loss_weight: 0.3,
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  status: 'idle',
  currentRunId: null,
  currentEpoch: null,
  totalEpochs: null,
  latestMetrics: null,
  epochHistory: [],
  history: [],
  config: defaultConfig,
  loading: false,

  setConfig: (cfg) => set(s => ({ config: { ...s.config, ...cfg } })),

  startTraining: async () => {
    const { config } = get()
    set({ loading: true, epochHistory: [], currentEpoch: null, latestMetrics: null })
    try {
      const data = await trainingApi.start(config)
      set({ currentRunId: data.run_id, status: 'running', totalEpochs: config.num_epochs, loading: false })
      toast.info('Training started')
    } catch (e: any) {
      set({ loading: false })
      toast.error(`Failed to start training: ${e.message}`)
    }
  },

  stopTraining: async () => {
    await trainingApi.stop()
    set({ status: 'stopping' })
    toast.info('Stop signal sent — finishing current batch…')
  },

  fetchHistory: async () => {
    set({ loading: true })
    try {
      const data = await trainingApi.history()
      set({ history: data, loading: false })
    } catch (e: any) {
      set({ loading: false })
      toast.error(`Failed to fetch training history: ${e.message}`)
    }
  },

  fetchStatus: async () => {
    try {
      const data = await trainingApi.status()
      set({ status: data.status, currentRunId: data.run_id })
    } catch {}
  },

  handleWSMessage: (msg: WSMessage) => {
    if (msg.type === 'epoch_start') {
      set({ currentEpoch: msg.epoch ?? null })
    } else if (msg.type === 'epoch_end') {
      const entry: EpochData = {
        epoch: msg.epoch!,
        train_loss: msg.train_loss ?? null,
        val_loss: msg.val_loss ?? null,
        train_acc: msg.train_acc ?? null,
        val_acc: msg.val_acc ?? null,
      }
      set(s => ({
        epochHistory: [...s.epochHistory, entry],
        currentEpoch: msg.epoch ?? null,
        latestMetrics: {
          train_loss: msg.train_loss ?? null,
          val_loss: msg.val_loss ?? null,
          train_acc: msg.train_acc ?? null,
          val_acc: msg.val_acc ?? null,
        },
      }))
    } else if (msg.type === 'completed') {
      set({ status: 'idle', currentEpoch: null })
      toast.success('Training complete!')
      get().fetchHistory()
    } else if (msg.type === 'error') {
      set({ status: 'error' })
      toast.error(`Training error: ${msg.message}`)
    }
  },
}))
