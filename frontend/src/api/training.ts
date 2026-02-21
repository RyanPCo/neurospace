import client from './client'
import type { TrainingRun, TrainingStatus, TrainingConfig, TrainingEpoch } from '../types'

export const trainingApi = {
  start: (config: TrainingConfig) =>
    client.post<{ run_id: string; message: string }>('/training/start', config).then(r => r.data),

  stop: () =>
    client.post('/training/stop').then(r => r.data),

  status: () =>
    client.get<TrainingStatus>('/training/status').then(r => r.data),

  history: () =>
    client.get<TrainingRun[]>('/training/history').then(r => r.data),

  epochs: (run_id: string) =>
    client.get<TrainingEpoch[]>(`/training/${run_id}/epochs`).then(r => r.data),
}
