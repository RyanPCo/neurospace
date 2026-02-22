import client from './client'
import type { Gradcam3DRunRequest, Gradcam3DRunResponse } from '../types'

export const volumesApi = {
  runGradcam3d: (payload: Gradcam3DRunRequest) =>
    client.post<Gradcam3DRunResponse>('/volumes/gradcam3d/run', payload).then(r => r.data),
}
