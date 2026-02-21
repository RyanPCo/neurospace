import { useEffect } from 'react'
import { useKernelStore } from '../store/kernelStore'
import { KernelGallery } from '../components/kernel/KernelGallery'
import { KernelDetail } from '../components/kernel/KernelDetail'

const LAYERS = [
  'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.conv3',
  'layer4.1.conv1', 'layer4.1.conv2', 'layer4.1.conv3',
  'layer4.2.conv1', 'layer4.2.conv2', 'layer4.2.conv3',
]

export function KernelExplorer() {
  const { selectedLayer, setLayer, fetchKernels, selectedKernel } = useKernelStore()

  useEffect(() => {
    fetchKernels()
  }, [])

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <h1 className="font-bold text-lg text-gray-100">Kernel Explorer</h1>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Layer:</span>
          <select
            value={selectedLayer}
            onChange={e => setLayer(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100"
          >
            <option value="">All layers</option>
            {LAYERS.map(l => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </div>
      </div>

      <KernelGallery />
      {selectedKernel && <KernelDetail />}
    </div>
  )
}
