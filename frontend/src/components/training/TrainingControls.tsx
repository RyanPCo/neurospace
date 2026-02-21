import { useTrainingStore } from '../../store/trainingStore'
import { LoadingSpinner } from '../shared/LoadingSpinner'

export function TrainingControls() {
  const { config, status, loading, setConfig, startTraining, stopTraining } = useTrainingStore()
  const isRunning = status === 'running'

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 space-y-4">
      <h3 className="font-semibold text-sm text-gray-300">Training Configuration</h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <label className="space-y-1">
          <span className="text-xs text-gray-500">Epochs</span>
          <input
            type="number"
            min={1} max={100}
            value={config.num_epochs}
            onChange={e => setConfig({ num_epochs: Number(e.target.value) })}
            disabled={isRunning}
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 disabled:opacity-50"
          />
        </label>
        <label className="space-y-1">
          <span className="text-xs text-gray-500">Learning Rate</span>
          <input
            type="number"
            step="0.0001" min={0.00001} max={0.1}
            value={config.learning_rate}
            onChange={e => setConfig({ learning_rate: Number(e.target.value) })}
            disabled={isRunning}
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 disabled:opacity-50"
          />
        </label>
        <label className="space-y-1">
          <span className="text-xs text-gray-500">Batch Size</span>
          <input
            type="number"
            min={4} max={128} step={4}
            value={config.batch_size}
            onChange={e => setConfig({ batch_size: Number(e.target.value) })}
            disabled={isRunning}
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 disabled:opacity-50"
          />
        </label>
        <label className="space-y-1">
          <span className="text-xs text-gray-500">Weight Decay</span>
          <input
            type="number"
            step="0.0001" min={0}
            value={config.weight_decay}
            onChange={e => setConfig({ weight_decay: Number(e.target.value) })}
            disabled={isRunning}
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 disabled:opacity-50"
          />
        </label>
        <label className="space-y-1">
          <span className="text-xs text-gray-500">Annotation Weight</span>
          <input
            type="number"
            step="0.05" min={0} max={1}
            value={config.annotation_weight}
            onChange={e => setConfig({ annotation_weight: Number(e.target.value) })}
            disabled={isRunning}
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 disabled:opacity-50"
          />
        </label>
      </div>

      <div className="flex gap-3 pt-1">
        {!isRunning ? (
          <button
            onClick={startTraining}
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2 bg-brand-600 text-white rounded-lg text-sm font-medium hover:bg-brand-700 disabled:opacity-50 transition-colors"
          >
            {loading && <LoadingSpinner size="sm" />}
            Start Training
          </button>
        ) : (
          <button
            onClick={stopTraining}
            className="px-5 py-2 bg-red-700 text-white rounded-lg text-sm font-medium hover:bg-red-600 transition-colors"
          >
            Stop Training
          </button>
        )}
        <div className="flex items-center gap-2 text-sm text-gray-500">
          Status: <span className="text-gray-300">{status}</span>
        </div>
      </div>
    </div>
  )
}
