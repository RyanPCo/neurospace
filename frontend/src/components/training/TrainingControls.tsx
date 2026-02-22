import { useTrainingStore } from '../../store/trainingStore'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { formatLoss, formatAcc } from '../../utils/formatters'

export function TrainingControls() {
  const { config, status, loading, currentEpoch, totalEpochs, latestMetrics, setConfig, startTraining, stopTraining } = useTrainingStore()
  const isRunning = status === 'running'
  const isStopping = status === 'stopping'

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-200">Training Configuration</h3>
        <StatusChip status={status} />
      </div>

      {/* Live metrics bar — shown during training */}
      {isRunning && (
        <div className="bg-gray-800 rounded-lg px-4 py-3 flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <LoadingSpinner size="sm" />
            <span className="text-amber-400 font-medium">
              {currentEpoch != null ? `Epoch ${currentEpoch}/${totalEpochs ?? config.num_epochs}` : 'Starting…'}
            </span>
          </div>
          {latestMetrics && (
            <>
              <Metric label="Train Loss" value={formatLoss(latestMetrics.train_loss)} />
              <Metric label="Val Loss" value={formatLoss(latestMetrics.val_loss)} />
              <Metric label="Val Acc" value={formatAcc(latestMetrics.val_acc)} />
            </>
          )}
        </div>
      )}

      {/* Config inputs */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {[
          { label: 'Epochs', key: 'num_epochs', type: 'number', min: 1, max: 100, step: 1 },
          { label: 'Learning Rate', key: 'learning_rate', type: 'number', min: 0.00001, max: 0.1, step: 0.00001 },
          { label: 'Batch Size', key: 'batch_size', type: 'number', min: 4, max: 128, step: 4 },
          { label: 'Weight Decay', key: 'weight_decay', type: 'number', min: 0, max: 0.1, step: 0.0001 },
          { label: 'Spatial Loss Weight', key: 'spatial_loss_weight', type: 'number', min: 0, max: 5, step: 0.05 },
        ].map(f => (
          <label key={f.key} className="space-y-1.5">
            <span className="text-xs font-medium text-gray-500">{f.label}</span>
            <input
              type={f.type} min={f.min} max={f.max} step={f.step}
              value={config[f.key as keyof typeof config]}
              onChange={e => setConfig({ [f.key]: Number(e.target.value) })}
              disabled={isRunning || isStopping}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 disabled:opacity-50 focus:outline-none focus:border-brand-500 transition-colors"
            />
          </label>
        ))}
      </div>

      <div className="flex gap-3 pt-1">
        {!isRunning && !isStopping ? (
          <button onClick={startTraining} disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 bg-brand-600 text-white rounded-lg text-sm font-medium hover:bg-brand-700 disabled:opacity-50 transition-colors shadow-md shadow-brand-600/20">
            {loading && <LoadingSpinner size="sm" />}
            Start Training
          </button>
        ) : (
          <button onClick={stopTraining} disabled={isStopping}
            className="px-5 py-2.5 bg-red-700 text-white rounded-lg text-sm font-medium hover:bg-red-600 disabled:opacity-50 transition-colors">
            {isStopping ? 'Stopping…' : 'Stop Training'}
          </button>
        )}
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-sm font-mono text-gray-200">{value}</div>
    </div>
  )
}

function StatusChip({ status }: { status: string }) {
  const map: Record<string, string> = {
    idle: 'bg-gray-700 text-gray-400',
    running: 'bg-amber-500/20 text-amber-400 border border-amber-500/30',
    stopping: 'bg-orange-500/20 text-orange-400 border border-orange-500/30',
    completed: 'bg-green-500/20 text-green-400 border border-green-500/30',
    error: 'bg-red-500/20 text-red-400 border border-red-500/30',
  }
  return (
    <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${map[status] ?? map.idle}`}>
      {status}
    </span>
  )
}
