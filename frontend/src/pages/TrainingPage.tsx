import { useEffect, useState } from 'react'
import { useTrainingStore } from '../store/trainingStore'
import { TrainingControls } from '../components/training/TrainingControls'
import { LossChart } from '../components/training/LossChart'
import { AccuracyChart } from '../components/training/AccuracyChart'
import { useTrainingWS } from '../hooks/useTrainingWS'
import { formatDatetime } from '../utils/formatters'

export function TrainingPage() {
  const { history, fetchHistory, fetchStatus } = useTrainingStore()
  const [showAnim, setShowAnim] = useState(false)
  useTrainingWS()

  useEffect(() => {
    fetchStatus()
    fetchHistory()
  }, [])

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <h1 className="font-bold text-lg text-gray-100">Training Dashboard</h1>
        <button
          onClick={() => setShowAnim(v => !v)}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-gray-800 text-amber-400 hover:bg-gray-700 transition-colors border border-gray-700"
        >
          ▶ {showAnim ? 'Hide' : 'Show'} Math Animation
        </button>
      </div>

      {showAnim && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800 flex items-center justify-between">
            <span className="text-sm font-semibold text-gray-300">Constrained GradCAM Loss — How It Works</span>
            <button onClick={() => setShowAnim(false)} className="text-gray-600 hover:text-gray-400 text-xs">✕ Close</button>
          </div>
          <video
            src="/constrained_cam.mp4"
            controls
            autoPlay
            className="w-full"
            style={{ background: '#111827' }}
          />
        </div>
      )}

      <TrainingControls />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <LossChart />
        <AccuracyChart />
      </div>

      {/* History */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
        <h3 className="font-semibold text-sm text-gray-300 mb-4">Training History</h3>
        {history.length === 0 ? (
          <div className="text-sm text-gray-600">No training runs yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-gray-400">
              <thead>
                <tr className="border-b border-gray-800 text-gray-500">
                  <th className="text-left py-2 pr-4">Run ID</th>
                  <th className="text-left py-2 pr-4">Status</th>
                  <th className="text-left py-2 pr-4">Started</th>
                  <th className="text-right py-2 pr-4">Val Loss</th>
                  <th className="text-right py-2">Val Acc</th>
                </tr>
              </thead>
              <tbody>
                {history.map(run => (
                  <tr key={run.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2 pr-4 font-mono">{run.id.slice(0, 8)}</td>
                    <td className="py-2 pr-4">
                      <StatusPill status={run.status} />
                    </td>
                    <td className="py-2 pr-4">{formatDatetime(run.start_time)}</td>
                    <td className="py-2 pr-4 text-right font-mono">
                      {run.final_val_loss?.toFixed(4) ?? '—'}
                    </td>
                    <td className="py-2 text-right font-mono">
                      {run.final_val_acc != null ? `${(run.final_val_acc * 100).toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function StatusPill({ status }: { status: string }) {
  const colors: Record<string, string> = {
    completed: 'bg-green-900 text-green-300',
    running: 'bg-amber-900 text-amber-300',
    error: 'bg-red-900 text-red-300',
    stopped: 'bg-gray-700 text-gray-400',
    pending: 'bg-gray-700 text-gray-400',
  }
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[status] ?? 'bg-gray-700 text-gray-400'}`}>
      {status}
    </span>
  )
}
