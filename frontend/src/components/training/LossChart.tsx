import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useTrainingStore } from '../../store/trainingStore'

export function LossChart() {
  const { epochHistory } = useTrainingStore()

  const data = epochHistory.map(e => ({
    epoch: e.epoch,
    'Train Loss': e.train_loss != null ? Number(e.train_loss.toFixed(4)) : null,
    'Val Loss': e.val_loss != null ? Number(e.val_loss.toFixed(4)) : null,
  }))

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
      <h3 className="font-semibold text-sm text-gray-300 mb-4">Loss</h3>
      {data.length === 0 ? (
        <div className="text-center text-gray-600 py-12 text-sm">No training data yet</div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
            <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
            <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="Train Loss" stroke="#0ea5e9" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Val Loss" stroke="#f59e0b" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
