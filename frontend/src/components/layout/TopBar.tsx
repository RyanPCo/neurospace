import { useEffect } from 'react'
import { useTrainingStore } from '../../store/trainingStore'

export function TopBar() {
  const { status, fetchStatus } = useTrainingStore()

  useEffect(() => {
    fetchStatus()
    const t = setInterval(fetchStatus, 10000)
    return () => clearInterval(t)
  }, [fetchStatus])

  return (
    <header className="h-12 bg-gray-900 border-b border-gray-800 flex items-center px-4 gap-4">
      <span className="text-sm font-medium text-gray-300 flex-1">Breast Cancer Histopathology Workbench</span>
      <div className="flex items-center gap-2 text-xs">
        <span className="text-gray-500">Training:</span>
        <StatusBadge status={status} />
      </div>
    </header>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    idle: 'bg-gray-700 text-gray-300',
    running: 'bg-amber-500 text-black animate-pulse',
    stopping: 'bg-orange-500 text-black',
    completed: 'bg-green-600 text-white',
    error: 'bg-red-600 text-white',
  }
  return (
    <span className={`px-2 py-0.5 rounded-full font-medium ${colors[status] ?? colors.idle}`}>
      {status}
    </span>
  )
}
