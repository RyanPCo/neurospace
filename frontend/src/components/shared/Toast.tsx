import { useEffect, useState } from 'react'

interface Toast {
  id: string
  message: string
  type: 'error' | 'success' | 'info'
}

type Listener = (toasts: Toast[]) => void

// Tiny global toast bus (no extra deps needed)
let _toasts: Toast[] = []
const _listeners: Set<Listener> = new Set()

function notify() {
  _listeners.forEach(l => l([..._toasts]))
}

export const toast = {
  error: (message: string) => {
    const id = Math.random().toString(36).slice(2)
    _toasts = [..._toasts, { id, message, type: 'error' }]
    notify()
    setTimeout(() => toast.dismiss(id), 5000)
  },
  success: (message: string) => {
    const id = Math.random().toString(36).slice(2)
    _toasts = [..._toasts, { id, message, type: 'success' }]
    notify()
    setTimeout(() => toast.dismiss(id), 3500)
  },
  info: (message: string) => {
    const id = Math.random().toString(36).slice(2)
    _toasts = [..._toasts, { id, message, type: 'info' }]
    notify()
    setTimeout(() => toast.dismiss(id), 3500)
  },
  dismiss: (id: string) => {
    _toasts = _toasts.filter(t => t.id !== id)
    notify()
  },
}

const ICONS = { error: '✕', success: '✓', info: 'ℹ' }
const COLORS = {
  error: 'bg-red-900 border-red-700 text-red-100',
  success: 'bg-green-900 border-green-700 text-green-100',
  info: 'bg-gray-800 border-gray-600 text-gray-100',
}

export function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>([])

  useEffect(() => {
    _listeners.add(setToasts)
    return () => { _listeners.delete(setToasts) }
  }, [])

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map(t => (
        <div
          key={t.id}
          className={`flex items-start gap-3 px-4 py-3 rounded-lg border shadow-xl text-sm animate-fade-in ${COLORS[t.type]}`}
        >
          <span className="font-bold mt-0.5">{ICONS[t.type]}</span>
          <span className="flex-1">{t.message}</span>
          <button
            onClick={() => toast.dismiss(t.id)}
            className="opacity-60 hover:opacity-100 transition-opacity ml-1 text-xs leading-none"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}
