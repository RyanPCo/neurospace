import { useEffect, useRef } from 'react'
import { useTrainingStore } from '../store/trainingStore'
import type { WSMessage } from '../types'

export function useTrainingWS() {
  const wsRef = useRef<WebSocket | null>(null)
  const handleWSMessage = useTrainingStore(s => s.handleWSMessage)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data)
        handleWSMessage(msg)
      } catch {}
    }

    ws.onerror = () => {
      console.warn('Training WebSocket error')
    }

    return () => {
      ws.close()
    }
  }, [handleWSMessage])

  return wsRef
}
