import { useRef, useState, useCallback } from 'react'
import { Stage, Layer, Image as KonvaImage, Line, Circle, Group } from 'react-konva'
import useImage from 'use-image'
import { useAnnotationStore } from '../../store/annotationStore'
import { ToolBar } from './ToolBar'
import { AnnotationList } from './AnnotationList'
import { LoadingSpinner } from '../shared/LoadingSpinner'
import { imagesApi } from '../../api/images'

interface Props { imageId: string; width: number; height: number }

function AnnotationOverlay({ stageW, stageH }: { stageW: number; stageH: number }) {
  const { annotations } = useAnnotationStore()
  return (
    <>
      {annotations.map(ann => {
        const colorMap: Record<string, string> = {
          glioma: 'rgba(239,68,68,0.35)', meningioma: 'rgba(249,115,22,0.35)',
          pituitary: 'rgba(37,99,235,0.35)', notumor: 'rgba(34,197,94,0.35)',
          gradcam_focus: 'rgba(245,158,11,0.35)',
        }
        const strokeMap: Record<string, string> = {
          glioma: '#ef4444', meningioma: '#f97316',
          pituitary: '#2563eb', notumor: '#22c55e',
          gradcam_focus: '#f59e0b',
        }
        const color  = colorMap[ann.label_class]  ?? 'rgba(156,163,175,0.35)'
        const stroke = strokeMap[ann.label_class] ?? '#9ca3af'
        if (ann.geometry_type === 'polygon') {
          const geom = JSON.parse(ann.geometry_json)
          const pts = geom.points.flatMap((p: any) => [p.x * stageW, p.y * stageH])
          return <Line key={ann.id} points={pts} closed fill={color} stroke={stroke} strokeWidth={2} />
        }
        if (ann.geometry_type === 'brush') {
          const geom = JSON.parse(ann.geometry_json)
          const r = geom.radius * Math.min(stageW, stageH)
          return (
            <Group key={ann.id}>
              {geom.strokes.map((pt: any, i: number) => (
                <Circle key={i} x={pt.x * stageW} y={pt.y * stageH} radius={r} fill={color} />
              ))}
            </Group>
          )
        }
        return null
      })}
    </>
  )
}

export function AnnotationCanvas({ imageId, width, height }: Props) {
  const { activeTool, activeClass, currentPolygon, currentBrush, brushRadius, saving,
    addPolygonPoint, addBrushStroke, clearCurrentDraw, savePolygon, saveBrush } = useAnnotationStore()
  const [img] = useImage(imagesApi.fileUrl(imageId))
  const [isDrawing, setIsDrawing] = useState(false)
  const stageRef = useRef<any>(null)

  const getPos = useCallback(() => {
    const pos = stageRef.current?.getPointerPosition()
    if (!pos) return { x: 0, y: 0 }
    return { x: pos.x / width, y: pos.y / height }
  }, [width, height])

  const handleMouseDown = useCallback(() => {
    if (activeTool === 'brush') { setIsDrawing(true); addBrushStroke(getPos()) }
    else addPolygonPoint(getPos())
  }, [activeTool, addBrushStroke, addPolygonPoint, getPos])

  const handleMouseMove = useCallback(() => {
    if (activeTool === 'brush' && isDrawing) addBrushStroke(getPos())
  }, [activeTool, isDrawing, addBrushStroke, getPos])

  const handleMouseUp = useCallback(() => {
    if (activeTool === 'brush') setIsDrawing(false)
  }, [activeTool])

  const handleDblClick = useCallback(async () => {
    if (activeTool === 'polygon') await savePolygon(imageId)
  }, [activeTool, savePolygon, imageId])

  const activeColorMap: Record<string, string> = {
    glioma: '#ef4444', meningioma: '#f97316', pituitary: '#2563eb',
    notumor: '#22c55e', gradcam_focus: '#f59e0b',
  }
  const activeFillMap: Record<string, string> = {
    glioma: 'rgba(239,68,68,0.4)', meningioma: 'rgba(249,115,22,0.4)',
    pituitary: 'rgba(37,99,235,0.4)', notumor: 'rgba(34,197,94,0.4)',
    gradcam_focus: 'rgba(245,158,11,0.4)',
  }
  const activeColor = activeColorMap[activeClass] ?? '#9ca3af'
  const activeFill  = activeFillMap[activeClass]  ?? 'rgba(156,163,175,0.4)'
  const polyPts = currentPolygon.flatMap(p => [p.x * width, p.y * height])

  return (
    <div className="space-y-2">
      <ToolBar />

      <div className="relative rounded-xl overflow-hidden border border-gray-700">
        <Stage ref={stageRef} width={width} height={height}
          onMouseDown={handleMouseDown} onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp} onDblClick={handleDblClick}
          style={{ cursor: activeTool === 'polygon' ? 'crosshair' : 'cell', display: 'block' }}>
          <Layer>
            {img && <KonvaImage image={img} width={width} height={height} />}
            <AnnotationOverlay stageW={width} stageH={height} />
            {/* In-progress polygon — dashed to distinguish from saved */}
            {currentPolygon.length > 0 && (
              <>
                <Line points={polyPts} stroke={activeColor} strokeWidth={2} dash={[6, 3]} />
                {currentPolygon.map((pt, i) => (
                  <Circle key={i} x={pt.x * width} y={pt.y * height} radius={4}
                    fill={activeColor} opacity={0.8} />
                ))}
              </>
            )}
            {/* In-progress brush */}
            {currentBrush.map((pt, i) => (
              <Circle key={i} x={pt.x * width} y={pt.y * height}
                radius={brushRadius * Math.min(width, height)} fill={activeFill} />
            ))}
          </Layer>
        </Stage>

        {/* Hint overlay */}
        {activeTool === 'polygon' && currentPolygon.length === 0 && (
          <div className="absolute bottom-2 left-0 right-0 flex justify-center pointer-events-none">
            <span className="text-xs bg-black/60 text-gray-300 px-2 py-0.5 rounded">
              Click to add points · Double-click to close
            </span>
          </div>
        )}
        {activeTool === 'brush' && currentBrush.length === 0 && (
          <div className="absolute bottom-2 left-0 right-0 flex justify-center pointer-events-none">
            <span className="text-xs bg-black/60 text-gray-300 px-2 py-0.5 rounded">
              Click and drag to paint
            </span>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        {activeTool === 'brush' && currentBrush.length > 0 && (
          <button onClick={() => saveBrush(imageId)} disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-brand-600 text-white text-sm rounded-lg hover:bg-brand-700 disabled:opacity-50 transition-colors">
            {saving && <LoadingSpinner size="sm" />} Save Brush
          </button>
        )}
        {activeTool === 'polygon' && currentPolygon.length >= 3 && (
          <button onClick={() => savePolygon(imageId)} disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-brand-600 text-white text-sm rounded-lg hover:bg-brand-700 disabled:opacity-50 transition-colors">
            {saving && <LoadingSpinner size="sm" />} Close & Save
          </button>
        )}
        {(currentPolygon.length > 0 || currentBrush.length > 0) && (
          <button onClick={clearCurrentDraw}
            className="px-3 py-1.5 bg-gray-700 text-gray-300 text-sm rounded-lg hover:bg-gray-600 transition-colors">
            Cancel
          </button>
        )}
      </div>

      {/* Saved annotations */}
      <div className="mt-1">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
          Saved Annotations
        </div>
        <AnnotationList />
      </div>
    </div>
  )
}
