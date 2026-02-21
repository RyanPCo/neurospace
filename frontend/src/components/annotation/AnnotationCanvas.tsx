import { useRef, useState, useCallback } from 'react'
import { Stage, Layer, Image as KonvaImage, Line, Circle, Group } from 'react-konva'
import useImage from 'use-image'
import { useAnnotationStore } from '../../store/annotationStore'
import { ToolBar } from './ToolBar'
import { AnnotationList } from './AnnotationList'
import { imagesApi } from '../../api/images'

interface Props {
  imageId: string
  width: number
  height: number
}

function AnnotationOverlay({
  stageW,
  stageH,
}: {
  stageW: number
  stageH: number
}) {
  const { annotations } = useAnnotationStore()

  return (
    <>
      {annotations.map(ann => {
        if (ann.geometry_type === 'polygon') {
          const geom = JSON.parse(ann.geometry_json)
          const pts: number[] = geom.points.flatMap((p: { x: number; y: number }) => [
            p.x * stageW,
            p.y * stageH,
          ])
          const color = ann.label_class === 'malignant' ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.4)'
          const stroke = ann.label_class === 'malignant' ? '#ef4444' : '#22c55e'
          return (
            <Line
              key={ann.id}
              points={pts}
              closed
              fill={color}
              stroke={stroke}
              strokeWidth={2}
            />
          )
        } else if (ann.geometry_type === 'brush') {
          const geom = JSON.parse(ann.geometry_json)
          const radius = geom.radius * Math.min(stageW, stageH)
          const color = ann.label_class === 'malignant' ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.4)'
          return (
            <Group key={ann.id}>
              {geom.strokes.map((pt: { x: number; y: number }, i: number) => (
                <Circle
                  key={i}
                  x={pt.x * stageW}
                  y={pt.y * stageH}
                  radius={radius}
                  fill={color}
                />
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
  const { activeTool, activeClass, currentPolygon, currentBrush, addPolygonPoint, addBrushStroke, clearCurrentDraw, savePolygon, saveBrush } = useAnnotationStore()
  const [img] = useImage(imagesApi.fileUrl(imageId))
  const [isDrawing, setIsDrawing] = useState(false)
  const stageRef = useRef<any>(null)

  const stageW = width
  const stageH = height

  const getPos = useCallback(() => {
    const stage = stageRef.current
    if (!stage) return { x: 0, y: 0 }
    const pos = stage.getPointerPosition()
    return { x: pos.x / stageW, y: pos.y / stageH }
  }, [stageW, stageH])

  const handleMouseDown = useCallback(() => {
    if (activeTool === 'brush') {
      setIsDrawing(true)
      addBrushStroke(getPos())
    } else if (activeTool === 'polygon') {
      addPolygonPoint(getPos())
    }
  }, [activeTool, addBrushStroke, addPolygonPoint, getPos])

  const handleMouseMove = useCallback(() => {
    if (activeTool === 'brush' && isDrawing) {
      addBrushStroke(getPos())
    }
  }, [activeTool, isDrawing, addBrushStroke, getPos])

  const handleMouseUp = useCallback(() => {
    if (activeTool === 'brush' && isDrawing) {
      setIsDrawing(false)
    }
  }, [activeTool, isDrawing])

  const handleDoubleClick = useCallback(async () => {
    if (activeTool === 'polygon') {
      await savePolygon(imageId)
    }
  }, [activeTool, savePolygon, imageId])

  const handleSaveBrush = useCallback(async () => {
    await saveBrush(imageId)
  }, [saveBrush, imageId])

  // Current polygon preview
  const polyPts = currentPolygon.flatMap(p => [p.x * stageW, p.y * stageH])
  // Current brush preview
  const brushColor = activeClass === 'malignant' ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.5)'

  return (
    <div className="space-y-2">
      <ToolBar />

      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <Stage
          ref={stageRef}
          width={stageW}
          height={stageH}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onDblClick={handleDoubleClick}
          style={{ cursor: activeTool === 'polygon' ? 'crosshair' : 'cell' }}
        >
          <Layer>
            {img && <KonvaImage image={img} width={stageW} height={stageH} />}
            <AnnotationOverlay stageW={stageW} stageH={stageH} />
            {/* In-progress polygon */}
            {currentPolygon.length > 0 && (
              <Line
                points={polyPts}
                stroke={activeClass === 'malignant' ? '#ef4444' : '#22c55e'}
                strokeWidth={2}
                dash={[6, 3]}
              />
            )}
            {/* In-progress brush strokes */}
            {currentBrush.map((pt, i) => (
              <Circle
                key={i}
                x={pt.x * stageW}
                y={pt.y * stageH}
                radius={0.02 * Math.min(stageW, stageH)}
                fill={brushColor}
              />
            ))}
          </Layer>
        </Stage>
      </div>

      <div className="flex gap-2">
        {activeTool === 'brush' && currentBrush.length > 0 && (
          <button
            onClick={handleSaveBrush}
            className="px-3 py-1.5 bg-brand-600 text-white text-sm rounded hover:bg-brand-700 transition-colors"
          >
            Save Brush
          </button>
        )}
        {activeTool === 'polygon' && currentPolygon.length >= 3 && (
          <button
            onClick={() => savePolygon(imageId)}
            className="px-3 py-1.5 bg-brand-600 text-white text-sm rounded hover:bg-brand-700 transition-colors"
          >
            Close Polygon
          </button>
        )}
        {(currentPolygon.length > 0 || currentBrush.length > 0) && (
          <button
            onClick={clearCurrentDraw}
            className="px-3 py-1.5 bg-gray-700 text-gray-200 text-sm rounded hover:bg-gray-600 transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      <div className="mt-2">
        <div className="text-xs font-medium text-gray-500 mb-1.5">Saved Annotations</div>
        <AnnotationList />
      </div>
    </div>
  )
}
