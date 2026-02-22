import { create } from 'zustand'
import type { Annotation, AnnotationCreate } from '../types'
import { annotationsApi } from '../api/annotations'
import { toast } from '../components/shared/Toast'

const ANNOTATION_SYNC_KEY = 'cancerscope:annotation-updated'

function broadcastAnnotationUpdate(imageId: string) {
  try {
    localStorage.setItem(
      ANNOTATION_SYNC_KEY,
      JSON.stringify({ imageId, ts: Date.now() })
    )
  } catch {
    // no-op in restricted environments
  }
}

export type Tool = 'polygon' | 'brush'
export type LabelClass = 'glioma' | 'meningioma' | 'notumor' | 'pituitary' | 'gradcam_focus'

interface Point { x: number; y: number }

interface AnnotationStore {
  annotations: Annotation[]
  activeTool: Tool
  activeClass: LabelClass
  brushRadius: number
  currentPolygon: Point[]
  currentBrush: Point[]
  saving: boolean

  fetchAnnotations: (imageId: string) => Promise<void>
  setTool: (tool: Tool) => void
  setClass: (cls: LabelClass) => void
  setBrushRadius: (r: number) => void
  addPolygonPoint: (pt: Point) => void
  addBrushStroke: (pt: Point) => void
  clearCurrentDraw: () => void
  savePolygon: (imageId: string) => Promise<void>
  saveBrush: (imageId: string) => Promise<void>
  deleteAnnotation: (id: number) => Promise<void>
}

export const useAnnotationStore = create<AnnotationStore>((set, get) => ({
  annotations: [],
  activeTool: 'polygon',
  activeClass: 'glioma',
  brushRadius: 0.02,
  currentPolygon: [],
  currentBrush: [],
  saving: false,

  fetchAnnotations: async (imageId) => {
    try {
      const data = await annotationsApi.list(imageId)
      set({ annotations: data })
    } catch (e: any) {
      toast.error(`Failed to load annotations: ${e.message}`)
    }
  },

  setTool: (tool) => set({ activeTool: tool, currentPolygon: [], currentBrush: [] }),
  setClass: (cls) => set({ activeClass: cls }),
  setBrushRadius: (r) => set({ brushRadius: r }),

  addPolygonPoint: (pt) => set(s => ({ currentPolygon: [...s.currentPolygon, pt] })),
  addBrushStroke: (pt) => set(s => ({ currentBrush: [...s.currentBrush, pt] })),
  clearCurrentDraw: () => set({ currentPolygon: [], currentBrush: [] }),

  savePolygon: async (imageId) => {
    const { currentPolygon, activeClass } = get()
    if (currentPolygon.length < 3) return
    set({ saving: true })
    try {
      const body: AnnotationCreate = {
        image_id: imageId, label_class: activeClass, geometry_type: 'polygon',
        geometry_json: JSON.stringify({ points: currentPolygon }),
      }
      const ann = await annotationsApi.create(body)
      set(s => ({ annotations: [...s.annotations, ann], currentPolygon: [], saving: false }))
      broadcastAnnotationUpdate(imageId)
      toast.success('Annotation saved')
    } catch (e: any) {
      set({ saving: false })
      toast.error(`Failed to save annotation: ${e.message}`)
    }
  },

  saveBrush: async (imageId) => {
    const { currentBrush, activeClass, brushRadius } = get()
    if (currentBrush.length === 0) return
    set({ saving: true })
    try {
      const body: AnnotationCreate = {
        image_id: imageId, label_class: activeClass, geometry_type: 'brush',
        geometry_json: JSON.stringify({ strokes: currentBrush, radius: brushRadius }),
      }
      const ann = await annotationsApi.create(body)
      set(s => ({ annotations: [...s.annotations, ann], currentBrush: [], saving: false }))
      broadcastAnnotationUpdate(imageId)
      toast.success('Annotation saved')
    } catch (e: any) {
      set({ saving: false })
      toast.error(`Failed to save annotation: ${e.message}`)
    }
  },

  deleteAnnotation: async (id) => {
    try {
      const imageId = get().annotations.find(a => a.id === id)?.image_id
      await annotationsApi.delete(id)
      set(s => ({ annotations: s.annotations.filter(a => a.id !== id) }))
      if (imageId) broadcastAnnotationUpdate(imageId)
      toast.success('Annotation deleted')
    } catch (e: any) {
      toast.error(`Failed to delete annotation: ${e.message}`)
    }
  },
}))
