import { create } from 'zustand'
import type { Annotation, AnnotationCreate } from '../types'
import { annotationsApi } from '../api/annotations'

export type Tool = 'polygon' | 'brush' | 'eraser'
export type LabelClass = 'malignant' | 'benign'

interface Point { x: number; y: number }

interface AnnotationStore {
  annotations: Annotation[]
  activeTool: Tool
  activeClass: LabelClass
  brushRadius: number
  // In-progress drawing state
  currentPolygon: Point[]
  currentBrush: Point[]
  loading: boolean
  error: string | null

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
  activeClass: 'malignant',
  brushRadius: 0.02,
  currentPolygon: [],
  currentBrush: [],
  loading: false,
  error: null,

  fetchAnnotations: async (imageId) => {
    set({ loading: true })
    try {
      const data = await annotationsApi.list(imageId)
      set({ annotations: data, loading: false })
    } catch (e: any) {
      set({ error: e.message, loading: false })
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
    const body: AnnotationCreate = {
      image_id: imageId,
      label_class: activeClass,
      geometry_type: 'polygon',
      geometry_json: JSON.stringify({ points: currentPolygon }),
    }
    const ann = await annotationsApi.create(body)
    set(s => ({ annotations: [...s.annotations, ann], currentPolygon: [] }))
  },

  saveBrush: async (imageId) => {
    const { currentBrush, activeClass, brushRadius } = get()
    if (currentBrush.length === 0) return
    const body: AnnotationCreate = {
      image_id: imageId,
      label_class: activeClass,
      geometry_type: 'brush',
      geometry_json: JSON.stringify({ strokes: currentBrush, radius: brushRadius }),
    }
    const ann = await annotationsApi.create(body)
    set(s => ({ annotations: [...s.annotations, ann], currentBrush: [] }))
  },

  deleteAnnotation: async (id) => {
    await annotationsApi.delete(id)
    set(s => ({ annotations: s.annotations.filter(a => a.id !== id) }))
  },
}))
