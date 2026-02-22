import { useState } from 'react'
import { volumesApi } from '../api/volumes'
import type { Gradcam3DRunResponse } from '../types'

const defaults = {
  volume_path:
    '/Users/rohan/CancerScope/data/raw/brats20_sample/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii',
  segmentation_path:
    '/Users/rohan/CancerScope/data/raw/brats20_sample/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii',
  model_path: '/Users/rohan/CancerScope/models/best_model.pt',
  out_dir: '/Users/rohan/CancerScope/data/gradcam_outputs',
}

export function VolumeViewerPage() {
  const [volumePath, setVolumePath] = useState(defaults.volume_path)
  const [segPath, setSegPath] = useState(defaults.segmentation_path)
  const [modelPath, setModelPath] = useState(defaults.model_path)
  const [outDir, setOutDir] = useState(defaults.out_dir)

  const [axis, setAxis] = useState<'axial' | 'coronal' | 'sagittal'>('axial')
  const [sliceIndex, setSliceIndex] = useState(80)
  const [threshold, setThreshold] = useState(0.75)
  const [launchViewer, setLaunchViewer] = useState(true)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Gradcam3DRunResponse | null>(null)
  const [previewSlice, setPreviewSlice] = useState(80)
  const [refreshKey, setRefreshKey] = useState(0)

  const onRun = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await volumesApi.runGradcam3d({
        volume_path: volumePath,
        segmentation_path: segPath || null,
        model_path: modelPath,
        axis,
        slice_index: sliceIndex,
        out_dir: outDir,
        gradcam_threshold: threshold,
        launch_viewer: launchViewer,
      })      
      setResult(res)
      setPreviewSlice(res.default_slice_index)
      setRefreshKey(k => k + 1)
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || 'Failed to run 3D Grad-CAM pipeline'
      setError(String(msg))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      <div>
        <h1 className="font-bold text-lg text-gray-100">3D Grad-CAM Viewer</h1>
        <p className="text-sm text-gray-400 mt-1">
          Generate 3D Grad-CAM heatmap and optionally launch the interactive viewer.
        </p>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-4">
        <Field label="Volume Path" value={volumePath} onChange={setVolumePath} />
        <Field label="Segmentation Path (optional)" value={segPath} onChange={setSegPath} />
        <Field label="Model Path" value={modelPath} onChange={setModelPath} />
        <Field label="Output Directory" value={outDir} onChange={setOutDir} />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <div className="text-xs text-gray-400 mb-1">Axis</div>
            <select
              value={axis}
              onChange={e => setAxis(e.target.value as any)}
              className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm"
            >
              <option value="axial">axial</option>
              <option value="coronal">coronal</option>
              <option value="sagittal">sagittal</option>
            </select>
          </div>

          <NumberField label="Slice Index" value={sliceIndex} onChange={setSliceIndex} step={1} />
          <NumberField label="GradCAM Threshold" value={threshold} onChange={setThreshold} step={0.05} />
        </div>

        <label className="inline-flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={launchViewer}
            onChange={e => setLaunchViewer(e.target.checked)}
          />
          Launch interactive viewer window
        </label>

        <div>
          <button
            onClick={onRun}
            disabled={loading || !volumePath.trim()}
            className="px-4 py-2 rounded-md bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white text-sm font-medium"
          >
            {loading ? 'Running...' : 'Run 3D Grad-CAM'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-950 border border-red-900 text-red-300 rounded-xl p-3 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-2 text-sm">
          <div className="text-green-300">{result.message}</div>
          <div className="pt-2">
            <div className="text-xs text-gray-400 mb-2">Rendered In UI (slice overlay preview)</div>
            <img
              src={slicePreviewUrl(result, previewSlice, refreshKey)}
              alt="Grad-CAM slice preview"
              className="w-full max-w-3xl border border-gray-700 rounded-lg"
            />
            <div className="mt-3">
              <input
                type="range"
                min={0}
                max={Math.max(0, result.num_slices - 1)}
                step={1}
                value={previewSlice}
                onChange={e => setPreviewSlice(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-400 mt-1">
                Slice: {previewSlice} / {Math.max(0, result.num_slices - 1)} ({result.axis})
              </div>
            </div>
          </div>

          <div className="pt-2">
            <div className="text-xs text-gray-400 mb-1">Generated Preview PNG</div>
            <img
              src={workspaceFileUrl(result.preview_png)}
              alt="Generated preview png"
              className="w-full max-w-3xl border border-gray-700 rounded-lg"
            />
          </div>

          <PathRow label="Preview PNG" value={result.preview_png} />
          <PathRow label="Heatmap 3D" value={result.heatmap_3d} />
          <PathRow label="Overlay 4D" value={result.overlay_4d} />
          <PathRow label="Apply Command" value={result.apply_command} />
          {result.render_command && <PathRow label="Render Command" value={result.render_command} />}
        </div>
      )}
    </div>
  )
}

function workspaceFileUrl(path: string): string {
  return `/api/volumes/file?path=${encodeURIComponent(path)}`
}

function slicePreviewUrl(result: Gradcam3DRunResponse, sliceIndex: number, key: number): string {
  const params = new URLSearchParams({
    volume_path: result.volume_path,
    heatmap_path: result.heatmap_3d,
    axis: result.axis,
    slice_index: String(sliceIndex),
    alpha: '0.45',
    _k: String(key), // cache busting
  })
  return `/api/volumes/heatmap-preview?${params.toString()}`
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm"
      />
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  step,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  step: number
}) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <input
        type="number"
        value={value}
        step={step}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm"
      />
    </div>
  )
}

function PathRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[140px,1fr] gap-3 items-start">
      <div className="text-gray-400">{label}</div>
      <code className="text-xs text-gray-200 break-all">{value}</code>
    </div>
  )
}
