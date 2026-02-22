// ─── Images ────────────────────────────────────────────────────────────────────

export interface ImageSummary {
  id: string
  filename: string
  ground_truth: string
  split: string
  width: number | null
  height: number | null
  predicted_class: string | null
  confidence: number | null
  annotation_count: number
}

export interface ImageDetail extends ImageSummary {
  file_path: string
  model_version: string | null
  class_probs: Record<string, number> | null
}

export interface GradCAMResult {
  image_id: string
  heatmap_b64: string
  overlay_b64: string
  top_kernel_indices: number[]
  predicted_class: string
  confidence: number
}

export interface AnnotationRetrainRequest {
  steps?: number
  lr?: number
  alpha?: number
  beta?: number
  gamma?: number
  preserve_ce_weight?: number
  target_class?: string
  model_in?: string
  model_out?: string
}

export interface AnnotationRetrainResult {
  image_id: string
  message: string
  roi_mask_path: string
  neg_mask_path: string
  model_in: string
  model_out: string
  command: string
  log_tail: string
}

export interface PaginatedImages {
  items: ImageSummary[]
  total: number
  page: number
  page_size: number
}

// ─── Kernels ───────────────────────────────────────────────────────────────────

export interface KernelSummary {
  id: string
  layer_name: string
  filter_index: number
  importance_score: number
  assigned_class: string | null
  is_deleted: boolean
  doctor_notes: string | null
  last_scored_at: string | null
}

export interface TopActivatingImage {
  image_id: string
  max_activation: number
  activation_map_b64: string
}

export interface KernelActivations {
  kernel_id: string
  top_images: TopActivatingImage[]
}

export interface PaginatedKernels {
  items: KernelSummary[]
  total: number
  page: number
  page_size: number
}

// ─── Annotations ───────────────────────────────────────────────────────────────

export interface AnnotationCreate {
  image_id: string
  label_class: string
  geometry_type: 'polygon' | 'brush'
  geometry_json: string
  notes?: string
}

export interface Annotation {
  id: number
  image_id: string
  label_class: string
  geometry_type: string
  geometry_json: string
  notes: string | null
  is_active: boolean
  created_at: string
}

// ─── Predictions ───────────────────────────────────────────────────────────────

export interface Prediction {
  id: number
  image_id: string
  model_version: string
  predicted_class: string
  confidence: number
  class_probs_json: string | null
  subtype_predicted: string | null
  created_at: string
}

// ─── Training ──────────────────────────────────────────────────────────────────

export interface TrainingConfig {
  num_epochs: number
  learning_rate: number
  weight_decay: number
  batch_size: number
  spatial_loss_weight: number
}

export interface TrainingRun {
  id: string
  status: string
  config_json: string | null
  start_time: string | null
  end_time: string | null
  final_train_loss: number | null
  final_val_loss: number | null
  final_train_acc: number | null
  final_val_acc: number | null
  model_version: string | null
  error_message: string | null
}

export interface TrainingEpoch {
  id: number
  run_id: string
  epoch: number
  train_loss: number | null
  val_loss: number | null
  train_acc: number | null
  val_acc: number | null
  duration_sec: number | null
}

export interface TrainingStatus {
  run_id: string | null
  status: string
  current_epoch: number | null
  total_epochs: number | null
  latest_metrics: Record<string, number> | null
}

// ─── WebSocket messages ────────────────────────────────────────────────────────

export interface WSMessage {
  type: 'epoch_start' | 'batch' | 'epoch_end' | 'completed' | 'error' | 'status'
  run_id?: string
  epoch?: number
  batch?: number | null
  total_batches?: number | null
  train_loss?: number | null
  val_loss?: number | null
  train_acc?: number | null
  val_acc?: number | null
  message?: string
  status?: string
}

// ─── 3D Grad-CAM ──────────────────────────────────────────────────────────────

export interface Gradcam3DRunRequest {
  volume_path: string
  segmentation_path?: string | null
  model_path?: string
  axis?: 'axial' | 'coronal' | 'sagittal'
  slice_index?: number
  out_dir?: string
  save_every?: number
  brain_threshold?: number
  min_brain_fraction?: number
  cam_percentile?: number
  global_percentile?: number
  gradcam_threshold?: number
  launch_viewer?: boolean
}

export interface Gradcam3DRunResponse {
  message: string
  preview_png: string
  heatmap_3d: string
  overlay_4d: string
  volume_path: string
  axis: 'axial' | 'coronal' | 'sagittal'
  num_slices: number
  default_slice_index: number
  viewer_launched: boolean
  apply_command: string
  render_command: string | null
}
