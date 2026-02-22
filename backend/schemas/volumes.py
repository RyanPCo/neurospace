from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Gradcam3DRunRequest(BaseModel):
    volume_path: str
    segmentation_path: str | None = None
    model_path: str = "/Users/rohan/CancerScope/models/best_model.pt"

    axis: Literal["axial", "coronal", "sagittal"] = "axial"
    slice_index: int = 80
    out_dir: str = "/Users/rohan/CancerScope/data/gradcam_outputs"

    save_every: int = 1
    brain_threshold: float = 0.08
    min_brain_fraction: float = 0.02
    cam_percentile: float = 96.0
    global_percentile: float = 99.0

    gradcam_threshold: float = 0.75
    launch_viewer: bool = True


class Gradcam3DRunResponse(BaseModel):
    message: str
    preview_png: str
    heatmap_3d: str
    overlay_4d: str
    volume_path: str
    axis: Literal["axial", "coronal", "sagittal"]
    num_slices: int
    default_slice_index: int
    viewer_launched: bool
    apply_command: str
    render_command: str | None = None
