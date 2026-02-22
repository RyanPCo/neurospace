import shlex
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from schemas.volumes import Gradcam3DRunRequest, Gradcam3DRunResponse

router = APIRouter(prefix="/api/volumes", tags=["volumes"])
WORKSPACE_ROOT = Path("/Users/rohan/CancerScope").resolve()


def _safe_resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    try:
        p.relative_to(WORKSPACE_ROOT)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Path must be inside workspace") from e
    return p


def _get_num_slices(shape: tuple[int, int, int], axis: str) -> int:
    if axis == "axial":
        return int(shape[2])
    if axis == "coronal":
        return int(shape[1])
    if axis == "sagittal":
        return int(shape[0])
    raise HTTPException(status_code=400, detail=f"Unsupported axis: {axis}")


def _extract_slice(arr: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "axial":
        return arr[:, :, idx]
    if axis == "coronal":
        return arr[:, idx, :]
    if axis == "sagittal":
        return arr[idx, :, :]
    raise HTTPException(status_code=400, detail=f"Unsupported axis: {axis}")


def _apply_jet(gray: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


@router.post("/gradcam3d/run", response_model=Gradcam3DRunResponse)
def run_gradcam3d(payload: Gradcam3DRunRequest):
    volume_path = _safe_resolve(payload.volume_path)
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail=f"Volume not found: {volume_path}")

    model_path = _safe_resolve(payload.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    seg_path = None
    if payload.segmentation_path:
        seg_path = _safe_resolve(payload.segmentation_path)
        if not seg_path.exists():
            raise HTTPException(status_code=404, detail=f"Segmentation not found: {seg_path}")

    out_dir = _safe_resolve(payload.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    apply_script = scripts_dir / "apply_gradcam_3d.py"
    render_script = scripts_dir / "render_mri_3d.py"

    if not apply_script.exists() or not render_script.exists():
        raise HTTPException(status_code=500, detail="Required scripts not found in backend/scripts")

    apply_cmd = [
        sys.executable,
        str(apply_script),
        "--volume",
        str(volume_path),
        "--model",
        str(model_path),
        "--axis",
        payload.axis,
        "--slice-index",
        str(payload.slice_index),
        "--out-dir",
        str(out_dir),
        "--save-every",
        str(payload.save_every),
        "--brain-threshold",
        str(payload.brain_threshold),
        "--min-brain-fraction",
        str(payload.min_brain_fraction),
        "--cam-percentile",
        str(payload.cam_percentile),
        "--global-percentile",
        str(payload.global_percentile),
    ]

    try:
        subprocess.run(apply_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or str(e)).strip()
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {detail}")

    heatmap = out_dir / f"{volume_path.stem}_{payload.axis}_gradcam_heatmap_3d.nii.gz"
    overlay = out_dir / f"{volume_path.stem}_{payload.axis}_gradcam_overlay_rgb_4d.nii.gz"
    preview = out_dir / f"{volume_path.stem}_{payload.axis}_slice{payload.slice_index:03d}_gradcam.png"

    if not heatmap.exists():
        raise HTTPException(status_code=500, detail="Grad-CAM heatmap file was not generated")

    vol_shape = nib.load(str(volume_path)).shape
    if len(vol_shape) < 3:
        raise HTTPException(status_code=500, detail="Input volume is not 3D")
    num_slices = _get_num_slices((int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2])), payload.axis)
    default_slice_index = max(0, min(payload.slice_index, num_slices - 1))

    render_cmd_list = None
    if payload.launch_viewer:
        render_cmd_list = [
            sys.executable,
            str(render_script),
            "--volume",
            str(volume_path),
            "--gradcam-heatmap",
            str(heatmap),
            "--gradcam-threshold",
            str(payload.gradcam_threshold),
        ]
        if seg_path is not None:
            render_cmd_list.extend(["--segmentation", str(seg_path)])

        try:
            # Launch detached so HTTP can return immediately.
            subprocess.Popen(render_cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to launch viewer: {e}")

    return Gradcam3DRunResponse(
        message="Grad-CAM 3D generated",
        preview_png=str(preview),
        heatmap_3d=str(heatmap),
        overlay_4d=str(overlay),
        volume_path=str(volume_path),
        axis=payload.axis,
        num_slices=num_slices,
        default_slice_index=default_slice_index,
        viewer_launched=payload.launch_viewer,
        apply_command=" ".join(shlex.quote(x) for x in apply_cmd),
        render_command=(" ".join(shlex.quote(x) for x in render_cmd_list) if render_cmd_list else None),
    )


@router.get("/file")
def get_workspace_file(path: str):
    file_path = _safe_resolve(path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))


@router.get("/heatmap-preview")
def heatmap_preview(
    volume_path: str,
    heatmap_path: str,
    axis: str = "axial",
    slice_index: int = 80,
    alpha: float = 0.45,
):
    vol_path = _safe_resolve(volume_path)
    hm_path = _safe_resolve(heatmap_path)

    if not vol_path.exists() or not hm_path.exists():
        raise HTTPException(status_code=404, detail="Volume/heatmap file not found")
    if not (0.0 <= alpha <= 1.0):
        raise HTTPException(status_code=400, detail="alpha must be in [0,1]")

    vol = np.asarray(nib.load(str(vol_path)).get_fdata(dtype=np.float32))
    hm = np.asarray(nib.load(str(hm_path)).get_fdata(dtype=np.float32))
    if vol.shape[:3] != hm.shape[:3]:
        raise HTTPException(status_code=400, detail="Volume and heatmap shape mismatch")

    n = _get_num_slices((vol.shape[0], vol.shape[1], vol.shape[2]), axis)
    if slice_index < 0 or slice_index >= n:
        raise HTTPException(status_code=400, detail=f"slice_index must be in [0,{n-1}]")

    vol_sl = _extract_slice(vol, axis, slice_index)
    hm_sl = np.clip(_extract_slice(hm, axis, slice_index), 0.0, 1.0)

    vmin, vmax = float(np.nanmin(vol_sl)), float(np.nanmax(vol_sl))
    if vmax > vmin:
        vol_norm = (vol_sl - vmin) / (vmax - vmin)
    else:
        vol_norm = np.zeros_like(vol_sl, dtype=np.float32)

    gray = (np.clip(vol_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    color = _apply_jet(hm_sl)
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    blended = np.clip((1.0 - alpha) * base + alpha * color.astype(np.float32), 0, 255).astype(np.uint8)

    bio = BytesIO()
    Image.fromarray(blended).save(bio, format="PNG")
    bio.seek(0)
    return StreamingResponse(bio, media_type="image/png")
