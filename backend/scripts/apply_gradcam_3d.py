#!/usr/bin/env python3
"""Apply a trained 2D CNN Grad-CAM to MRI slices and reconstruct 3D outputs.

Outputs:
1) A single-slice PNG preview with Grad-CAM coloring
2) A full 3D Grad-CAM heatmap NIfTI and RGB overlay NIfTI built from all slices

Example:
  python backend/scripts/apply_gradcam_3d.py \
    --volume /path/to/BraTS20_Training_001_flair.nii \
    --model /Users/rohan/CancerScope/models/best_model.pt \
    --axis axial \
    --slice-index 80 \
    --out-dir /Users/rohan/CancerScope/data/gradcam_outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow running as a standalone script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, TUMOR_CLASSES
from core.dataset.transforms import inference_transforms
from core.model.cnn import load_model


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not np.any(finite):
        raise ValueError("Volume has no finite values.")

    valid = volume[finite]
    lo, hi = np.percentile(valid, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1e-6

    out = np.zeros_like(volume, dtype=np.float32)
    out[finite] = np.clip((volume[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def apply_jet(gray: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0, 1)
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def alpha_blend(gray: np.ndarray, color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    over = color.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * over
    return np.clip(out, 0, 255).astype(np.uint8)


def get_slice(volume: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "axial":
        return volume[:, :, idx]
    if axis == "coronal":
        return volume[:, idx, :]
    if axis == "sagittal":
        return volume[idx, :, :]
    raise ValueError(f"Unsupported axis: {axis}")


def put_slice(target: np.ndarray, axis: str, idx: int, arr: np.ndarray) -> None:
    if axis == "axial":
        target[:, :, idx] = arr
        return
    if axis == "coronal":
        target[:, idx, :] = arr
        return
    if axis == "sagittal":
        target[idx, :, :] = arr
        return
    raise ValueError(f"Unsupported axis: {axis}")


def put_slice_rgb(target: np.ndarray, axis: str, idx: int, arr: np.ndarray) -> None:
    if axis == "axial":
        target[:, :, idx, :] = arr
        return
    if axis == "coronal":
        target[:, idx, :, :] = arr
        return
    if axis == "sagittal":
        target[idx, :, :, :] = arr
        return
    raise ValueError(f"Unsupported axis: {axis}")


def num_slices(shape: tuple[int, int, int], axis: str) -> int:
    if axis == "axial":
        return shape[2]
    if axis == "coronal":
        return shape[1]
    if axis == "sagittal":
        return shape[0]
    raise ValueError(f"Unsupported axis: {axis}")


class SliceGradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target = self.model.get_layer_for_gradcam()
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, image_2d_norm: np.ndarray, device: str) -> tuple[np.ndarray, int, float]:
        img_u8 = np.clip(image_2d_norm * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")

        # input -> (1,1,256,256)
        x = inference_transforms(pil).unsqueeze(0).to(device)
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1)
        cls = int(torch.argmax(logits, dim=1).item())
        conf = float(probs[0, cls].item())

        score = logits[0, cls]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not produce activations/gradients")

        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.activations).sum(dim=1, keepdim=True))

        # Resize CAM back to original slice spatial size
        target_hw = image_2d_norm.shape
        cam_up = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
        cam_np = cam_up[0, 0].detach().cpu().numpy()

        cmin, cmax = float(cam_np.min()), float(cam_np.max())
        if cmax > cmin:
            cam_np = (cam_np - cmin) / (cmax - cmin)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        return cam_np.astype(np.float32), cls, conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Grad-CAM to a 3D MRI via 2D slices")
    parser.add_argument("--volume", type=Path, required=True, help="Path to 3D MRI NIfTI volume")
    parser.add_argument("--model", type=Path, default=Path("/Users/rohan/CancerScope/models/best_model.pt"))
    parser.add_argument("--axis", type=str, default="axial", choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--slice-index", type=int, default=-1, help="Index for preview PNG; -1 uses middle slice")
    parser.add_argument("--out-dir", type=Path, default=Path("/Users/rohan/CancerScope/data/gradcam_outputs"))
    parser.add_argument("--save-every", type=int, default=1, help="Process every Nth slice for faster runs")
    parser.add_argument("--device", type=str, default=None, help="Override device (mps/cuda/cpu)")
    parser.add_argument(
        "--brain-threshold",
        type=float,
        default=0.08,
        help="Normalized intensity threshold for brain mask (0..1).",
    )
    parser.add_argument(
        "--min-brain-fraction",
        type=float,
        default=0.02,
        help="Skip slice if brain-mask area fraction is below this value.",
    )
    parser.add_argument(
        "--cam-percentile",
        type=float,
        default=95.0,
        help="Per-slice percentile threshold for CAM inside brain mask.",
    )
    parser.add_argument(
        "--global-percentile",
        type=float,
        default=98.0,
        help="Final 3D percentile threshold for CAM (helps remove slab artifacts).",
    )
    return parser.parse_args()


def disable_inplace_relu(model: torch.nn.Module) -> None:
    # Grad-CAM backward hooks can fail with in-place ReLU ops.
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU) and module.inplace:
            module.inplace = False


def main() -> None:
    args = parse_args()

    volume_path = args.volume.expanduser().resolve()
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume not found: {volume_path}")

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if not (0.0 <= args.brain_threshold <= 1.0):
        raise ValueError("--brain-threshold must be in [0,1]")
    if not (0.0 <= args.min_brain_fraction <= 1.0):
        raise ValueError("--min-brain-fraction must be in [0,1]")
    if not (0.0 <= args.cam_percentile <= 100.0):
        raise ValueError("--cam-percentile must be in [0,100]")
    if not (0.0 <= args.global_percentile <= 100.0):
        raise ValueError("--global-percentile must be in [0,100]")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nii = nib.load(str(volume_path))
    vol = np.asarray(nii.get_fdata(dtype=np.float32))
    vol_norm = normalize_volume(vol)

    device = args.device or settings.device
    model = load_model(path=str(model_path), device=device)
    disable_inplace_relu(model)
    model.eval()
    gradcam = SliceGradCAM(model)

    total = num_slices(vol.shape, args.axis)
    if args.slice_index < 0:
        preview_idx = total // 2
    else:
        preview_idx = args.slice_index
    if preview_idx >= total:
        raise IndexError(f"slice-index {preview_idx} out of range [0, {total-1}]")

    heatmap_3d = np.zeros(vol.shape, dtype=np.float32)
    overlay_4d = np.zeros((*vol.shape, 3), dtype=np.uint8)

    preview_png = out_dir / f"{volume_path.stem}_{args.axis}_slice{preview_idx:03d}_gradcam.png"

    for idx in range(0, total, args.save_every):
        sl = get_slice(vol_norm, args.axis, idx)
        brain_mask = sl > args.brain_threshold
        brain_frac = float(brain_mask.mean())

        if brain_frac < args.min_brain_fraction:
            cam = np.zeros_like(sl, dtype=np.float32)
            pred_idx = 2  # notumor fallback label id
            conf = 0.0
        else:
            cam, pred_idx, conf = gradcam.compute(sl, device)
            cam *= brain_mask.astype(np.float32)

            # Keep only strongest activations per-slice within brain region.
            cam_vals = cam[brain_mask]
            if cam_vals.size > 0 and args.cam_percentile > 0:
                cut = float(np.percentile(cam_vals, args.cam_percentile))
                cam = np.where(cam >= cut, cam, 0.0).astype(np.float32)
                cmax = float(cam.max())
                if cmax > 0:
                    cam /= cmax

        put_slice(heatmap_3d, args.axis, idx, cam)

        gray_u8 = np.clip(sl * 255.0, 0, 255).astype(np.uint8)
        color = apply_jet(cam)
        overlay = alpha_blend(gray_u8, color, alpha=0.45)
        put_slice_rgb(overlay_4d, args.axis, idx, overlay)

        if idx == preview_idx:
            Image.fromarray(overlay).save(preview_png)
            print(
                f"Preview slice {idx}: class={TUMOR_CLASSES[pred_idx]} conf={conf:.3f} -> {preview_png}"
            )

    # Global cleanup to remove weak spread-out slabs across volume.
    nonzero = heatmap_3d[heatmap_3d > 0]
    if nonzero.size > 0 and args.global_percentile > 0:
        cut3d = float(np.percentile(nonzero, args.global_percentile))
        heatmap_3d = np.where(heatmap_3d >= cut3d, heatmap_3d, 0.0).astype(np.float32)
        vmax = float(heatmap_3d.max())
        if vmax > 0:
            heatmap_3d /= vmax

    heatmap_path = out_dir / f"{volume_path.stem}_{args.axis}_gradcam_heatmap_3d.nii.gz"
    overlay_path = out_dir / f"{volume_path.stem}_{args.axis}_gradcam_overlay_rgb_4d.nii.gz"

    nib.save(nib.Nifti1Image(heatmap_3d, affine=nii.affine, header=nii.header), str(heatmap_path))

    # RGB 4D NIfTI (X,Y,Z,3). Keep float32 for broad reader compatibility.
    overlay_float = overlay_4d.astype(np.float32)
    nib.save(nib.Nifti1Image(overlay_float, affine=nii.affine), str(overlay_path))

    print(f"Saved 3D heatmap: {heatmap_path}")
    print(f"Saved 4D RGB overlay: {overlay_path}")


if __name__ == "__main__":
    main()
