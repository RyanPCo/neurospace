#!/usr/bin/env python3
"""Interactive 3D MRI renderer for BraTS-style NIfTI datasets.

Supports:
- Volume rendering (pan/zoom/rotate via mouse)
- Optional segmentation overlay as labeled surfaces
- BraTS directory discovery similar to Kaggle notebooks

Example usage:
  python backend/scripts/render_mri_3d.py \
    --dataset-dir /path/to/MICCAI_BraTS2020_TrainingData \
    --subject-index 0 \
    --modality flair

  python backend/scripts/render_mri_3d.py \
    --volume /path/to/BraTS20_Training_001_flair.nii \
    --segmentation /path/to/BraTS20_Training_001_seg.nii

Dependencies:
  pip install nibabel pyvista vtk numpy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pyvista as pv


SEGMENTATION_COLORS = {
    1: "#f94144",  # necrotic / non-enhancing tumor core
    2: "#f8961e",  # edema
    4: "#277da1",  # enhancing tumor
}


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    zooms = image.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms)
    return data, spacing


def normalize_mri(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not np.any(finite):
        raise ValueError("Volume has no finite values.")

    v = np.zeros_like(volume, dtype=np.float32)
    valid = volume[finite]
    lo, hi = np.percentile(valid, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1e-6

    v[finite] = np.clip((volume[finite] - lo) / (hi - lo), 0.0, 1.0)
    return v


def to_grid(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    as_point_data: bool = False,
) -> pv.ImageData:
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) if as_point_data else np.array(volume.shape) + 1
    grid.spacing = spacing
    grid.origin = (0.0, 0.0, 0.0)
    if as_point_data:
        grid.point_data["values"] = volume.flatten(order="F")
    else:
        grid.cell_data["values"] = volume.flatten(order="F")
    return grid


def find_subject_modalities(
    dataset_dir: Path,
    modality: str,
) -> list[tuple[Path, Path | None]]:
    exts = (".nii", ".nii.gz")
    volume_paths: list[Path] = []

    for ext in exts:
        volume_paths.extend(sorted(dataset_dir.rglob(f"*_{modality}{ext}")))

    unique_paths = sorted(set(volume_paths))
    pairs: list[tuple[Path, Path | None]] = []

    for volume_path in unique_paths:
        seg_path = None
        name = volume_path.name
        if f"_{modality}.nii.gz" in name:
            candidate = volume_path.with_name(name.replace(f"_{modality}.nii.gz", "_seg.nii.gz"))
            if candidate.exists():
                seg_path = candidate
        elif f"_{modality}.nii" in name:
            candidate = volume_path.with_name(name.replace(f"_{modality}.nii", "_seg.nii"))
            if candidate.exists():
                seg_path = candidate

        pairs.append((volume_path, seg_path))

    return pairs


def add_segmentation_surfaces(
    plotter: pv.Plotter,
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    labels: Iterable[int],
) -> None:
    for label in labels:
        mask = (segmentation == label).astype(np.uint8)
        if mask.max() == 0:
            continue

        grid = to_grid(mask, spacing, as_point_data=True)
        surface = grid.contour(isosurfaces=[0.5], scalars="values")
        if surface.n_cells == 0:
            continue

        plotter.add_mesh(
            surface,
            color=SEGMENTATION_COLORS.get(label, "#90be6d"),
            opacity=0.42,
            name=f"seg_label_{label}",
            smooth_shading=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive 3D MRI renderer")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-dir",
        type=Path,
        help="BraTS dataset root directory to auto-discover cases.",
    )
    source.add_argument("--volume", type=Path, help="Path to a single MRI NIfTI volume.")

    parser.add_argument(
        "--segmentation",
        type=Path,
        default=None,
        help="Optional segmentation NIfTI path when using --volume.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="flair",
        choices=["flair", "t1", "t1ce", "t2"],
        help="MRI modality used for discovery in --dataset-dir mode.",
    )
    parser.add_argument(
        "--subject-index",
        type=int,
        default=0,
        help="Discovered subject index in sorted order (dataset mode).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Downsample factor per axis to reduce GPU load (>=1).",
    )
    parser.add_argument(
        "--seg-labels",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="Segmentation labels to render as surfaces.",
    )
    parser.add_argument(
        "--gradcam-heatmap",
        type=Path,
        default=None,
        help="Optional 3D Grad-CAM heatmap NIfTI to overlay.",
    )
    parser.add_argument(
        "--generate-gradcam",
        action="store_true",
        help="Generate Grad-CAM heatmap via apply_gradcam_3d.py before rendering.",
    )
    parser.add_argument(
        "--gradcam-model",
        type=Path,
        default=Path("/Users/rohan/CancerScope/models/best_model.pt"),
        help="Model path used when --generate-gradcam is enabled.",
    )
    parser.add_argument(
        "--gradcam-out-dir",
        type=Path,
        default=Path("/Users/rohan/CancerScope/data/gradcam_outputs"),
        help="Output directory used when --generate-gradcam is enabled.",
    )
    parser.add_argument(
        "--gradcam-axis",
        type=str,
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Slice axis used when --generate-gradcam is enabled.",
    )
    parser.add_argument(
        "--gradcam-threshold",
        type=float,
        default=0.25,
        help="Threshold (0..1) for displaying Grad-CAM volume overlay.",
    )

    return parser.parse_args()


def maybe_generate_gradcam(args: argparse.Namespace, volume_path: Path) -> Path | None:
    if args.gradcam_heatmap is not None:
        path = args.gradcam_heatmap.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Grad-CAM heatmap not found: {path}")
        return path

    if not args.generate_gradcam:
        return None

    script_path = Path(__file__).with_name("apply_gradcam_3d.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Grad-CAM script not found: {script_path}")

    out_dir = args.gradcam_out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--volume",
        str(volume_path),
        "--model",
        str(args.gradcam_model.expanduser().resolve()),
        "--axis",
        args.gradcam_axis,
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    # Output naming follows apply_gradcam_3d.py convention.
    stem = volume_path.stem
    return out_dir / f"{stem}_{args.gradcam_axis}_gradcam_heatmap_3d.nii.gz"


def main() -> None:
    args = parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not (0.0 <= args.gradcam_threshold <= 1.0):
        raise ValueError("--gradcam-threshold must be between 0 and 1")

    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        cases = find_subject_modalities(dataset_dir, args.modality)
        if not cases:
            raise FileNotFoundError(
                f"No '*_{args.modality}.nii' or '*_{args.modality}.nii.gz' files found under {dataset_dir}"
            )

        if args.subject_index < 0 or args.subject_index >= len(cases):
            raise IndexError(f"--subject-index must be in [0, {len(cases) - 1}]")

        volume_path, seg_path = cases[args.subject_index]
    else:
        volume_path = args.volume.expanduser().resolve()
        if not volume_path.exists():
            raise FileNotFoundError(f"Volume not found: {volume_path}")
        seg_path = args.segmentation.expanduser().resolve() if args.segmentation else None
        if seg_path is not None and not seg_path.exists():
            raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    gradcam_path = maybe_generate_gradcam(args, volume_path)

    volume, spacing = load_nifti(volume_path)
    if args.stride > 1:
        volume = volume[:: args.stride, :: args.stride, :: args.stride]
        spacing = tuple(s * args.stride for s in spacing)

    volume = normalize_mri(volume)
    volume_grid = to_grid(volume, spacing)

    plotter = pv.Plotter(window_size=[1400, 900])
    plotter.set_background("#0e1116")

    plotter.add_volume(
        volume_grid,
        scalars="values",
        cmap="gray",
        opacity="sigmoid_6",
        shade=True,
        ambient=0.25,
        diffuse=0.7,
        specular=0.1,
    )

    if seg_path is not None:
        segmentation, _ = load_nifti(seg_path)
        if args.stride > 1:
            segmentation = segmentation[:: args.stride, :: args.stride, :: args.stride]
        add_segmentation_surfaces(plotter, segmentation, spacing, args.seg_labels)

    if gradcam_path is not None:
        gradcam, _ = load_nifti(gradcam_path)
        if args.stride > 1:
            gradcam = gradcam[:: args.stride, :: args.stride, :: args.stride]
        if gradcam.shape != volume.shape:
            raise ValueError(
                f"Grad-CAM shape {gradcam.shape} does not match MRI volume shape {volume.shape}"
            )
        gradcam = np.clip(gradcam, 0.0, 1.0)
        # Threshold and remap so overlay focuses on salient regions.
        if args.gradcam_threshold > 0:
            gradcam = np.clip((gradcam - args.gradcam_threshold) / (1.0 - args.gradcam_threshold), 0.0, 1.0)

        grad_grid = to_grid(gradcam, spacing)
        plotter.add_volume(
            grad_grid,
            scalars="values",
            cmap="jet",
            opacity=[0.0, 0.0, 0.15, 0.35, 0.55, 0.75],
            shade=False,
            name="gradcam_overlay",
        )

    info_lines = [
        f"Volume: {volume_path.name}",
        f"Shape: {tuple(volume.shape)}",
        f"Spacing(mm): {tuple(round(s, 3) for s in spacing)}",
        "Mouse: left=rotate, right=zoom, middle(or shift+left)=pan",
    ]
    if seg_path is not None:
        info_lines.append(f"Segmentation: {seg_path.name}")
    if gradcam_path is not None:
        info_lines.append(f"GradCAM: {gradcam_path.name}")

    plotter.add_text("\n".join(info_lines), position="upper_left", font_size=10)
    plotter.show_grid(color="#4b5563")
    plotter.show(title="3D MRI Viewer")


if __name__ == "__main__":
    main()
