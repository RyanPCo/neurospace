#!/usr/bin/env python3
"""Train a CNN on 2D BraTS MRI slices for cancer-related classification.

This script builds 2D axial slices from 3D BraTS NIfTI volumes and trains a
small CNN classifier.

Tasks:
- tumor_presence: binary labels (0=no tumor, 1=tumor)
- tumor_region: 4 labels (0=no_tumor, 1=necrotic_core, 2=edema, 3=enhancing)

Notes:
- Labels are derived from the segmentation slice. For tumor_region, the tumor
  class is the dominant tumor label on that slice.
- Split is performed at case level (not slice level) to avoid leakage.

Dependencies:
  pip install torch torchvision nibabel numpy
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEG_TO_REGION = {
    1: 1,  # necrotic/non-enhancing tumor core
    2: 2,  # edema
    4: 3,  # enhancing tumor
}


@dataclass
class SliceSample:
    image: np.ndarray
    label: int


class SliceTensorDataset(Dataset):
    def __init__(self, samples: list[SliceSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        x = torch.from_numpy(sample.image).unsqueeze(0)  # (1, H, W)
        y = torch.tensor(sample.label, dtype=torch.long)
        return x, y


class SmallSliceCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            feat_dim = int(np.prod(self.features(dummy).shape[1:]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_cases(data_root: Path, modality: str) -> list[tuple[Path, Path]]:
    cases: list[tuple[Path, Path]] = []
    for vol_path in sorted(data_root.rglob(f"*_{modality}.nii")):
        seg_path = Path(str(vol_path).replace(f"_{modality}.nii", "_seg.nii"))
        if seg_path.exists():
            cases.append((vol_path, seg_path))

    for vol_path in sorted(data_root.rglob(f"*_{modality}.nii.gz")):
        seg_path = Path(str(vol_path).replace(f"_{modality}.nii.gz", "_seg.nii.gz"))
        if seg_path.exists():
            cases.append((vol_path, seg_path))

    unique = sorted(set(cases), key=lambda x: str(x[0]))
    return unique


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


def resize_slice(slice_img: np.ndarray, image_size: int) -> np.ndarray:
    t = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy().astype(np.float32)


def tumor_region_label(seg_slice: np.ndarray) -> int:
    tumor = seg_slice[seg_slice > 0]
    if tumor.size == 0:
        return 0
    vals, counts = np.unique(tumor, return_counts=True)
    dominant = int(vals[np.argmax(counts)])
    return SEG_TO_REGION.get(dominant, 0)


def build_samples(
    case_pairs: list[tuple[Path, Path]],
    task: str,
    image_size: int,
    max_bg_ratio: float,
    max_slices_per_case: int,
) -> list[SliceSample]:
    pos_samples: list[SliceSample] = []
    bg_samples: list[SliceSample] = []

    for vol_path, seg_path in case_pairs:
        vol = np.asarray(nib.load(str(vol_path)).get_fdata(dtype=np.float32))
        seg = np.asarray(nib.load(str(seg_path)).get_fdata(dtype=np.float32))

        if vol.shape != seg.shape:
            continue

        vol = normalize_volume(vol)
        z_indices = list(range(vol.shape[2]))
        if max_slices_per_case > 0 and len(z_indices) > max_slices_per_case:
            step = max(1, len(z_indices) // max_slices_per_case)
            z_indices = z_indices[::step][:max_slices_per_case]

        for z in z_indices:
            img = vol[:, :, z]
            seg_slice = seg[:, :, z]
            img = resize_slice(img, image_size)

            if task == "tumor_presence":
                label = 1 if np.any(seg_slice > 0) else 0
            else:
                label = tumor_region_label(seg_slice)

            sample = SliceSample(image=img, label=label)
            if label == 0:
                bg_samples.append(sample)
            else:
                pos_samples.append(sample)

    if max_bg_ratio <= 0:
        selected_bg = []
    else:
        max_bg = int(len(pos_samples) * max_bg_ratio) if pos_samples else len(bg_samples)
        random.shuffle(bg_samples)
        selected_bg = bg_samples[:max_bg]

    all_samples = pos_samples + selected_bg
    random.shuffle(all_samples)
    return all_samples


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == labels).float().mean().item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        b = x.size(0)
        loss_sum += float(loss.item()) * b
        acc_sum += accuracy(logits, y) * b
        n += b

    return loss_sum / max(n, 1), acc_sum / max(n, 1)


def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            b = x.size(0)
            loss_sum += float(loss.item()) * b
            acc_sum += accuracy(logits, y) * b
            n += b

    return loss_sum / max(n, 1), acc_sum / max(n, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on 2D BraTS slices")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing BraTS training cases")
    parser.add_argument("--modality", type=str, default="flair", choices=["flair", "t1", "t1ce", "t2"])
    parser.add_argument("--task", type=str, default="tumor_region", choices=["tumor_region", "tumor_presence"])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-bg-ratio", type=float, default=1.5, help="Max no_tumor:positive sample ratio")
    parser.add_argument("--max-slices-per-case", type=int, default=155, help="<=0 means all slices")
    parser.add_argument("--output", type=Path, default=Path("models/brats2d_cnn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    cases = find_cases(data_root, args.modality)
    if len(cases) < 2:
        raise RuntimeError("Need at least 2 cases with modality+seg files to split train/val.")

    random.shuffle(cases)
    val_count = max(1, int(len(cases) * args.val_fraction))
    val_cases = cases[:val_count]
    train_cases = cases[val_count:]
    if not train_cases:
        raise RuntimeError("Training split is empty. Reduce --val-fraction.")

    print(f"Found {len(cases)} cases ({len(train_cases)} train / {len(val_cases)} val)")
    print(f"Task={args.task}, modality={args.modality}, image_size={args.image_size}")

    train_samples = build_samples(
        train_cases,
        task=args.task,
        image_size=args.image_size,
        max_bg_ratio=args.max_bg_ratio,
        max_slices_per_case=args.max_slices_per_case,
    )
    val_samples = build_samples(
        val_cases,
        task=args.task,
        image_size=args.image_size,
        max_bg_ratio=args.max_bg_ratio,
        max_slices_per_case=args.max_slices_per_case,
    )

    if not train_samples or not val_samples:
        raise RuntimeError("No samples generated. Check dataset paths and labels.")

    num_classes = 2 if args.task == "tumor_presence" else 4

    train_loader = DataLoader(
        SliceTensorDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        SliceTensorDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = get_device()
    print(f"Device: {device}")
    print(f"Train slices: {len(train_samples)} | Val slices: {len(val_samples)}")

    model = SmallSliceCNN(num_classes=num_classes, image_size=args.image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task": args.task,
                    "modality": args.modality,
                    "image_size": args.image_size,
                    "num_classes": num_classes,
                    "history": history,
                },
                args.output,
            )

    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": args.task,
                "modality": args.modality,
                "num_classes": num_classes,
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
                "train_slices": len(train_samples),
                "val_slices": len(val_samples),
                "output": str(args.output),
                "best_val_loss": best_val,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"Saved best model: {args.output}")
    print(f"Saved metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
