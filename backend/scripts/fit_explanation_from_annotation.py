#!/usr/bin/env python3
"""Fine-tune BrainTumorCNN so explanations align with annotation masks.

This script performs explanation-guided fine-tuning with:
- Positive ROI encouragement
- Negative region suppression
- Optional teacher-student preservation (KL + CE on a preserve batch)

It is designed for "one annotated image" workflows but can be run repeatedly.

Example:
  python backend/scripts/fit_explanation_from_annotation.py \
    --annotated-image /path/to/image.jpg \
    --roi-mask /path/to/roi_mask.png \
    --neg-mask /path/to/neg_mask.png \
    --target-class meningioma \
    --steps 300 \
    --model-in /Users/rohan/CancerScope/models/best_model.pt \
    --model-out /Users/rohan/CancerScope/models/best_model_expl_tuned.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TUMOR_CLASSES, settings
from core.dataset.transforms import inference_transforms
from core.model.cnn import BrainTumorCNN, load_model


EPS = 1e-8


class PreserveJsonDataset(Dataset):
    """Loads preserve images from split JSON: [{file_path, ground_truth, ...}, ...]."""

    def __init__(self, split_json: Path):
        with open(split_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.records = [r for r in self.records if "file_path" in r and Path(r["file_path"]).exists()]
        self.class_to_idx = {c: i for i, c in enumerate(TUMOR_CLASSES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img = Image.open(rec["file_path"]).convert("L")
        x = inference_transforms(img)

        gt = rec.get("ground_truth", "notumor")
        y = self.class_to_idx.get(gt, self.class_to_idx["notumor"])
        return x, torch.tensor(y, dtype=torch.long)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def disable_inplace_relu(model: nn.Module) -> None:
    # Safer for second-order gradient setups.
    for module in model.modules():
        if isinstance(module, nn.ReLU) and module.inplace:
            module.inplace = False


def freeze_for_targeted_finetune(model: BrainTumorCNN) -> None:
    # Freeze early feature blocks; unfreeze last block + classifier.
    for p in model.parameters():
        p.requires_grad = False

    for p in model.conv_block3.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def parse_class_idx(target_class: str | None) -> int | None:
    if target_class is None:
        return None
    mapping = {c: i for i, c in enumerate(TUMOR_CLASSES)}
    if target_class not in mapping:
        raise ValueError(f"Unknown --target-class '{target_class}'. Choices: {TUMOR_CLASSES}")
    return mapping[target_class]


def load_mask(path: Path | None, image_size: int) -> torch.Tensor:
    if path is None:
        arr = np.zeros((image_size, image_size), dtype=np.float32)
    else:
        img = Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST)
        arr = (np.asarray(img, dtype=np.float32) / 255.0)
        arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)


def load_annotated_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    return inference_transforms(img).unsqueeze(0)  # (1,1,256,256)


def compute_saliency_map(model: BrainTumorCNN, x: torch.Tensor, class_idx: int) -> torch.Tensor:
    """Differentiable saliency map in [0,1], shape (B,H,W)."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)["logits"]
    score = logits[:, class_idx].sum()
    grads = torch.autograd.grad(score, x, create_graph=True)[0]

    e = grads.abs().mean(dim=1)  # (B,H,W)
    flat = e.flatten(start_dim=1)
    emin = flat.min(dim=1)[0].view(-1, 1, 1)
    emax = flat.max(dim=1)[0].view(-1, 1, 1)
    e = (e - emin) / (emax - emin + EPS)
    return e


def make_preserve_loader(split_json: Path, batch_size: int) -> DataLoader | None:
    if not split_json.exists():
        return None
    ds = PreserveJsonDataset(split_json)
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune model explanation from annotation masks")
    parser.add_argument("--annotated-image", type=Path, required=True, help="Annotated input image")
    parser.add_argument("--roi-mask", type=Path, required=True, help="Mask where explanation should be high")
    parser.add_argument("--neg-mask", type=Path, default=None, help="Mask where explanation should be low")
    parser.add_argument("--target-class", type=str, default=None, help=f"One of {TUMOR_CLASSES}; default uses teacher prediction")

    parser.add_argument("--model-in", type=Path, default=Path("/Users/rohan/CancerScope/models/best_model.pt"))
    parser.add_argument("--model-out", type=Path, default=Path("/Users/rohan/CancerScope/models/best_model_expl_tuned.pt"))

    parser.add_argument("--preserve-split-json", type=Path, default=Path("/Users/rohan/CancerScope/data/splits/train.json"))
    parser.add_argument("--preserve-batch-size", type=int, default=16)

    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--alpha", type=float, default=0.5, help="ROI alignment weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Negative mask suppression weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Prediction preserve (KL) weight")
    parser.add_argument("--preserve-ce-weight", type=float, default=0.5, help="CE weight on preserve batch")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    seed_everything(args.seed)

    model_in = args.model_in.expanduser().resolve()
    if not model_in.exists():
        raise FileNotFoundError(f"Model not found: {model_in}")

    ann_img = args.annotated_image.expanduser().resolve()
    roi_mask_path = args.roi_mask.expanduser().resolve()
    neg_mask_path = args.neg_mask.expanduser().resolve() if args.neg_mask else None

    if not ann_img.exists():
        raise FileNotFoundError(f"Annotated image not found: {ann_img}")
    if not roi_mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {roi_mask_path}")
    if neg_mask_path is not None and not neg_mask_path.exists():
        raise FileNotFoundError(f"Negative mask not found: {neg_mask_path}")

    device = settings.device

    teacher = load_model(path=str(model_in), device=device)
    teacher.eval()
    disable_inplace_relu(teacher)
    for p in teacher.parameters():
        p.requires_grad = False

    student = load_model(path=str(model_in), device=device)
    disable_inplace_relu(student)
    freeze_for_targeted_finetune(student)
    student.train()

    x_ann = load_annotated_image(ann_img).to(device)
    roi_mask = load_mask(roi_mask_path, settings.image_size).to(device)
    neg_mask = load_mask(neg_mask_path, settings.image_size).to(device)

    target_class_idx = parse_class_idx(args.target_class)
    if target_class_idx is None:
        with torch.no_grad():
            pred = teacher(x_ann)["logits"].argmax(dim=1).item()
        target_class_idx = int(pred)

    y_ann = torch.tensor([target_class_idx], dtype=torch.long, device=device)

    preserve_loader = make_preserve_loader(args.preserve_split_json.expanduser().resolve(), args.preserve_batch_size)
    preserve_iter = iter(preserve_loader) if preserve_loader is not None else None

    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters after freezing strategy.")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Target class: {TUMOR_CLASSES[target_class_idx]} ({target_class_idx})")
    print(f"Trainable params: {sum(p.numel() for p in params):,}")
    print(f"Preserve batch enabled: {preserve_loader is not None}")

    for step in range(1, args.steps + 1):
        logits_ann = student(x_ann)["logits"]
        l_task_ann = ce(logits_ann, y_ann)

        sal = compute_saliency_map(student, x_ann, target_class_idx)  # (1,H,W)

        neg_denom = neg_mask.sum() + EPS
        l_neg = (sal * neg_mask).sum() / neg_denom

        roi_sum = roi_mask.sum()
        if roi_sum.item() <= 0:
            l_pos = torch.tensor(0.0, device=device)
        else:
            mass_in_roi = (sal * roi_mask).sum()
            total_mass = sal.sum() + EPS
            l_pos = 1.0 - (mass_in_roi / total_mass)

        l_preserve = torch.tensor(0.0, device=device)
        l_preserve_ce = torch.tensor(0.0, device=device)

        if preserve_loader is not None:
            (x_p, y_p), preserve_iter = next_batch(preserve_iter, preserve_loader)
            x_p = x_p.to(device)
            y_p = y_p.to(device)

            s_logits = student(x_p)["logits"]
            with torch.no_grad():
                t_logits = teacher(x_p)["logits"]

            t = args.temperature
            l_preserve = F.kl_div(
                F.log_softmax(s_logits / t, dim=1),
                F.softmax(t_logits / t, dim=1),
                reduction="batchmean",
            ) * (t * t)
            l_preserve_ce = ce(s_logits, y_p)

        loss = (
            l_task_ann
            + args.alpha * l_pos
            + args.beta * l_neg
            + args.gamma * l_preserve
            + args.preserve_ce_weight * l_preserve_ce
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                probs = torch.softmax(student(x_ann)["logits"], dim=1)[0]
                conf = float(probs[target_class_idx].item())
            print(
                f"step {step:04d}/{args.steps} "
                f"loss={float(loss.item()):.4f} "
                f"task={float(l_task_ann.item()):.4f} "
                f"pos={float(l_pos.item()):.4f} "
                f"neg={float(l_neg.item()):.4f} "
                f"preserve={float(l_preserve.item()):.4f} "
                f"target_conf={conf:.3f}"
            )

    out_path = args.model_out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "source_model": str(model_in),
            "target_class": TUMOR_CLASSES[target_class_idx],
            "target_class_idx": target_class_idx,
            "annotated_image": str(ann_img),
            "roi_mask": str(roi_mask_path),
            "neg_mask": str(neg_mask_path) if neg_mask_path else None,
            "steps": args.steps,
            "lr": args.lr,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "preserve_ce_weight": args.preserve_ce_weight,
            "temperature": args.temperature,
        },
        out_path,
    )

    print(f"Saved tuned model to: {out_path}")


if __name__ == "__main__":
    main()
