"""Brain Tumor MRI dataset loader.

Dataset structure (Kaggle masoudnickparvar/brain-tumor-mri-dataset):
  Training/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/
  Testing/ /{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/
"""
import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import settings, TUMOR_CLASSES


# Map directory name variants → canonical class name
_DIR_TO_CLASS = {
    "glioma_tumor":     "glioma",
    "glioma":           "glioma",
    "meningioma_tumor": "meningioma",
    "meningioma":       "meningioma",
    "no_tumor":         "notumor",
    "notumor":          "notumor",
    "pituitary_tumor":  "pituitary",
    "pituitary":        "pituitary",
}

TUMOR_LABEL = {cls: i for i, cls in enumerate(TUMOR_CLASSES)}


class BrainTumorDataset(Dataset):
    """
    Reads from a split JSON file.
    Each entry: {"id": ..., "file_path": ..., "ground_truth": <class>}
    Returns (image_tensor, label_int, image_id).
    """

    def __init__(self, split_json: str | Path, transform: Callable | None = None):
        self.transform = transform
        with open(split_json) as f:
            self.records: list[dict] = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["file_path"]).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        gt = rec.get("ground_truth")
        if gt not in TUMOR_LABEL:
            raise ValueError(f"Unexpected class in split JSON: {gt!r}")
        label = TUMOR_LABEL[gt]
        return img, label, rec["id"]


def make_dataloaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
    splits_dir: Path | None = None,
) -> dict[str, DataLoader]:
    from core.dataset.transforms import train_transforms, val_transforms

    batch_size = batch_size or settings.batch_size
    num_workers = num_workers or settings.num_workers
    splits_dir = splits_dir or settings.splits_dir

    loaders = {}
    for split, transform in [("train", train_transforms), ("val", val_transforms), ("test", val_transforms)]:
        json_path = splits_dir / f"{split}.json"
        if not json_path.exists():
            continue
        ds = BrainTumorDataset(json_path, transform=transform)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
