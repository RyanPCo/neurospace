"""BreakHis dataset loader with split support and image_id tracking."""
import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import settings


BINARY_LABEL = {"benign": 0, "malignant": 1}
SUBTYPE_LABEL = {"A": 0, "F": 1, "PT": 2, "TA": 3, "DC": 4, "LC": 5, "MC": 6, "PC": 7}


def parse_breakhis_path(path: Path) -> dict:
    """
    BreakHis path looks like:
      .../breast_cancer/benign/SOB/adenosis/SOB_B_A-14-22549AB/40X/SOB_B_A...jpg
    Returns dict with keys: ground_truth, subtype, magnification
    """
    parts = path.parts
    try:
        bc_idx = next(i for i, p in enumerate(parts) if "breast_cancer" in p or "BreaKHis" in p.lower())
    except StopIteration:
        bc_idx = 0

    ground_truth = "benign"
    subtype = "unknown"
    magnification = "40X"

    for part in parts:
        if part.lower() == "benign":
            ground_truth = "benign"
        elif part.lower() == "malignant":
            ground_truth = "malignant"
        if part in SUBTYPE_LABEL:
            subtype = part
        if part.upper() in {"40X", "100X", "200X", "400X"}:
            magnification = part.upper()

    # Try to parse subtype from filename: SOB_B_A-... (A=adenosis) or SOB_M_DC-... (DC=ductal)
    stem = path.stem
    fname_parts = stem.split("_")
    if len(fname_parts) >= 3:
        st = fname_parts[2].split("-")[0].upper()
        if st in SUBTYPE_LABEL:
            subtype = st

    return {
        "ground_truth": ground_truth,
        "subtype": subtype,
        "magnification": magnification.lower().replace("x", "X").replace("X", "x"),
    }


class BreakHisDataset(Dataset):
    """
    Dataset that reads from a split JSON file.
    Each entry in the JSON: {"id": ..., "file_path": ..., "ground_truth": ..., "subtype": ..., "magnification": ...}
    """

    def __init__(self, split_json: str | Path, transform: Callable | None = None):
        self.transform = transform
        with open(split_json) as f:
            self.records: list[dict] = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["file_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        binary_label = BINARY_LABEL[rec["ground_truth"]]
        subtype_label = SUBTYPE_LABEL.get(rec["subtype"], 0)
        return img, (binary_label, subtype_label), rec["id"]


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
        ds = BreakHisDataset(json_path, transform=transform)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
