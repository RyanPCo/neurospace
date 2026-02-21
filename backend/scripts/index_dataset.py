"""Walk BreakHis dataset, populate images table, create train/val/test splits."""
import sys
import json
import hashlib
import random
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image as PILImage
from config import settings
from db.database import init_db, SessionLocal
from db.crud import upsert_image
from core.dataset.breakhis import parse_breakhis_path


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def compute_image_id(file_path: str) -> str:
    return hashlib.sha256(file_path.encode()).hexdigest()[:16]


def walk_breakhis(root: Path) -> list[dict]:
    records = []
    for img_path in root.rglob("*.png"):
        meta = parse_breakhis_path(img_path)
        records.append({
            "id": compute_image_id(str(img_path)),
            "filename": img_path.name,
            "file_path": str(img_path),
            **meta,
        })
    # Also scan .jpg
    for img_path in root.rglob("*.jpg"):
        meta = parse_breakhis_path(img_path)
        records.append({
            "id": compute_image_id(str(img_path)),
            "filename": img_path.name,
            "file_path": str(img_path),
            **meta,
        })
    return records


def get_image_dimensions(file_path: str) -> tuple[int, int]:
    try:
        with PILImage.open(file_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (700, 460)  # BreakHis default


def main():
    if not settings.raw_data_dir.exists():
        print(f"ERROR: Dataset not found at {settings.raw_data_dir}")
        print("Please download BreakHis and place it at data/raw/BreaKHis_v1/")
        sys.exit(1)

    print(f"Scanning {settings.raw_data_dir} ...")
    records = walk_breakhis(settings.raw_data_dir)
    print(f"Found {len(records)} images")

    if not records:
        print("No images found. Check that the dataset path is correct.")
        sys.exit(1)

    # Assign splits (stratified by ground_truth)
    random.seed(SEED)
    benign = [r for r in records if r["ground_truth"] == "benign"]
    malignant = [r for r in records if r["ground_truth"] == "malignant"]

    def split_group(group):
        random.shuffle(group)
        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]
        for r in train:
            r["split"] = "train"
        for r in val:
            r["split"] = "val"
        for r in test:
            r["split"] = "test"
        return train, val, test

    b_train, b_val, b_test = split_group(benign)
    m_train, m_val, m_test = split_group(malignant)

    all_train = b_train + m_train
    all_val = b_val + m_val
    all_test = b_test + m_test

    # Initialize DB and insert
    init_db()
    db = SessionLocal()

    print("Indexing images into database...")
    for i, rec in enumerate(records):
        w, h = get_image_dimensions(rec["file_path"])
        upsert_image(
            db,
            id=rec["id"],
            filename=rec["filename"],
            file_path=rec["file_path"],
            magnification=rec.get("magnification", "40X"),
            subtype=rec.get("subtype", "unknown"),
            ground_truth=rec["ground_truth"],
            split=rec["split"],
            width=w,
            height=h,
        )
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(records)} indexed...")

    db.close()

    # Write split JSONs
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", all_train), ("val", all_val), ("test", all_test)]:
        path = settings.splits_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(split_data, f)
        print(f"  {split_name}: {len(split_data)} images → {path}")

    print(f"\nDone! {len(records)} images indexed.")
    print(f"  Train: {len(all_train)}, Val: {len(all_val)}, Test: {len(all_test)}")
    print(f"  Benign: {len(benign)}, Malignant: {len(malignant)}")


if __name__ == "__main__":
    main()
