"""Walk brain tumor MRI dataset, populate images table, create train/val/test splits.

Dataset expected at data/raw/brain_tumor_mri/:
  Training/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/
  Testing/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/
"""
import sys
import json
import hashlib
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image as PILImage
from config import settings
from db.database import init_db, SessionLocal
from db.crud import upsert_image
from core.dataset.brain_tumor import _DIR_TO_CLASS


VAL_RATIO = 0.15
SEED = 42

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_image_id(file_path: str) -> str:
    return hashlib.sha256(file_path.encode()).hexdigest()[:16]


def get_dimensions(file_path: str) -> tuple[int, int]:
    try:
        with PILImage.open(file_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (512, 512)


def walk_split_dir(split_dir: Path, split_name: str) -> list[dict]:
    """Walk one directory (Training or Testing) and return record list."""
    records = []
    if not split_dir.exists():
        return records
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = _DIR_TO_CLASS.get(cls_dir.name.lower())
        if cls_name is None:
            print(f"  Skipping unknown class dir: {cls_dir.name}")
            continue
        for img_file in cls_dir.iterdir():
            if img_file.suffix.lower() not in IMG_EXTENSIONS:
                continue
            records.append({
                "id":           compute_image_id(str(img_file)),
                "filename":     img_file.name,
                "file_path":    str(img_file),
                "ground_truth": cls_name,
                "subtype":      cls_name,   # reuse for display
                "magnification": "",
                "split":        split_name,
            })
    return records


def main():
    train_dir = settings.raw_data_dir / "Training"
    test_dir  = settings.raw_data_dir / "Testing"

    if not train_dir.exists():
        print(f"ERROR: Dataset not found at {settings.raw_data_dir}")
        print("Expected structure:")
        print("  data/raw/brain_tumor_mri/Training/{glioma_tumor,...}/")
        print("  data/raw/brain_tumor_mri/Testing/{glioma_tumor,...}/")
        sys.exit(1)

    print(f"Scanning {settings.raw_data_dir}...")
    train_records = walk_split_dir(train_dir, "train_raw")  # we'll re-split below
    test_records  = walk_split_dir(test_dir,  "test")

    print(f"  Training dir: {len(train_records)} images")
    print(f"  Testing  dir: {len(test_records)}  images")

    # Stratified train/val split from the Training directory
    random.seed(SEED)
    by_class: dict[str, list] = {}
    for rec in train_records:
        by_class.setdefault(rec["ground_truth"], []).append(rec)

    all_train, all_val = [], []
    for cls_name, recs in by_class.items():
        random.shuffle(recs)
        n_val = max(1, int(len(recs) * VAL_RATIO))
        for r in recs[:n_val]:
            r["split"] = "val"
        for r in recs[n_val:]:
            r["split"] = "train"
        all_val.extend(recs[:n_val])
        all_train.extend(recs[n_val:])

    all_records = all_train + all_val + test_records

    # Init DB and insert
    init_db()
    db = SessionLocal()
    print(f"\nIndexing {len(all_records)} images into database...")

    for i, rec in enumerate(all_records):
        w, h = get_dimensions(rec["file_path"])
        upsert_image(
            db,
            id=rec["id"],
            filename=rec["filename"],
            file_path=rec["file_path"],
            magnification=rec["magnification"],
            subtype=rec["subtype"],
            ground_truth=rec["ground_truth"],
            split=rec["split"],
            width=w,
            height=h,
        )
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(all_records)} indexed...")

    db.close()

    # Write split JSONs
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, data in [("train", all_train), ("val", all_val), ("test", test_records)]:
        path = settings.splits_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  {split_name}: {len(data)} images → {path}")

    print(f"\nDone! {len(all_records)} total images indexed.")
    for cls, recs in by_class.items():
        print(f"  {cls}: {len(recs)} training images")
    print(f"  test: {len(test_records)} images")


if __name__ == "__main__":
    main()
