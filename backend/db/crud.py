import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from db.models import Image, Prediction, Annotation, Kernel, TrainingRun, TrainingEpoch


# ─── Images ────────────────────────────────────────────────────────────────────

def get_images(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    magnification: Optional[str] = None,
    subtype: Optional[str] = None,
    predicted_class: Optional[str] = None,
    split: Optional[str] = None,
):
    q = db.query(Image)
    if magnification:
        q = q.filter(Image.magnification == magnification)
    if subtype:
        q = q.filter(Image.subtype == subtype)
    if split:
        q = q.filter(Image.split == split)
    if predicted_class:
        # join latest prediction
        q = q.join(Prediction, Image.id == Prediction.image_id).filter(
            Prediction.predicted_class == predicted_class
        )
    total = q.count()
    items = q.offset(skip).limit(limit).all()
    return items, total


def get_image(db: Session, image_id: str) -> Optional[Image]:
    return db.query(Image).filter(Image.id == image_id).first()


def create_image(db: Session, **kwargs) -> Image:
    img = Image(**kwargs)
    db.add(img)
    db.commit()
    db.refresh(img)
    return img


def upsert_image(db: Session, **kwargs) -> Image:
    img = db.query(Image).filter(Image.id == kwargs["id"]).first()
    if img:
        for k, v in kwargs.items():
            setattr(img, k, v)
    else:
        img = Image(**kwargs)
        db.add(img)
    db.commit()
    db.refresh(img)
    return img


# ─── Predictions ───────────────────────────────────────────────────────────────

def get_latest_prediction(db: Session, image_id: str) -> Optional[Prediction]:
    return (
        db.query(Prediction)
        .filter(Prediction.image_id == image_id)
        .order_by(desc(Prediction.created_at))
        .first()
    )


def get_predictions_by_image(db: Session, image_id: str) -> list[Prediction]:
    return db.query(Prediction).filter(Prediction.image_id == image_id).all()


def upsert_prediction(db: Session, image_id: str, model_version: str, **kwargs) -> Prediction:
    pred = (
        db.query(Prediction)
        .filter(Prediction.image_id == image_id, Prediction.model_version == model_version)
        .first()
    )
    if pred:
        for k, v in kwargs.items():
            setattr(pred, k, v)
    else:
        pred = Prediction(image_id=image_id, model_version=model_version, **kwargs)
        db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


# ─── Annotations ───────────────────────────────────────────────────────────────

def create_annotation(db: Session, **kwargs) -> Annotation:
    ann = Annotation(**kwargs)
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return ann


def get_annotations(db: Session, image_id: str) -> list[Annotation]:
    return (
        db.query(Annotation)
        .filter(Annotation.image_id == image_id, Annotation.is_active == True)
        .order_by(Annotation.created_at)
        .all()
    )


def get_all_active_annotations(db: Session) -> dict[str, list[Annotation]]:
    """Returns {image_id: [annotations]} for all active annotations."""
    anns = db.query(Annotation).filter(Annotation.is_active == True).all()
    result: dict[str, list] = {}
    for a in anns:
        result.setdefault(a.image_id, []).append(a)
    return result


def delete_annotation(db: Session, annotation_id: int) -> bool:
    ann = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not ann:
        return False
    ann.is_active = False
    db.commit()
    return True


# ─── Kernels ───────────────────────────────────────────────────────────────────

def get_kernels(
    db: Session,
    layer_name: Optional[str] = None,
    assigned_class: Optional[str] = None,
    include_deleted: bool = False,
    sort_by: str = "importance",
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[Kernel], int]:
    q = db.query(Kernel)
    if not include_deleted:
        q = q.filter(Kernel.is_deleted == False)
    if layer_name:
        q = q.filter(Kernel.layer_name == layer_name)
    if assigned_class:
        q = q.filter(Kernel.assigned_class == assigned_class)
    if sort_by == "importance":
        q = q.order_by(desc(Kernel.importance_score))
    else:
        q = q.order_by(Kernel.id)
    total = q.count()
    items = q.offset(skip).limit(limit).all()
    return items, total


def get_kernel(db: Session, kernel_id: str) -> Optional[Kernel]:
    return db.query(Kernel).filter(Kernel.id == kernel_id).first()


def upsert_kernel(db: Session, kernel_id: str, **kwargs) -> Kernel:
    k = db.query(Kernel).filter(Kernel.id == kernel_id).first()
    if k:
        for key, val in kwargs.items():
            setattr(k, key, val)
    else:
        k = Kernel(id=kernel_id, **kwargs)
        db.add(k)
    db.commit()
    db.refresh(k)
    return k


def update_kernel(db: Session, kernel_id: str, **kwargs) -> Optional[Kernel]:
    k = db.query(Kernel).filter(Kernel.id == kernel_id).first()
    if not k:
        return None
    for key, val in kwargs.items():
        setattr(k, key, val)
    db.commit()
    db.refresh(k)
    return k


def soft_delete_kernel(db: Session, kernel_id: str) -> bool:
    k = db.query(Kernel).filter(Kernel.id == kernel_id).first()
    if not k:
        return False
    k.is_deleted = True
    db.commit()
    return True


# ─── Training Runs ─────────────────────────────────────────────────────────────

def create_training_run(db: Session, run_id: str, config: dict) -> TrainingRun:
    run = TrainingRun(
        id=run_id,
        status="pending",
        config_json=json.dumps(config),
        start_time=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_training_run(db: Session, run_id: str) -> Optional[TrainingRun]:
    return db.query(TrainingRun).filter(TrainingRun.id == run_id).first()


def get_all_training_runs(db: Session) -> list[TrainingRun]:
    return db.query(TrainingRun).order_by(desc(TrainingRun.start_time)).all()


def get_latest_training_run(db: Session) -> Optional[TrainingRun]:
    return db.query(TrainingRun).order_by(desc(TrainingRun.start_time)).first()


def update_training_run(db: Session, run_id: str, **kwargs) -> Optional[TrainingRun]:
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        return None
    for k, v in kwargs.items():
        setattr(run, k, v)
    db.commit()
    db.refresh(run)
    return run


def add_training_epoch(db: Session, run_id: str, **kwargs) -> TrainingEpoch:
    epoch = TrainingEpoch(run_id=run_id, **kwargs)
    db.add(epoch)
    db.commit()
    db.refresh(epoch)
    return epoch


def get_training_epochs(db: Session, run_id: str) -> list[TrainingEpoch]:
    return (
        db.query(TrainingEpoch)
        .filter(TrainingEpoch.run_id == run_id)
        .order_by(TrainingEpoch.epoch)
        .all()
    )
