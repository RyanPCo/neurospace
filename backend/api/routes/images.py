"""Image endpoints: list, detail, gradcam, file streaming."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from db.database import get_db
from db import crud
from schemas.images import (
    AnnotationRetrainRequest,
    AnnotationRetrainResponse,
    ImageSummary,
    ImageDetail,
    GradCAMResponse,
    PaginatedImages,
)
from services.model_manager import get_model_manager
from config import TUMOR_CLASSES
from config import settings

router = APIRouter(prefix="/api/images", tags=["images"])


def _enrich(image, db: Session, mm=None) -> dict:
    pred = crud.get_latest_prediction(db, image.id)
    ann_count = len(crud.get_annotations(db, image.id))
    data = {
        "id": image.id,
        "filename": image.filename,
        "ground_truth": image.ground_truth,
        "split": image.split,
        "width": image.width,
        "height": image.height,
        "annotation_count": ann_count,
        "predicted_class": None,
        "confidence": None,
    }
    if pred:
        data["predicted_class"] = pred.predicted_class
        data["confidence"] = pred.confidence
    elif mm and mm.is_loaded() and Path(image.file_path).exists():
        try:
            result = mm.predict(image.file_path, image.id, use_cache=True)
            data["predicted_class"] = result["predicted_class"]
            data["confidence"] = result["confidence"]
            # Persist prediction
            crud.upsert_prediction(
                db, image.id, mm.model_version,
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                class_probs_json=json.dumps(result["class_probs"]),
            )
        except Exception:
            pass
    return data


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _annotation_points_to_pixels(points: list[dict], size: int) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for p in points:
        if not isinstance(p, dict) or "x" not in p or "y" not in p:
            continue
        x = _clip01(p["x"]) * (size - 1)
        y = _clip01(p["y"]) * (size - 1)
        coords.append((x, y))
    return coords


def _build_mask_for_annotations(annotations: list, size: int) -> np.ndarray:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        try:
            geom = json.loads(ann.geometry_json)
        except Exception:
            continue

        if ann.geometry_type == "polygon":
            pts = _annotation_points_to_pixels(geom.get("points", []), size)
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        elif ann.geometry_type == "brush":
            pts = _annotation_points_to_pixels(geom.get("strokes", []), size)
            radius_norm = float(geom.get("radius", 0.02))
            radius_px = max(1, int(round(radius_norm * size)))
            for x, y in pts:
                draw.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=255)

    return np.asarray(mask, dtype=np.uint8)


@router.get("", response_model=PaginatedImages)
def list_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.page_size, ge=1, le=200),
    predicted_class: Optional[str] = Query(None),
    split: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    skip = (page - 1) * page_size
    images, total = crud.get_images(
        db, skip=skip, limit=page_size,
        predicted_class=predicted_class, split=split,
    )
    mm = get_model_manager()
    items = [ImageSummary(**_enrich(img, db, mm)) for img in images]
    return PaginatedImages(items=items, total=total, page=page, page_size=page_size)


@router.get("/{image_id}", response_model=ImageDetail)
def get_image_detail(image_id: str, db: Session = Depends(get_db)):
    image = crud.get_image(db, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    mm = get_model_manager()
    data = _enrich(image, db, mm)
    data["file_path"] = image.file_path
    pred = crud.get_latest_prediction(db, image_id)
    if pred:
        data["model_version"] = pred.model_version
        if pred.class_probs_json:
            data["class_probs"] = json.loads(pred.class_probs_json)
    return ImageDetail(**data)


@router.get("/{image_id}/gradcam", response_model=GradCAMResponse)
async def get_gradcam(image_id: str, db: Session = Depends(get_db)):
    image = crud.get_image(db, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    if not Path(image.file_path).exists():
        raise HTTPException(404, "Image file not found on disk")
    mm = get_model_manager()
    try:
        result = await mm.get_gradcam(image.file_path, image_id)
    except Exception as e:
        raise HTTPException(500, f"GradCAM failed: {e}")
    return GradCAMResponse(**result)


@router.get("/{image_id}/file")
def get_image_file(image_id: str, db: Session = Depends(get_db)):
    image = crud.get_image(db, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    path = Path(image.file_path)
    if not path.exists():
        raise HTTPException(404, "Image file not found on disk")
    return FileResponse(str(path), media_type="image/jpeg")


@router.post("/{image_id}/retrain-from-annotation", response_model=AnnotationRetrainResponse)
def retrain_from_annotation(
    image_id: str,
    body: AnnotationRetrainRequest,
    db: Session = Depends(get_db),
):
    image = crud.get_image(db, image_id)
    if not image:
        raise HTTPException(404, "Image not found")

    image_path = Path(image.file_path)
    if not image_path.exists():
        raise HTTPException(404, "Image file not found on disk")

    annotations = crud.get_annotations(db, image_id)
    if not annotations:
        raise HTTPException(400, "No active annotations found for this image")

    roi_anns = [a for a in annotations if a.label_class != "notumor"]
    neg_anns = [a for a in annotations if a.label_class == "notumor"]

    if not roi_anns:
        raise HTTPException(400, "No ROI annotations found. Add at least one non-'notumor' annotation.")

    artifact_dir = settings.data_dir / "annotation_training" / image_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    roi_mask_path = artifact_dir / f"{image_id}_roi_mask.png"
    neg_mask_path = artifact_dir / f"{image_id}_neg_mask.png"

    roi_mask = _build_mask_for_annotations(roi_anns, settings.image_size)
    neg_mask = _build_mask_for_annotations(neg_anns, settings.image_size)

    if int((roi_mask > 0).sum()) == 0:
        raise HTTPException(400, "ROI mask is empty after rasterization; please redraw annotation.")

    Image.fromarray(roi_mask).save(roi_mask_path)
    Image.fromarray(neg_mask).save(neg_mask_path)

    latest_pred = crud.get_latest_prediction(db, image_id)
    target_class = body.target_class or (latest_pred.predicted_class if latest_pred else image.ground_truth)
    if target_class not in TUMOR_CLASSES:
        target_class = "notumor"

    model_in = Path(body.model_in).expanduser().resolve() if body.model_in else settings.best_model_path.resolve()
    if not model_in.exists():
        raise HTTPException(404, f"Input model not found: {model_in}")

    model_out = (
        Path(body.model_out).expanduser().resolve()
        if body.model_out
        else (settings.models_dir / "best_model_expl_tuned.pt").resolve()
    )

    script_path = (Path(__file__).resolve().parents[2] / "scripts" / "fit_explanation_from_annotation.py").resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--annotated-image", str(image_path),
        "--roi-mask", str(roi_mask_path),
        "--neg-mask", str(neg_mask_path),
        "--target-class", target_class,
        "--model-in", str(model_in),
        "--model-out", str(model_out),
        "--steps", str(body.steps),
        "--lr", str(body.lr),
        "--alpha", str(body.alpha),
        "--beta", str(body.beta),
        "--gamma", str(body.gamma),
        "--preserve-ce-weight", str(body.preserve_ce_weight),
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
    except subprocess.CalledProcessError as e:
        detail = e.stderr.strip() or e.stdout.strip() or str(e)
        raise HTTPException(500, f"Annotation retraining failed: {detail}")

    mm = get_model_manager()
    try:
        mm.reload(str(model_out))
    except Exception as e:
        raise HTTPException(500, f"Model retrained but reload failed: {e}")

    log_tail = (proc.stdout or "").strip().splitlines()[-12:]
    return AnnotationRetrainResponse(
        image_id=image_id,
        message="Retraining completed and tuned model is active for Grad-CAM.",
        roi_mask_path=str(roi_mask_path),
        neg_mask_path=str(neg_mask_path),
        model_in=str(model_in),
        model_out=str(model_out),
        command=" ".join(cmd),
        log_tail="\n".join(log_tail),
    )
