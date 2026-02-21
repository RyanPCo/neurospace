"""Image endpoints: list, detail, gradcam, file streaming."""
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from db.database import get_db
from db import crud
from schemas.images import ImageSummary, ImageDetail, GradCAMResponse, PaginatedImages
from services.model_manager import get_model_manager
from config import settings

router = APIRouter(prefix="/api/images", tags=["images"])


def _enrich(image, db: Session, mm=None) -> dict:
    pred = crud.get_latest_prediction(db, image.id)
    ann_count = len(crud.get_annotations(db, image.id))
    data = {
        "id": image.id,
        "filename": image.filename,
        "magnification": image.magnification,
        "subtype": image.subtype,
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
                subtype_predicted=result["subtype_predicted"],
                subtype_probs_json=json.dumps(result["subtype_probs"]),
            )
        except Exception:
            pass
    return data


@router.get("", response_model=PaginatedImages)
def list_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.page_size, ge=1, le=200),
    magnification: Optional[str] = Query(None),
    subtype: Optional[str] = Query(None),
    predicted_class: Optional[str] = Query(None),
    split: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    skip = (page - 1) * page_size
    images, total = crud.get_images(
        db, skip=skip, limit=page_size,
        magnification=magnification, subtype=subtype,
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
        data["subtype_predicted"] = pred.subtype_predicted
    return ImageDetail(**data)


@router.get("/{image_id}/gradcam", response_model=GradCAMResponse)
async def get_gradcam(image_id: str, db: Session = Depends(get_db)):
    image = crud.get_image(db, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    if not Path(image.file_path).exists():
        raise HTTPException(404, "Image file not found on disk")
    mm = get_model_manager()
    if not mm.is_loaded():
        raise HTTPException(503, "Model not loaded. Run 'make train' first.")
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
