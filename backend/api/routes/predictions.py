"""Prediction endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional

from db.database import get_db
from db import crud
from schemas.predictions import PredictionResponse

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("", response_model=list[PredictionResponse])
def get_predictions(
    image_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    if image_id:
        preds = crud.get_predictions_by_image(db, image_id)
    else:
        # Return latest prediction per image — limited query
        from db.models import Prediction
        preds = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(200).all()
    return [PredictionResponse.model_validate(p) for p in preds]
