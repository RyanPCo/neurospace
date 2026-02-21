from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class PredictionResponse(BaseModel):
    id: int
    image_id: str
    model_version: str
    predicted_class: str
    confidence: float
    class_probs_json: Optional[str] = None
    subtype_predicted: Optional[str] = None
    subtype_probs_json: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
