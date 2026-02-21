from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AnnotationCreate(BaseModel):
    image_id: str
    label_class: str  # "malignant" or "benign"
    geometry_type: str  # "polygon" or "brush"
    geometry_json: str  # JSON string with normalized coords
    notes: Optional[str] = None


class AnnotationResponse(BaseModel):
    id: int
    image_id: str
    label_class: str
    geometry_type: str
    geometry_json: str
    notes: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
