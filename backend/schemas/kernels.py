from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class KernelSummary(BaseModel):
    id: str
    layer_name: str
    filter_index: int
    importance_score: float = 0.0
    assigned_class: Optional[str] = None
    is_deleted: bool = False
    doctor_notes: Optional[str] = None
    last_scored_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class KernelUpdate(BaseModel):
    assigned_class: Optional[str] = None
    doctor_notes: Optional[str] = None


class TopActivatingImage(BaseModel):
    image_id: str
    max_activation: float
    activation_map_b64: str


class KernelActivationsResponse(BaseModel):
    kernel_id: str
    top_images: list[TopActivatingImage]


class PaginatedKernels(BaseModel):
    items: list[KernelSummary]
    total: int
    page: int
    page_size: int


class BatchKernelUpdate(BaseModel):
    kernel_ids: list[str]
    action: str  # "delete" or "reclassify"
    assigned_class: Optional[str] = None
