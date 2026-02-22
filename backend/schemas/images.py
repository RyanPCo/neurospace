from pydantic import BaseModel
from typing import Optional


class ImageSummary(BaseModel):
    id: str
    filename: str
    ground_truth: str
    split: str
    width: Optional[int] = None
    height: Optional[int] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    annotation_count: int = 0

    model_config = {"from_attributes": True}


class ImageDetail(ImageSummary):
    file_path: str
    model_version: Optional[str] = None
    class_probs: Optional[dict] = None


class GradCAMResponse(BaseModel):
    image_id: str
    heatmap_b64: str
    overlay_b64: str
    top_kernel_indices: list[int]
    predicted_class: str
    confidence: float


class PaginatedImages(BaseModel):
    items: list[ImageSummary]
    total: int
    page: int
    page_size: int


class AnnotationRetrainRequest(BaseModel):
    steps: int = 150
    lr: float = 5e-5
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 1.0
    preserve_ce_weight: float = 0.5
    target_class: Optional[str] = None
    model_in: Optional[str] = None
    model_out: Optional[str] = None


class AnnotationRetrainResponse(BaseModel):
    image_id: str
    message: str
    roi_mask_path: str
    neg_mask_path: str
    model_in: str
    model_out: str
    command: str
    log_tail: str
