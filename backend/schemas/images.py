from pydantic import BaseModel
from typing import Optional


class ImageSummary(BaseModel):
    id: str
    filename: str
    magnification: str
    subtype: str
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
    subtype_predicted: Optional[str] = None


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
