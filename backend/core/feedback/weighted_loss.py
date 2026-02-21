"""Annotation-weighted loss: CrossEntropy + spatial alignment with doctor annotations."""
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from config import settings


class AnnotationWeightedLoss(nn.Module):
    """
    Loss = 0.7 * CE(binary) + 0.3 * CE(subtype) + 0.3 * L_spatial

    L_spatial aligns the class activation map (CAM) with doctor-drawn annotation masks.
    Only computed for images in the batch that have annotations.
    """

    def __init__(
        self,
        binary_weight: float = None,
        subtype_weight: float = None,
        spatial_weight: float = None,
    ):
        super().__init__()
        self.binary_weight = binary_weight or settings.binary_loss_weight
        self.subtype_weight = subtype_weight or settings.subtype_loss_weight
        self.spatial_weight = spatial_weight or settings.spatial_loss_weight
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        binary_labels: torch.Tensor,
        subtype_labels: torch.Tensor,
        annotation_masks: Optional[list[Optional[torch.Tensor]]] = None,
        binary_head_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            outputs: {"binary_logits", "subtype_logits", "feature_map"}
            binary_labels: (B,) long tensor
            subtype_labels: (B,) long tensor
            annotation_masks: list of B items, each is (1, 1, H_img, W_img) mask or None
            binary_head_weight: (2, 2048) weight from binary_head Linear layer
        """
        binary_loss = self.ce(outputs["binary_logits"], binary_labels)
        subtype_loss = self.ce(outputs["subtype_logits"], subtype_labels)

        spatial_loss = torch.tensor(0.0, device=binary_labels.device)
        if annotation_masks is not None and binary_head_weight is not None:
            spatial_loss = self._compute_spatial_loss(
                outputs["feature_map"], binary_labels, annotation_masks, binary_head_weight
            )

        total = (
            self.binary_weight * binary_loss
            + self.subtype_weight * subtype_loss
            + self.spatial_weight * spatial_loss
        )

        metrics = {
            "binary_loss": binary_loss.item(),
            "subtype_loss": subtype_loss.item(),
            "spatial_loss": spatial_loss.item(),
            "total_loss": total.item(),
        }

        return total, metrics

    def _compute_spatial_loss(
        self,
        feature_map: torch.Tensor,
        binary_labels: torch.Tensor,
        annotation_masks: list[Optional[torch.Tensor]],
        binary_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCELoss between CAM and downsampled annotation masks.
        Only for images that have annotations.
        """
        B, C, H, W = feature_map.shape
        losses = []

        for i in range(B):
            if annotation_masks[i] is None:
                continue

            mask = annotation_masks[i]  # (1, 1, H_img, W_img)
            # Downsample mask to feature map resolution (7x7)
            mask_ds = F.interpolate(
                mask.to(feature_map.device).float(), size=(H, W), mode="bilinear", align_corners=False
            )  # (1, 1, H, W)
            mask_ds = mask_ds.squeeze(0).squeeze(0)  # (H, W)

            # Class-weighted CAM: sum_c(w_c * A_c) where w_c from binary head for predicted class
            label = binary_labels[i].item()
            weights = binary_head_weight[label]  # (2048,)
            weights = weights.view(C, 1, 1)  # broadcast over spatial
            cam = (weights * feature_map[i]).sum(dim=0)  # (H, W)
            cam = F.relu(cam)

            # Normalize
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam_norm = (cam - cam_min) / (cam_max - cam_min)
            else:
                continue

            losses.append(self.bce(cam_norm, mask_ds.clamp(0, 1)))

        if not losses:
            return torch.tensor(0.0, device=feature_map.device)

        return torch.stack(losses).mean()


# ─── Rasterization helpers ─────────────────────────────────────────────────────

def rasterize_annotation(
    geometry_json: str,
    geometry_type: str,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Converts stored geometry (normalized [0,1] coords) to a binary mask of shape (H, W).
    """
    geom = json.loads(geometry_json)
    mask_img = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask_img)

    if geometry_type == "polygon":
        points = geom.get("points", [])
        if len(points) >= 3:
            pixel_pts = [(p["x"] * image_width, p["y"] * image_height) for p in points]
            draw.polygon(pixel_pts, fill=255)
    elif geometry_type == "brush":
        strokes = geom.get("strokes", [])
        radius = geom.get("radius", 0.02) * min(image_width, image_height)
        for pt in strokes:
            x, y = pt["x"] * image_width, pt["y"] * image_height
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    return np.array(mask_img, dtype=np.float32) / 255.0


def build_annotation_mask(
    annotations: list,  # list of Annotation ORM objects
    image_width: int,
    image_height: int,
) -> Optional[torch.Tensor]:
    """
    Merge all annotations for one image into a single binary mask tensor (1, 1, H, W).
    Returns None if no valid annotations.
    """
    if not annotations:
        return None

    combined = np.zeros((image_height, image_width), dtype=np.float32)
    for ann in annotations:
        m = rasterize_annotation(ann.geometry_json, ann.geometry_type, image_width, image_height)
        combined = np.maximum(combined, m)

    if combined.max() == 0:
        return None

    return torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
