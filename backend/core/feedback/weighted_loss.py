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
    Loss = CE(logits, labels) + spatial_weight * L_spatial

    L_spatial aligns the mean feature-map activation with doctor-drawn masks.
    Two annotation types:
      - Regular (tumor class labels): BCE(mean_cam, mask)
      - GradCAM-focus ('gradcam_focus'): constrained loss
          L = -mean(cam * focus_mask) + 2 * mean(cam * (1 - focus_mask))
    Only computed for images that have annotations.
    """

    def __init__(self, spatial_weight: float = None):
        super().__init__()
        self.spatial_weight = spatial_weight or settings.spatial_loss_weight
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        annotation_masks: Optional[list[Optional[torch.Tensor]]] = None,
        focus_masks: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        task_loss = self.ce(outputs["logits"], labels)

        spatial_loss = torch.tensor(0.0, device=labels.device)
        if annotation_masks is not None or focus_masks is not None:
            spatial_loss = self._compute_spatial_loss(
                outputs["feature_map"], annotation_masks, focus_masks
            )

        total = task_loss + self.spatial_weight * spatial_loss

        return total, {
            "task_loss":    task_loss.item(),
            "spatial_loss": spatial_loss.item(),
            "total_loss":   total.item(),
        }

    def _compute_spatial_loss(
        self,
        feature_map: torch.Tensor,
        annotation_masks: Optional[list[Optional[torch.Tensor]]],
        focus_masks: Optional[list[Optional[torch.Tensor]]],
    ) -> torch.Tensor:
        B, C, H, W = feature_map.shape
        losses = []

        def _ds(mask):
            return F.interpolate(
                mask.to(feature_map.device).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0).clamp(0, 1)

        def _cam(i):
            """Mean-channel activation map for image i, normalised to [0,1]."""
            act = feature_map[i].mean(dim=0)    # (H, W)
            act = F.relu(act)
            mn, mx = act.min(), act.max()
            return (act - mn) / (mx - mn) if mx > mn else None

        for i in range(B):
            has_reg   = annotation_masks is not None and annotation_masks[i] is not None
            has_focus = focus_masks      is not None and focus_masks[i]      is not None
            if not has_reg and not has_focus:
                continue

            cam_norm = _cam(i)
            if cam_norm is None:
                continue

            if has_reg:
                mask_ds = _ds(annotation_masks[i])
                losses.append(self.bce(cam_norm, mask_ds))

            if has_focus:
                fm = _ds(focus_masks[i])
                l_inside  = -(cam_norm * fm).mean()
                l_outside = (cam_norm * (1.0 - fm)).mean()
                losses.append(l_inside + 2.0 * l_outside)

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
    annotations: list,
    image_width: int,
    image_height: int,
    label_class_filter: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Merge annotations into a single binary mask tensor (1, 1, H, W).
    Pass label_class_filter='gradcam_focus' to get only focus annotations.
    Without filter, gradcam_focus annotations are excluded from the regular mask.
    """
    if not annotations:
        return None

    combined = np.zeros((image_height, image_width), dtype=np.float32)
    any_drawn = False
    for ann in annotations:
        if label_class_filter is not None and ann.label_class != label_class_filter:
            continue
        if label_class_filter is None and ann.label_class == "gradcam_focus":
            continue
        m = rasterize_annotation(ann.geometry_json, ann.geometry_type, image_width, image_height)
        combined = np.maximum(combined, m)
        any_drawn = True

    if not any_drawn or combined.max() == 0:
        return None

    return torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)


def build_gradcam_focus_mask(
    annotations: list,
    image_width: int,
    image_height: int,
) -> Optional[torch.Tensor]:
    return build_annotation_mask(annotations, image_width, image_height,
                                  label_class_filter="gradcam_focus")
