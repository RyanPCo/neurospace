"""Singleton model manager with asyncio lock for thread-safe GradCAM."""
import asyncio
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from config import settings
from core.model.resnet import CancerScopeModel, load_model, BINARY_CLASSES, SUBTYPE_CLASSES
from core.model.gradcam import GradCAM
from core.dataset.transforms import inference_transforms


class ModelManager:
    """
    Singleton that holds the loaded model and GradCAM instance.
    Uses an asyncio.Lock to prevent concurrent GradCAM calls.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.model: Optional[CancerScopeModel] = None
        self.gradcam: Optional[GradCAM] = None
        self.device: str = settings.device
        self.model_version: str = "none"
        self._lock = asyncio.Lock()
        self._prediction_cache: dict[str, dict] = {}

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, model_path: Optional[str] = None):
        """Load model from path (or best_model.pt if exists, else pretrained)."""
        path = model_path
        if path is None and settings.best_model_path.exists():
            path = str(settings.best_model_path)

        self.model = load_model(path=path, device=self.device)
        self.model.eval()

        if self.gradcam:
            self.gradcam.remove_hooks()
        self.gradcam = GradCAM(self.model)

        if path:
            self.model_version = Path(path).stem
        else:
            self.model_version = "pretrained_base"

        self._prediction_cache.clear()

    def ensure_loaded(self):
        if self.model is None:
            self.load()

    async def get_gradcam(self, image_path: str, image_id: str) -> dict:
        """Thread-safe GradCAM computation."""
        self.ensure_loaded()
        async with self._lock:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            tensor = inference_transforms(img).unsqueeze(0).to(self.device)
            result = self.gradcam.compute(tensor, original_size=(w, h))
            result["image_id"] = image_id
            return result

    def predict(self, image_path: str, image_id: str, use_cache: bool = True) -> dict:
        """Run inference and return prediction dict."""
        self.ensure_loaded()

        if use_cache and image_id in self._prediction_cache:
            return self._prediction_cache[image_id]

        img = Image.open(image_path).convert("RGB")
        tensor = inference_transforms(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)

        binary_probs = torch.softmax(outputs["binary_logits"], dim=1)[0].cpu()
        subtype_probs = torch.softmax(outputs["subtype_logits"], dim=1)[0].cpu()

        predicted_idx = binary_probs.argmax().item()
        predicted_class = BINARY_CLASSES[predicted_idx]
        confidence = float(binary_probs[predicted_idx].item())

        subtype_idx = subtype_probs.argmax().item()
        subtype_predicted = SUBTYPE_CLASSES[subtype_idx]

        result = {
            "image_id": image_id,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probs": {c: float(binary_probs[i]) for i, c in enumerate(BINARY_CLASSES)},
            "subtype_predicted": subtype_predicted,
            "subtype_probs": {c: float(subtype_probs[i]) for i, c in enumerate(SUBTYPE_CLASSES)},
            "model_version": self.model_version,
        }

        if use_cache:
            self._prediction_cache[image_id] = result

        return result

    def reload(self, model_path: Optional[str] = None):
        """Reload model (called after retraining). Clears prediction cache."""
        self.load(model_path)

    def is_loaded(self) -> bool:
        return self.model is not None


# Global singleton accessor
def get_model_manager() -> ModelManager:
    return ModelManager.get_instance()
