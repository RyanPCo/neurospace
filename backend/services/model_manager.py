"""Singleton model manager with asyncio lock for thread-safe GradCAM."""
import asyncio
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from config import settings, TUMOR_CLASSES
from core.model.cnn import BrainTumorCNN, load_model
from core.model.gradcam import GradCAM
from core.dataset.transforms import inference_transforms


def _disable_inplace_relu(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.ReLU) and m.inplace:
            m.inplace = False


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.model: Optional[BrainTumorCNN] = None
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
        candidate_paths: list[Optional[str]] = []
        if model_path is not None:
            candidate_paths.append(model_path)
        else:
            tuned = settings.models_dir / "best_model_expl_tuned.pt"
            if tuned.exists():
                candidate_paths.append(str(tuned))
            if settings.best_model_path.exists():
                candidate_paths.append(str(settings.best_model_path))
            if not candidate_paths:
                candidate_paths.append(None)

        last_error: Exception | None = None
        for path in candidate_paths:
            try:
                self.model = load_model(path=path, device=self.device)
                _disable_inplace_relu(self.model)
                self.model.eval()

                if self.gradcam:
                    self.gradcam.remove_hooks()
                self.gradcam = GradCAM(self.model)

                self.model_version = Path(path).stem if path else "untrained"
                self._prediction_cache.clear()
                return
            except Exception as e:
                last_error = e

        raise RuntimeError(f"Failed to load model from candidates: {candidate_paths}. Last error: {last_error}")

    def ensure_loaded(self):
        if self.model is None:
            self.load()

    async def get_gradcam(self, image_path: str, image_id: str) -> dict:
        self.ensure_loaded()
        async with self._lock:
            img = Image.open(image_path).convert("L")   # grayscale
            w, h = img.size
            tensor = inference_transforms(img).unsqueeze(0).to(self.device)
            result = self.gradcam.compute(tensor, original_size=(w, h))
            result["image_id"] = image_id
            return result

    def predict(self, image_path: str, image_id: str, use_cache: bool = True) -> dict:
        self.ensure_loaded()

        if use_cache and image_id in self._prediction_cache:
            return self._prediction_cache[image_id]

        img = Image.open(image_path).convert("L")
        tensor = inference_transforms(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)

        probs = torch.softmax(outputs["logits"], dim=1)[0].cpu()
        predicted_idx = int(probs.argmax().item())
        predicted_class = TUMOR_CLASSES[predicted_idx]
        confidence = float(probs[predicted_idx].item())

        result = {
            "image_id":        image_id,
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "class_probs":     {c: float(probs[i]) for i, c in enumerate(TUMOR_CLASSES)},
            "subtype_predicted": None,
            "subtype_probs":   {},
            "model_version":   self.model_version,
        }

        if use_cache:
            self._prediction_cache[image_id] = result

        return result

    def reload(self, model_path: Optional[str] = None):
        self.load(model_path)

    def is_loaded(self) -> bool:
        return self.model is not None


def get_model_manager() -> ModelManager:
    return ModelManager.get_instance()
