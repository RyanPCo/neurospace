"""KernelAnalyzer: filter visualization, importance scoring, top-activating images."""
import base64
import io
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import settings
from core.model.resnet import CancerScopeModel


ANALYZED_LAYERS = {
    "layer4.0.conv1": "features.7.0.conv1",
    "layer4.0.conv2": "features.7.0.conv2",
    "layer4.0.conv3": "features.7.0.conv3",
    "layer4.1.conv1": "features.7.1.conv1",
    "layer4.1.conv2": "features.7.1.conv2",
    "layer4.1.conv3": "features.7.1.conv3",
    "layer4.2.conv1": "features.7.2.conv1",
    "layer4.2.conv2": "features.7.2.conv2",
    "layer4.2.conv3": "features.7.2.conv3",
}


def _get_module_by_path(model: CancerScopeModel, path: str) -> torch.nn.Module | None:
    module = model
    for attr in path.split("."):
        module = getattr(module, attr, None)
        if module is None:
            return None
    return module


def _arr_to_png_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class KernelAnalyzer:
    def __init__(self, model: CancerScopeModel, device: str | None = None):
        self.model = model
        self.device = device or settings.device

    def visualize_filter(self, layer_name: str, filter_index: int) -> bytes:
        """
        Returns raw PNG bytes for the filter visualization (128×128).
        Extracts weight[filter_index] from the named layer.
        """
        module_path = ANALYZED_LAYERS.get(layer_name)
        if module_path is None:
            raise ValueError(f"Layer {layer_name!r} not in ANALYZED_LAYERS")

        module = _get_module_by_path(self.model, module_path)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"Module at {module_path} has no weight")

        weight = module.weight.data[filter_index].cpu()  # (C_in, kH, kW)
        c_in, kH, kW = weight.shape

        if c_in >= 3:
            rgb = weight[:3]  # (3, kH, kW)
        else:
            rgb = weight.mean(0, keepdim=True).expand(3, -1, -1)

        # Normalize to [0, 255]
        mn, mx = rgb.min(), rgb.max()
        if mx > mn:
            rgb = (rgb - mn) / (mx - mn)
        else:
            rgb = torch.zeros_like(rgb)

        rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(rgb_np).resize((128, 128), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def compute_importance_scores(
        self, dataloader: DataLoader, layer_name: str
    ) -> dict[int, float]:
        """
        Runs the validation set through the model and computes per-filter importance
        as mean(spatial_max(activation)) over all images.
        Returns {filter_index: importance_score}.
        """
        module_path = ANALYZED_LAYERS.get(layer_name)
        if module_path is None:
            raise ValueError(f"Layer {layer_name!r} not in ANALYZED_LAYERS")

        module = _get_module_by_path(self.model, module_path)
        if module is None:
            raise ValueError(f"Module not found at {module_path}")

        accumulated: dict[int, float] = {}
        count = 0
        activations_storage: list[torch.Tensor] = []

        def hook(m, inp, out):
            activations_storage.append(out.detach().cpu())

        handle = module.register_forward_hook(hook)
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                activations_storage.clear()
                self.model(images)
                if not activations_storage:
                    continue
                acts = activations_storage[0]  # (B, C, H, W)
                # spatial_max per filter: (B, C)
                spatial_max = acts.flatten(2).max(dim=2).values
                # mean over batch
                mean_per_filter = spatial_max.mean(dim=0)  # (C,)
                for i, val in enumerate(mean_per_filter.tolist()):
                    accumulated[i] = accumulated.get(i, 0.0) + val
                count += 1

        handle.remove()

        if count == 0:
            return {}

        return {i: v / count for i, v in accumulated.items()}

    def get_top_activating_images(
        self, dataloader: DataLoader, layer_name: str, filter_index: int, top_k: int = 9
    ) -> list[dict]:
        """
        Returns top-k images with highest activation for a given filter.
        Each entry: {image_id, max_activation, activation_map_b64}
        """
        module_path = ANALYZED_LAYERS.get(layer_name)
        if module_path is None:
            raise ValueError(f"Layer {layer_name!r} not in ANALYZED_LAYERS")

        module = _get_module_by_path(self.model, module_path)
        if module is None:
            raise ValueError(f"Module not found at {module_path}")

        records: list[tuple[float, str, np.ndarray]] = []  # (score, image_id, act_map)
        activations_storage: list[torch.Tensor] = []

        def hook(m, inp, out):
            activations_storage.append(out.detach().cpu())

        handle = module.register_forward_hook(hook)
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                images, _, image_ids = batch[0], batch[1], batch[2]
                images = images.to(self.device)
                activations_storage.clear()
                self.model(images)
                if not activations_storage:
                    continue
                acts = activations_storage[0]  # (B, C, H, W)
                filter_acts = acts[:, filter_index, :, :]  # (B, H, W)

                for i in range(filter_acts.shape[0]):
                    act_map = filter_acts[i].numpy()  # (H, W)
                    max_act = float(act_map.max())
                    # Normalize
                    mn, mx = act_map.min(), act_map.max()
                    if mx > mn:
                        act_norm = (act_map - mn) / (mx - mn)
                    else:
                        act_norm = np.zeros_like(act_map)
                    act_img = (act_norm * 255).astype(np.uint8)
                    records.append((max_act, image_ids[i], act_img))

        handle.remove()

        records.sort(key=lambda x: x[0], reverse=True)
        top = records[:top_k]

        return [
            {
                "image_id": rec[1],
                "max_activation": rec[0],
                "activation_map_b64": _arr_to_png_b64(rec[2]),
            }
            for rec in top
        ]
