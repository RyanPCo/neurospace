"""KernelAnalyzer: filter visualization, importance scoring, top-activating images."""
import base64
import io
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import settings
from core.model.cnn import BrainTumorCNN


# Maps layer_name (kernel_id prefix) → dot-path to the Conv2d module
ANALYZED_LAYERS = {
    "conv_block1": "conv_block1.0",   # Conv2d(1,  32, 3)  — 32 filters
    "conv_block2": "conv_block2.0",   # Conv2d(32, 64, 3)  — 64 filters
    "conv_block3": "conv_block3.0",   # Conv2d(64, 128, 3) — 128 filters
}


def _get_module_by_path(model: BrainTumorCNN, path: str) -> torch.nn.Module | None:
    module = model
    for attr in path.split("."):
        if attr.isdigit():
            try:
                module = module[int(attr)]
            except (IndexError, TypeError):
                return None
        else:
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
    def __init__(self, model: BrainTumorCNN, device: str | None = None):
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

        # Average across all input channels → signed 2D weight map
        w_mean = weight.mean(dim=0).numpy()  # (kH, kW)

        # Normalize to [-1, 1] preserving sign (diverging scale)
        abs_max = np.abs(w_mean).max()
        w_norm = w_mean / abs_max if abs_max > 0 else np.zeros_like(w_mean)

        # Map [-1, 1] → [0, 1] for RdBu_r: blue=inhibitory(<0), red=excitatory(>0)
        w_01 = (w_norm + 1.0) / 2.0
        rgba = cm.get_cmap("RdBu_r")(w_01)  # (kH, kW, 4)
        rgb_np = (rgba[:, :, :3] * 255).astype(np.uint8)

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
