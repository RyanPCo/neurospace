"""GradCAM implementation with thread-safe forward/backward hooks."""
import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from core.model.cnn import BrainTumorCNN
from config import TUMOR_CLASSES


def _apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply jet colormap to normalised [0,1] grayscale array → RGB uint8."""
    gray = np.clip(gray, 0, 1)
    r = np.clip(1.5 - abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * gray - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _to_base64_png(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class GradCAM:
    """
    Computes class activation maps for BrainTumorCNN using the last conv layer.
    Hooks are registered once and reused. Not thread-safe — use ModelManager's Lock.
    """

    def __init__(self, model: BrainTumorCNN):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.get_layer_for_gradcam()

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self._handles = [h1, h2]

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def compute(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
        original_size: tuple[int, int] | None = None,
    ) -> dict:
        """
        Args:
            image_tensor: (1, 1, 256, 256) normalised grayscale tensor on model device
            target_class: 0=glioma,1=meningioma,2=notumor,3=pituitary; None → predicted
            original_size: (width, height) to upsample heatmap to

        Returns dict with heatmap_b64, overlay_b64, top_kernel_indices, predicted_class, confidence
        """
        self.model.eval()
        self._activations = None
        self._gradients = None

        image_tensor = image_tensor.requires_grad_(False)

        out = self.model(image_tensor)
        logits = out["logits"]       # (1, 4)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM hooks did not fire.")

        # alpha: global-average-pool of gradients over spatial dims
        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, 128, 1, 1)
        cam = F.relu((alpha * self._activations).sum(dim=1, keepdim=True))  # (1, 1, 30, 30)

        # Upsample to original image size
        target_hw = (original_size[1], original_size[0]) if original_size else (256, 256)
        cam_up = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
        cam_np = cam_up[0, 0].cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        cam_norm = (cam_np - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam_np)

        # Top kernel indices (by gradient magnitude at last conv)
        kernel_importance = self._gradients[0].mean(dim=(1, 2)).abs().cpu().numpy()
        top_kernels = np.argsort(kernel_importance)[::-1][:10].tolist()

        gray_arr    = (cam_norm * 255).astype(np.uint8)
        colored_arr = _apply_jet_colormap(cam_norm)

        return {
            "heatmap_b64":       _to_base64_png(gray_arr),
            "overlay_b64":       _to_base64_png(colored_arr),
            "top_kernel_indices": top_kernels,
            "predicted_class":   TUMOR_CLASSES[target_class],
            "confidence":        float(probs[0, target_class].item()),
        }
