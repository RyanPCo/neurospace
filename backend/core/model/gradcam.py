"""GradCAM implementation with thread-safe forward/backward hooks."""
import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from core.model.resnet import CancerScopeModel


def _apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply jet colormap to normalized [0,1] grayscale array → RGB uint8."""
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
    Computes class activation maps for a CancerScopeModel using the binary head.
    Hooks are registered once and reused. Not thread-safe — use ModelManager's Lock.
    """

    def __init__(self, model: CancerScopeModel):
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
            image_tensor: (1, 3, H, W) normalized tensor on model device
            target_class: 0=benign, 1=malignant; if None, uses predicted class
            original_size: (width, height) to upsample heatmap to

        Returns dict with:
            heatmap_b64: grayscale PNG base64
            overlay_b64: jet colormap PNG base64
            top_kernel_indices: list of int (top-10 most important kernel indices)
            predicted_class: "benign" or "malignant"
            confidence: float
        """
        self.model.eval()
        self._activations = None
        self._gradients = None

        image_tensor = image_tensor.requires_grad_(False)

        # Forward pass
        out = self.model(image_tensor)
        binary_logits = out["binary_logits"]  # (1, 2)
        probs = torch.softmax(binary_logits, dim=1)

        if target_class is None:
            target_class = int(binary_logits.argmax(dim=1).item())

        # Backward pass for target class
        self.model.zero_grad()
        score = binary_logits[0, target_class]
        score.backward()

        # Compute CAM
        # _gradients, _activations: (1, 2048, 7, 7)
        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM hooks did not fire — ensure model ran correctly.")

        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)
        cam = F.relu((alpha * self._activations).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)

        # Upsample
        target_hw = (original_size[1], original_size[0]) if original_size else (224, 224)
        cam_up = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
        cam_np = cam_up[0, 0].cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Top kernel indices (by gradient magnitude)
        kernel_importance = self._gradients[0].mean(dim=(1, 2)).abs().cpu().numpy()
        top_kernels = np.argsort(kernel_importance)[::-1][:10].tolist()

        # Generate outputs
        gray_arr = (cam_norm * 255).astype(np.uint8)
        colored_arr = _apply_jet_colormap(cam_norm)

        from core.model.resnet import BINARY_CLASSES
        return {
            "heatmap_b64": _to_base64_png(gray_arr),
            "overlay_b64": _to_base64_png(colored_arr),
            "top_kernel_indices": top_kernels,
            "predicted_class": BINARY_CLASSES[target_class],
            "confidence": float(probs[0, target_class].item()),
        }
