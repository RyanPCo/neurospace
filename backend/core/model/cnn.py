"""BrainTumorCNN — 3-block grayscale CNN for 4-class brain tumor MRI classification.

Architecture mirrors the Keras notebook exactly:
  Conv(32,3) → MaxPool(2)       feature map: (B, 32, 127, 127)
  Conv(64,3) → MaxPool(2)       feature map: (B, 64,  62,  62)
  Conv(128,3) → MaxPool(2)      feature map: (B, 128, 30,  30)  ← GradCAM target
  Flatten → Linear(115200,128) → ReLU → Dropout(0.5) → Linear(128,4)
"""
import torch
import torch.nn as nn
from config import settings, TUMOR_CLASSES


# Spatial size of the last feature map for 256×256 input (no padding)
# 256 → conv(3) → 254 → pool(2) → 127
# 127 → conv(3) → 125 → pool(2) → 62
#  62 → conv(3) →  60 → pool(2) → 30
_LAST_FM_SIZE = 30
_FLAT_DIM = 128 * _LAST_FM_SIZE * _LAST_FM_SIZE  # 115 200


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int = None, dropout: float = None):
        super().__init__()
        num_classes = num_classes or settings.num_classes
        dropout = dropout or settings.dropout_rate

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_FLAT_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, 256, 256) normalised grayscale tensor
        Returns:
            logits: (B, num_classes)
            feature_map: (B, 128, 30, 30) — last conv output before flatten
        """
        fm1 = self.conv_block1(x)         # (B, 32, 127, 127)
        fm2 = self.conv_block2(fm1)        # (B, 64, 62, 62)
        fm3 = self.conv_block3(fm2)        # (B, 128, 30, 30)
        logits = self.classifier(fm3)      # (B, 4)
        return {"logits": logits, "feature_map": fm3}

    def get_layer_for_gradcam(self) -> nn.Module:
        """Returns the last conv layer (conv_block3.conv) for GradCAM hooks."""
        return self.conv_block3[0]  # nn.Conv2d(64, 128, 3)

    def apply_kernel_deletions(self, deleted_kernel_ids: list[str]) -> int:
        """
        Zero out filter weights for soft-deleted kernels.
        kernel_id format: "{layer_name}_{filter_index}" e.g. "conv_block3_42"
        The layer_name is a key in ANALYZED_LAYERS (kernel_analyzer.py).
        """
        from core.model.kernel_analyzer import ANALYZED_LAYERS

        count = 0
        for kid in deleted_kernel_ids:
            parts = kid.rsplit("_", 1)
            if len(parts) != 2:
                continue
            layer_name, idx_str = parts
            try:
                filter_idx = int(idx_str)
            except ValueError:
                continue

            module_path = ANALYZED_LAYERS.get(layer_name)
            if module_path is None:
                continue

            module = _traverse(self, module_path)
            if module is None or not hasattr(module, "weight"):
                continue

            with torch.no_grad():
                if filter_idx < module.weight.shape[0]:
                    module.weight[filter_idx].zero_()
                    if module.bias is not None:
                        module.bias[filter_idx].zero_()
                    count += 1

        return count


def _traverse(model: nn.Module, path: str) -> nn.Module | None:
    """Walk a dot-separated path; handles integer indices for Sequential."""
    m = model
    for attr in path.split("."):
        if attr.isdigit():
            try:
                m = m[int(attr)]
            except (IndexError, TypeError):
                return None
        else:
            m = getattr(m, attr, None)
            if m is None:
                return None
    return m


def load_model(path: str | None = None, device: str | None = None) -> BrainTumorCNN:
    """Load model from checkpoint, or create a fresh untrained model."""
    dev = device or settings.device
    model = BrainTumorCNN()
    if path:
        state = torch.load(path, map_location=dev, weights_only=False)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
    model = model.to(dev)
    return model
