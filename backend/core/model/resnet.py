"""CancerScopeModel — Dual-head ResNet-50 for binary + subtype classification."""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from config import settings


BINARY_CLASSES = ["benign", "malignant"]
SUBTYPE_CLASSES = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]


class CancerScopeModel(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = None):
        super().__init__()
        dropout = dropout or settings.dropout_rate

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # Keep all layers except avgpool and fc as feature extractor
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=dropout)
        self.binary_head = nn.Linear(2048, 2)
        self.subtype_head = nn.Linear(2048, settings.num_subtypes)

        # Stored for GradCAM
        self._feature_map: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # features shape: (B, 2048, 7, 7) for 224x224 input
        feature_map = self.features(x)
        pooled = self.avgpool(feature_map).flatten(1)  # (B, 2048)
        dropped = self.dropout(pooled)

        binary_logits = self.binary_head(dropped)
        subtype_logits = self.subtype_head(dropped)

        return {
            "binary_logits": binary_logits,
            "subtype_logits": subtype_logits,
            "feature_map": feature_map,
        }

    def get_layer_for_gradcam(self) -> nn.Module:
        """Returns layer4 (last convolutional block) for GradCAM hooks."""
        return self.features[-1]  # layer4

    def apply_kernel_deletions(self, deleted_kernel_ids: list[str]) -> int:
        """
        Zero out filter weights for soft-deleted kernels.
        kernel_id format: "{layer_name}_{filter_index}", e.g. "layer4.2.conv3_127"
        Returns count of deleted kernels applied.
        """
        count = 0
        for kid in deleted_kernel_ids:
            # parse: everything before the last underscore is layer path, last part is index
            parts = kid.rsplit("_", 1)
            if len(parts) != 2:
                continue
            layer_path, idx_str = parts
            try:
                filter_idx = int(idx_str)
            except ValueError:
                continue

            # Traverse the module tree by attribute path
            # layer_path looks like "layer4.2.conv3"
            module = self
            for attr in layer_path.split("."):
                module = getattr(module, attr, None)
                if module is None:
                    break
            if module is None or not hasattr(module, "weight"):
                continue

            with torch.no_grad():
                if filter_idx < module.weight.shape[0]:
                    module.weight[filter_idx].zero_()
                    if module.bias is not None:
                        module.bias[filter_idx].zero_()
                    count += 1

        return count


def load_model(path: str | None = None, device: str | None = None) -> CancerScopeModel:
    """Load model from checkpoint or create fresh pretrained model."""
    dev = device or settings.device
    model = CancerScopeModel(pretrained=(path is None))
    if path:
        state = torch.load(path, map_location=dev)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    model = model.to(dev)
    return model
