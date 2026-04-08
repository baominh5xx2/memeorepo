from __future__ import annotations

import torch
import torch.nn as nn


class DeepLabSegModel(nn.Module):
    """DeepLab wrapper aligned with CLIP-ES stage-3 training protocol."""

    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        self.model = self._build_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["out"]
        return out

    def _build_model(self, num_classes: int, pretrained_backbone: bool) -> nn.Module:
        from torchvision.models.segmentation import deeplabv3_resnet101

        try:
            from torchvision.models import ResNet101_Weights

            backbone_weights = (
                ResNet101_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            )
        except Exception:
            backbone_weights = None

        model = deeplabv3_resnet101(
            weights=None,
            weights_backbone=backbone_weights,
            aux_loss=False,
        )
        classifier_in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(classifier_in_channels, num_classes, kernel_size=1)
        return model
