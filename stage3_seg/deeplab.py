from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabSegModel(nn.Module):
    """DeepLab wrapper aligned with CLIP-ES stage-3 training protocol."""

    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
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


class SegFormerSegModel(nn.Module):
    """SegFormer-B2 wrapper for optional ViT-family Stage3 backbone."""

    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.model = self._build_model(num_classes=num_classes, pretrained_backbone=pretrained_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def _build_model(self, num_classes: int, pretrained_backbone: bool) -> nn.Module:
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "SegFormer backbone requires transformers. "
                "Install it with: pip install transformers"
            ) from exc

        if pretrained_backbone:
            try:
                return SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b2-finetuned-ade-512-512",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
            except Exception:
                # Fallback to random init when pretrained checkpoint is unavailable offline.
                pass

        config = SegformerConfig(num_labels=num_classes)
        return SegformerForSemanticSegmentation(config)


def build_segmentation_model(
    architecture: str,
    num_classes: int,
    pretrained_backbone: bool = True,
) -> nn.Module:
    name = str(architecture).strip().lower()
    if name == "deeplabv3_resnet101":
        return DeepLabSegModel(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    if name in {"segformer_b2", "vit"}:
        return SegFormerSegModel(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    raise ValueError(f"Unsupported architecture: {architecture}")
