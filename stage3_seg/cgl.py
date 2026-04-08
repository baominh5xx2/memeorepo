from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceGuidedLoss(nn.Module):
    """Ignore low-confidence pseudo labels during segmentation training."""

    def __init__(self, confidence_threshold: float = 0.25, ignore_index: int = 255):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if confidence is None:
            # Fallback: use CLIP-ES style confidence proxy max(max_fg, 1-max_fg).
            probs = F.softmax(logits.detach(), dim=1)
            if probs.shape[1] > 1:
                max_fg = probs[:, 1:, ...].amax(dim=1)
            else:
                max_fg = probs[:, 0, ...]
            confidence = torch.maximum(max_fg, 1.0 - max_fg)

        per_pixel_loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = (target != self.ignore_index) & (confidence >= self.confidence_threshold)
        valid_count = valid.sum().item()
        if valid_count == 0:
            return logits.sum() * 0.0

        return (per_pixel_loss * valid.float()).sum() / (valid.float().sum() + 1e-8)
