from __future__ import annotations

import torch
import torch.nn.functional as F


class BidirectionalCoAttentionRefiner:
    """Parameter-free bidirectional text-pixel matching for CAM refinement."""

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        blend: float = 0.5,
        temperature: float = 0.07,
        eps: float = 1e-8,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.blend = float(blend)
        self.temperature = float(max(temperature, eps))
        self.eps = float(eps)

    def refine(
        self,
        cam_2d: torch.Tensor,
        patch_features: torch.Tensor,
        text_feature: torch.Tensor,
    ) -> torch.Tensor:
        cam_2d = cam_2d.float()
        patch_features = patch_features.float()
        text_feature = text_feature.float()

        h, w = cam_2d.shape
        if patch_features.ndim != 3:
            raise ValueError(f"Expected patch_features in [C,H,W], got {patch_features.shape}")

        if patch_features.shape[1] != h or patch_features.shape[2] != w:
            patch_features = F.interpolate(
                patch_features.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        c = patch_features.shape[0]
        patches = patch_features.reshape(c, h * w).t()  # [HW, C]
        patches = F.normalize(patches, dim=-1)

        text = text_feature.reshape(1, -1)
        if text.shape[-1] != c:
            raise ValueError(f"Text dim {text.shape[-1]} does not match patch dim {c}")
        text = F.normalize(text, dim=-1)

        # Text-to-pixel cross attention.
        text_to_pixel = torch.matmul(patches, text.t()).squeeze(1) / self.temperature
        text_to_pixel = torch.sigmoid(text_to_pixel)

        # Pixel-to-text cross attention via attended visual summary.
        weights = torch.softmax(text_to_pixel, dim=0)
        visual_summary = torch.matmul(weights.unsqueeze(0), patches).squeeze(0)
        visual_summary = F.normalize(visual_summary, dim=0)
        pixel_to_text = torch.matmul(patches, visual_summary.unsqueeze(1)).squeeze(1)
        pixel_to_text = torch.sigmoid(pixel_to_text)

        match = self.alpha * text_to_pixel + self.beta * pixel_to_text
        match = match.reshape(h, w)
        match = (match - match.min()) / (match.max() - match.min() + self.eps)

        cam_norm = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + self.eps)
        refined = self.blend * cam_norm + (1.0 - self.blend) * match
        refined = (refined - refined.min()) / (refined.max() - refined.min() + self.eps)
        return refined
