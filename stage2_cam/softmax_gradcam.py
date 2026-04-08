from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class CAMResult:
    cam: torch.Tensor
    affinity: torch.Tensor


class SoftmaxGradCAM:
    """Softmax-GradCAM for CLIP-like logits with token-space affinity estimation."""

    def __init__(
        self,
        damp_wrapper,
        use_softmax: bool = True,
        replace_cls_with_avg: bool = True,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        self.damp_wrapper = damp_wrapper
        self.use_softmax = use_softmax
        self.replace_cls_with_avg = replace_cls_with_avg
        self.amp_enabled = bool(amp_enabled)
        self.amp_dtype = amp_dtype

    def compute(self, image: torch.Tensor, text_features: torch.Tensor, class_index: int) -> CAMResult:
        return self.compute_for_classes(
            image=image,
            text_features=text_features,
            class_indices=[class_index],
        )[0]

    def compute_for_classes(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
        class_indices: List[int],
    ) -> List[CAMResult]:
        if not class_indices:
            return []

        image = image.to(self.damp_wrapper.device)
        image.requires_grad_(True)
        self.damp_wrapper.zero_grad()

        use_amp = bool(self.amp_enabled and self.damp_wrapper.device.type == "cuda")
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if use_amp
            else nullcontext()
        )

        with autocast_ctx:
            logits = self.damp_wrapper.forward_logits(
                image,
                text_features,
                replace_cls_with_avg=self.replace_cls_with_avg,
            )
            if self.use_softmax:
                scores = F.softmax(logits, dim=-1)
            else:
                scores = logits

        affinity = self.damp_wrapper.get_attention_affinity(num_layers=8).detach().float()
        out: List[CAMResult] = []
        for i, class_index in enumerate(class_indices):
            self.damp_wrapper.zero_grad()
            class_score = scores[:, class_index].sum()
            class_score.backward(retain_graph=(i < len(class_indices) - 1))

            activation = self.damp_wrapper.get_feature_map()
            gradient = self.damp_wrapper.get_feature_gradient()

            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            cam = cam[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            out.append(CAMResult(cam=cam.detach().float(), affinity=affinity))

        return out
