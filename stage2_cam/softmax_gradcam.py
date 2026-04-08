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
    patch_features: torch.Tensor
    text_feature: torch.Tensor


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

    def compute(
        self,
        image: torch.Tensor,
        class_index: int,
        text_features: torch.Tensor | None = None,
        tokenized_prompts: torch.Tensor | None = None,
    ) -> CAMResult:
        return self.compute_for_classes(
            image=image,
            text_features=text_features,
            tokenized_prompts=tokenized_prompts,
            class_indices=[class_index],
        )[0]

    def compute_for_classes(
        self,
        image: torch.Tensor,
        class_indices: List[int],
        text_features: torch.Tensor | None = None,
        tokenized_prompts: torch.Tensor | None = None,
        use_mutual_text: bool = True,
    ) -> List[CAMResult]:
        if not class_indices:
            return []
        if text_features is None and tokenized_prompts is None:
            raise ValueError("Either text_features or tokenized_prompts must be provided")

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
            if tokenized_prompts is not None:
                logits, text_features_batch = self.damp_wrapper.forward_logits_tokenized(
                    image=image,
                    tokenized_text=tokenized_prompts,
                    replace_cls_with_avg=self.replace_cls_with_avg,
                    use_mutual_text=use_mutual_text,
                )
            else:
                assert text_features is not None
                logits = self.damp_wrapper.forward_logits(
                    image,
                    text_features,
                    replace_cls_with_avg=self.replace_cls_with_avg,
                )
                normalized_text = F.normalize(text_features.to(self.damp_wrapper.device).float(), dim=-1)
                text_features_batch = normalized_text.unsqueeze(0).expand(logits.shape[0], -1, -1)

            if self.use_softmax:
                scores = F.softmax(logits, dim=-1)
            else:
                scores = logits

        affinity = self.damp_wrapper.get_attention_affinity(num_layers=8).detach().float()
        activation = self.damp_wrapper.get_feature_map()
        patch_features = activation[0].detach().float()

        out: List[CAMResult] = []
        for i, class_index in enumerate(class_indices):
            self.damp_wrapper.zero_grad()
            class_score = scores[:, class_index].sum()
            class_score.backward(retain_graph=(i < len(class_indices) - 1))

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

            out.append(
                CAMResult(
                    cam=cam.detach().float(),
                    affinity=affinity,
                    patch_features=patch_features,
                    text_feature=text_features_batch[0, class_index].detach().float(),
                )
            )

        return out
