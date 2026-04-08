from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class PromptBundle:
    class_names: List[str]
    background_names: List[str]
    full_names: List[str]
    class_features: torch.Tensor
    background_features: torch.Tensor
    full_features: torch.Tensor
    class_to_full_indices: List[int]


class PromptManager:
    """Build CLIP text features for class and background prompts."""

    def __init__(
        self,
        template: str,
        class_synonyms: Dict[str, List[str]],
        background_names: List[str],
    ):
        self.template = template
        self.class_synonyms = class_synonyms
        self.background_names = background_names

    def build(self, damp_wrapper) -> PromptBundle:
        device = damp_wrapper.device

        class_names = list(self.class_synonyms.keys())
        class_features = []
        for canonical_name in class_names:
            synonyms = self.class_synonyms[canonical_name]
            synonym_features = self._encode_phrases(damp_wrapper, synonyms, device)
            synonym_features = synonym_features / synonym_features.norm(dim=-1, keepdim=True)
            averaged = synonym_features.mean(dim=0)
            averaged = averaged / averaged.norm()
            class_features.append(averaged)

        class_features_tensor = torch.stack(class_features, dim=0)
        bg_features_tensor = self._encode_phrases(damp_wrapper, self.background_names, device)
        bg_features_tensor = bg_features_tensor / bg_features_tensor.norm(dim=-1, keepdim=True)

        full_features = torch.cat([class_features_tensor, bg_features_tensor], dim=0)
        full_features = full_features / full_features.norm(dim=-1, keepdim=True)

        class_to_full_indices = list(range(len(class_names)))
        return PromptBundle(
            class_names=class_names,
            background_names=list(self.background_names),
            full_names=class_names + list(self.background_names),
            class_features=class_features_tensor,
            background_features=bg_features_tensor,
            full_features=full_features,
            class_to_full_indices=class_to_full_indices,
        )

    def _encode_phrases(self, damp_wrapper, phrases: List[str], device: torch.device) -> torch.Tensor:
        prompts = [self.template.format(phrase) for phrase in phrases]
        tokenized = damp_wrapper.tokenize(prompts).to(device)
        return damp_wrapper.encode_text(tokenized)
