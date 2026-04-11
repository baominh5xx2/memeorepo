from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class PromptBundle:
    class_names: List[str]
    background_names: List[str]
    full_names: List[str]
    full_phrases: List[str]
    tokenized_full: torch.Tensor
    class_to_full_indices: List[int]
    class_prompt_map: Dict[str, str]


class PromptManager:
    """Build CLIP text features for class and background prompts."""

    def __init__(
        self,
        template: str,
        class_synonyms: Dict[str, List[str]],
        background_names: List[str],
        use_sharpness_selection: bool = True,
        use_synonym_fusion: bool = True,
        extra_templates: List[str] | None = None,
        sharpness_eps: float = 1e-6,
    ):
        self.template = template
        self.class_synonyms = class_synonyms
        self.background_names = background_names
        self.use_sharpness_selection = bool(use_sharpness_selection)
        self.use_synonym_fusion = bool(use_synonym_fusion)
        self.extra_templates = list(extra_templates or [])
        self.sharpness_eps = float(sharpness_eps)

    def build(self, damp_wrapper) -> PromptBundle:
        class_names = list(self.class_synonyms.keys())
        class_prototypes = self._encode_phrases(
            damp_wrapper=damp_wrapper,
            phrases=class_names,
            template=self.template,
        )

        class_prompt_map: Dict[str, str] = {}
        class_prompts: List[str] = []
        for class_idx, canonical_name in enumerate(class_names):
            synonyms = list(self.class_synonyms[canonical_name])
            if canonical_name not in synonyms:
                synonyms.insert(0, canonical_name)
            candidates = self._build_candidates(canonical_name=canonical_name, synonyms=synonyms)
            if self.use_sharpness_selection and len(candidates) > 1:
                selected = self._select_by_sharpness(
                    damp_wrapper=damp_wrapper,
                    candidates=candidates,
                    class_prototypes=class_prototypes,
                    target_index=class_idx,
                )
            else:
                selected = candidates[0]
            class_prompt_map[canonical_name] = selected
            class_prompts.append(selected)

        background_prompts = [self.template.format(name) for name in self.background_names]
        full_phrases = class_prompts + background_prompts
        tokenized_full = damp_wrapper.tokenize(full_phrases)

        class_to_full_indices = list(range(len(class_names)))
        return PromptBundle(
            class_names=class_names,
            background_names=list(self.background_names),
            full_names=class_names + list(self.background_names),
            full_phrases=full_phrases,
            tokenized_full=tokenized_full,
            class_to_full_indices=class_to_full_indices,
            class_prompt_map=class_prompt_map,
        )

    def _encode_phrases(
        self,
        damp_wrapper,
        phrases: List[str],
        template: str,
    ) -> torch.Tensor:
        prompts = [template.format(phrase) for phrase in phrases]
        tokenized = damp_wrapper.tokenize(prompts)
        return damp_wrapper.encode_text(tokenized)

    def _build_candidates(self, canonical_name: str, synonyms: List[str]) -> List[str]:
        templates = [self.template, *self.extra_templates]
        templates = [tpl for tpl in templates if isinstance(tpl, str) and tpl.strip()]
        if not templates:
            templates = [self.template]

        candidates: List[str] = []
        for tpl in templates:
            for phrase in synonyms:
                candidates.append(tpl.format(phrase))
            if self.use_synonym_fusion and len(synonyms) > 1:
                fused = self._fuse_synonyms(canonical_name=canonical_name, synonyms=synonyms)
                candidates.append(tpl.format(fused))

        # Stable de-dup while preserving order.
        seen = set()
        unique: List[str] = []
        for prompt in candidates:
            if prompt not in seen:
                seen.add(prompt)
                unique.append(prompt)
        return unique

    def _fuse_synonyms(self, canonical_name: str, synonyms: List[str]) -> str:
        tail = [s for s in synonyms if s != canonical_name]
        if not tail:
            return canonical_name
        return f"{canonical_name} with {', '.join(tail)}"

    def _select_by_sharpness(
        self,
        damp_wrapper,
        candidates: List[str],
        class_prototypes: torch.Tensor,
        target_index: int,
    ) -> str:
        best_prompt = candidates[0]
        best_sharpness = float("inf")

        del target_index

        for candidate in candidates:
            tokenized = damp_wrapper.tokenize([candidate])
            feature = damp_wrapper.encode_text(tokenized)[0]
            similarities = feature @ class_prototypes.t()
            sharpness = similarities.var() / (similarities.abs().mean() + self.sharpness_eps)
            value = float(sharpness.item())
            if value < best_sharpness:
                best_sharpness = value
                best_prompt = candidate

        return best_prompt
