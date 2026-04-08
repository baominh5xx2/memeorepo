from __future__ import annotations

from typing import Dict

import numpy as np

from .constants import BACKGROUND_LABEL, DomainLabelSpec, IGNORE_LABEL


class MaskRemapper:
    """Remap dataset-specific mask encoding into {0,1,2,255}."""

    def __init__(self, spec: DomainLabelSpec):
        self._spec = spec

    def remap(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            return self._remap_indexed(mask)
        if mask.ndim == 3 and mask.shape[2] == 3:
            return self._remap_rgb(mask)
        raise ValueError(
            f"Unsupported mask shape for {self._spec.domain_name}: {mask.shape}"
        )

    def _remap_indexed(self, indexed_mask: np.ndarray) -> np.ndarray:
        out = np.full(indexed_mask.shape, IGNORE_LABEL, dtype=np.uint8)
        for source_value, target_value in self._spec.index_to_target.items():
            out[indexed_mask == source_value] = target_value
        # Unknown palette index is treated as background to avoid dropping too much area.
        out[indexed_mask > max(self._spec.index_to_target)] = BACKGROUND_LABEL
        return out

    def _remap_rgb(self, rgb_mask: np.ndarray) -> np.ndarray:
        out = np.full(rgb_mask.shape[:2], IGNORE_LABEL, dtype=np.uint8)
        for rgb, target in self._spec.rgb_to_target.items():
            color_mask = np.all(rgb_mask == np.array(rgb, dtype=np.uint8), axis=-1)
            out[color_mask] = target
        return out
