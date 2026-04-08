from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass
class CRFParams:
    iter_max: int = 10
    pos_w: int = 3
    pos_xy_std: int = 1
    bi_w: int = 5
    bi_xy_std: int = 80
    bi_rgb_std: int = 13


class DenseCRFRefiner:
    """DenseCRF post-processing with graceful fallback when dependency is missing."""

    def __init__(self, params: CRFParams):
        self.params = params
        self._ready = False
        self._dcrf = None
        self._utils = None
        self._init_backend()

    def refine(self, image_rgb: np.ndarray, probs: np.ndarray) -> np.ndarray:
        if not self._ready:
            return probs

        c, h, w = probs.shape
        unary = self._utils.unary_from_softmax(probs)
        unary = np.ascontiguousarray(unary)
        image_rgb = np.ascontiguousarray(image_rgb)

        dense_crf = self._dcrf.DenseCRF2D(w, h, c)
        dense_crf.setUnaryEnergy(unary)
        dense_crf.addPairwiseGaussian(
            sxy=self.params.pos_xy_std,
            compat=self.params.pos_w,
        )
        dense_crf.addPairwiseBilateral(
            sxy=self.params.bi_xy_std,
            srgb=self.params.bi_rgb_std,
            rgbim=image_rgb,
            compat=self.params.bi_w,
        )
        out = dense_crf.inference(self.params.iter_max)
        return np.array(out).reshape((c, h, w))

    def _init_backend(self) -> None:
        try:
            import pydensecrf.densecrf as dcrf
            import pydensecrf.utils as utils
        except ImportError:
            self._ready = False
            warnings.warn(
                "pydensecrf is not installed. CRF refinement is disabled and pseudo-mask quality may drop.",
                RuntimeWarning,
            )
            return

        self._dcrf = dcrf
        self._utils = utils
        self._ready = True
