from __future__ import annotations

import unittest

import numpy as np

from datasets.constants import BCSS_SPEC, LUAD_SPEC, IGNORE_LABEL
from datasets.label_mappers import MaskRemapper


class TestMaskRemapper(unittest.TestCase):
    def test_remap_indexed_mask_bcss(self) -> None:
        remapper = MaskRemapper(BCSS_SPEC)
        mask = np.array([[0, 1, 2, 3, 4, 9]], dtype=np.uint8)
        out = remapper.remap(mask)
        expected = np.array([[1, 2, 255, 255, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_remap_rgb_mask_luad(self) -> None:
        remapper = MaskRemapper(LUAD_SPEC)
        mask = np.array(
            [
                [[205, 51, 51], [255, 165, 0]],
                [[0, 255, 0], [1, 2, 3]],
            ],
            dtype=np.uint8,
        )
        out = remapper.remap(mask)
        expected = np.array([[1, 2], [255, IGNORE_LABEL]], dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_invalid_shape_raises(self) -> None:
        remapper = MaskRemapper(BCSS_SPEC)
        with self.assertRaises(ValueError):
            remapper.remap(np.zeros((4,), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
