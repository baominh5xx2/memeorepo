from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from common.io import list_images
from tools.build_raw_smoke import build_raw_smoke


class TestBuildRawSmoke(unittest.TestCase):
    def test_build_raw_smoke_copies_training_and_labeled_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            out_root = tmp_path / "raw_smoke"

            self._build_minimal_hist(raw_root / "LUAD-HistoSeg" / "LUAD-HistoSeg")
            self._build_minimal_bcss(raw_root / "BCSS-WSSS" / "BCSS-WSSS")

            build_raw_smoke(
                raw_data_root=raw_root,
                output_root=out_root,
                n_train=2,
                n_val=1,
                n_test=1,
                overwrite=False,
            )

            hist_out = out_root / "LUAD-HistoSeg" / "LUAD-HistoSeg"
            bcss_out = out_root / "BCSS-WSSS" / "BCSS-WSSS"

            self.assertEqual(len(list_images(hist_out / "training")), 2)
            self.assertEqual(len(list_images(bcss_out / "training")), 2)

            self.assertEqual(len(list_images(hist_out / "val" / "img")), 1)
            self.assertEqual(len(list_images(hist_out / "val" / "mask")), 1)
            self.assertEqual(len(list_images(hist_out / "test" / "img")), 1)
            self.assertEqual(len(list_images(hist_out / "test" / "mask")), 1)

            self.assertEqual(len(list_images(bcss_out / "val" / "img")), 1)
            self.assertEqual(len(list_images(bcss_out / "val" / "mask")), 1)
            self.assertEqual(len(list_images(bcss_out / "test" / "img")), 1)
            self.assertEqual(len(list_images(bcss_out / "test" / "mask")), 1)

    def test_build_raw_smoke_overwrite_replaces_existing_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            out_root = tmp_path / "raw_smoke"

            self._build_minimal_hist(raw_root / "LUAD-HistoSeg" / "LUAD-HistoSeg")
            self._build_minimal_bcss(raw_root / "BCSS-WSSS" / "BCSS-WSSS")

            build_raw_smoke(
                raw_data_root=raw_root,
                output_root=out_root,
                n_train=2,
                n_val=1,
                n_test=1,
                overwrite=False,
            )

            stale_file = (
                out_root
                / "LUAD-HistoSeg"
                / "LUAD-HistoSeg"
                / "training"
                / "stale_file.png"
            )
            stale_file.write_bytes(b"stale")
            self.assertTrue(stale_file.exists())

            build_raw_smoke(
                raw_data_root=raw_root,
                output_root=out_root,
                n_train=1,
                n_val=1,
                n_test=1,
                overwrite=True,
            )

            self.assertFalse(stale_file.exists())
            hist_out = out_root / "LUAD-HistoSeg" / "LUAD-HistoSeg"
            self.assertEqual(len(list_images(hist_out / "training")), 1)

    def _build_minimal_hist(self, root: Path) -> None:
        (root / "training").mkdir(parents=True, exist_ok=True)
        (root / "val" / "img").mkdir(parents=True, exist_ok=True)
        (root / "val" / "mask").mkdir(parents=True, exist_ok=True)
        (root / "test" / "img").mkdir(parents=True, exist_ok=True)
        (root / "test" / "mask").mkdir(parents=True, exist_ok=True)

        for name in [
            "htrain-100-[1 0 0 1].png",
            "htrain-101-[1 0 1 1].png",
            "htrain-102-[0 1 0 1].png",
        ]:
            self._save_rgb(root / "training" / name, 64)

        for name in ["hval-0.png", "hval-1.png"]:
            self._save_rgb(root / "val" / "img" / name, 80)
            self._save_mask(root / "val" / "mask" / name, 1)

        for name in ["htest-0.png", "htest-1.png"]:
            self._save_rgb(root / "test" / "img" / name, 96)
            self._save_mask(root / "test" / "mask" / name, 2)

    def _build_minimal_bcss(self, root: Path) -> None:
        (root / "training").mkdir(parents=True, exist_ok=True)
        (root / "val" / "img").mkdir(parents=True, exist_ok=True)
        (root / "val" / "mask").mkdir(parents=True, exist_ok=True)
        (root / "test" / "img").mkdir(parents=True, exist_ok=True)
        (root / "test" / "mask").mkdir(parents=True, exist_ok=True)

        for name in [
            "btrain+0[1100].png",
            "btrain+1[1010].png",
            "btrain+2[0101].png",
        ]:
            self._save_rgb(root / "training" / name, 128)

        for name in ["bval+0.png", "bval+1.png"]:
            self._save_rgb(root / "val" / "img" / name, 160)
            self._save_mask(root / "val" / "mask" / name, 3)

        for name in ["btest+0.png", "btest+1.png"]:
            self._save_rgb(root / "test" / "img" / name, 192)
            self._save_mask(root / "test" / "mask" / name, 4)

    def _save_rgb(self, path: Path, value: int) -> None:
        Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(path)

    def _save_mask(self, path: Path, value: int) -> None:
        Image.fromarray(np.full((2, 2), value, dtype=np.uint8), mode="L").save(path)


if __name__ == "__main__":
    unittest.main()