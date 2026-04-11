from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.constants import BCSS_SPEC, LUAD_SPEC
from datasets.label_mappers import MaskRemapper
from tools.prepare_medical_datasets import (
    DomainPaths,
    DomainPreprocessor,
    WeakLabelParser,
    _build_domain_paths,
)


class TestPrepareMedicalDatasets(unittest.TestCase):
    def test_weak_label_parser(self) -> None:
        parser = WeakLabelParser()
        self.assertEqual(parser.parse("foo+0[1101]"), (1, 1, 0, 1))
        self.assertEqual(parser.parse("1031280-2300-27920-[1 0 0 1]"), (1, 0, 0, 1))

    def test_build_domain_paths(self) -> None:
        raw_root = Path("raw")
        out_root = Path("out")
        paths = _build_domain_paths(raw_root, out_root)

        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0].name, "Hist")
        self.assertEqual(paths[1].name, "BCSS")
        self.assertEqual(paths[0].raw_root, raw_root / "LUAD-HistoSeg" / "LUAD-HistoSeg")
        self.assertEqual(paths[1].raw_root, raw_root / "BCSS-WSSS" / "BCSS-WSSS")

    def test_domain_preprocessor_run_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            out_root = tmp_path / "out"

            hist_paths = DomainPaths(
                name="Hist",
                raw_root=raw_root / "LUAD-HistoSeg" / "LUAD-HistoSeg",
                output_root=out_root / "Hist",
            )
            bcss_paths = DomainPaths(
                name="BCSS",
                raw_root=raw_root / "BCSS-WSSS" / "BCSS-WSSS",
                output_root=out_root / "BCSS",
            )

            self._build_minimal_hist_raw(hist_paths.raw_root)
            self._build_minimal_bcss_raw(bcss_paths.raw_root)

            DomainPreprocessor(hist_paths, MaskRemapper(LUAD_SPEC)).run()
            DomainPreprocessor(bcss_paths, MaskRemapper(BCSS_SPEC)).run()

            self.assertTrue((hist_paths.out_splits_dir / "train.txt").exists())
            self.assertTrue((hist_paths.out_splits_dir / "val.txt").exists())
            self.assertTrue((hist_paths.out_splits_dir / "test.txt").exists())
            self.assertTrue((bcss_paths.out_splits_dir / "train.txt").exists())

            hist_val_id = (hist_paths.out_splits_dir / "val.txt").read_text(encoding="utf-8").strip()
            bcss_val_id = (bcss_paths.out_splits_dir / "val.txt").read_text(encoding="utf-8").strip()
            hist_mask = np.array(Image.open(hist_paths.out_masks_dir / f"{hist_val_id}.png"), dtype=np.uint8)
            bcss_mask = np.array(Image.open(bcss_paths.out_masks_dir / f"{bcss_val_id}.png"), dtype=np.uint8)

            np.testing.assert_array_equal(hist_mask, np.array([[1, 2], [255, 255]], dtype=np.uint8))
            np.testing.assert_array_equal(bcss_mask, np.array([[1, 2], [255, 0]], dtype=np.uint8))

            with (hist_paths.out_meta_dir / "train_weak_labels.csv").open("r", encoding="utf-8") as f:
                hist_rows = list(csv.DictReader(f))
            with (bcss_paths.out_meta_dir / "train_weak_labels.csv").open("r", encoding="utf-8") as f:
                bcss_rows = list(csv.DictReader(f))

            self.assertEqual(len(hist_rows), 1)
            self.assertEqual(hist_rows[0]["tumor"], "1")
            self.assertEqual(hist_rows[0]["stroma"], "1")
            self.assertEqual(hist_rows[0]["necrosis"], "0")

            self.assertEqual(len(bcss_rows), 1)
            self.assertEqual(bcss_rows[0]["tumor"], "1")
            self.assertEqual(bcss_rows[0]["stroma"], "1")
            self.assertEqual(bcss_rows[0]["necrosis"], "0")

    def _build_minimal_hist_raw(self, root: Path) -> None:
        (root / "training").mkdir(parents=True, exist_ok=True)
        (root / "val" / "img").mkdir(parents=True, exist_ok=True)
        (root / "val" / "mask").mkdir(parents=True, exist_ok=True)
        (root / "test" / "img").mkdir(parents=True, exist_ok=True)
        (root / "test" / "mask").mkdir(parents=True, exist_ok=True)

        Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(
            root / "training" / "htrain-1-2-[1 0 0 1].png"
        )

        val_name = "hval-1-2.png"
        test_name = "htest-1-2.png"
        Image.fromarray(np.full((2, 2, 3), 64, dtype=np.uint8)).save(root / "val" / "img" / val_name)
        Image.fromarray(np.full((2, 2, 3), 64, dtype=np.uint8)).save(root / "test" / "img" / test_name)

        hist_rgb_mask = np.array(
            [
                [[205, 51, 51], [255, 165, 0]],
                [[0, 255, 0], [1, 2, 3]],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(hist_rgb_mask, mode="RGB").save(root / "val" / "mask" / val_name)
        Image.fromarray(hist_rgb_mask, mode="RGB").save(root / "test" / "mask" / test_name)

    def _build_minimal_bcss_raw(self, root: Path) -> None:
        (root / "training").mkdir(parents=True, exist_ok=True)
        (root / "val" / "img").mkdir(parents=True, exist_ok=True)
        (root / "val" / "mask").mkdir(parents=True, exist_ok=True)
        (root / "test" / "img").mkdir(parents=True, exist_ok=True)
        (root / "test" / "mask").mkdir(parents=True, exist_ok=True)

        Image.fromarray(np.full((2, 2, 3), 192, dtype=np.uint8)).save(
            root / "training" / "btrain+0[1100].png"
        )

        val_name = "bval+0.png"
        test_name = "btest+0.png"
        Image.fromarray(np.full((2, 2, 3), 200, dtype=np.uint8)).save(root / "val" / "img" / val_name)
        Image.fromarray(np.full((2, 2, 3), 200, dtype=np.uint8)).save(root / "test" / "img" / test_name)

        indexed_mask = np.array([[0, 1], [2, 4]], dtype=np.uint8)
        Image.fromarray(indexed_mask, mode="L").save(root / "val" / "mask" / val_name)
        Image.fromarray(indexed_mask, mode="L").save(root / "test" / "mask" / test_name)


if __name__ == "__main__":
    unittest.main()
