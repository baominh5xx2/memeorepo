from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stage1_damp.data import SourceWeakLabelDataset, flags_to_label, parse_flags_from_sample_id


class TestStage1Data(unittest.TestCase):
    def test_parse_flags_fallback(self) -> None:
        self.assertEqual(parse_flags_from_sample_id("foo+0[1100]"), (1, 1))
        self.assertEqual(parse_flags_from_sample_id("bar-[1 0 0 1]"), (1, 1))
        self.assertIsNone(parse_flags_from_sample_id("sample_without_flags"))

        self.assertEqual(flags_to_label(0, 0), 0)
        self.assertEqual(flags_to_label(1, 0), 1)
        self.assertEqual(flags_to_label(0, 1), 2)
        self.assertEqual(flags_to_label(1, 1), 3)

    def test_source_weak_label_dataset_prefers_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Hist"
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "splits").mkdir(parents=True, exist_ok=True)
            (root / "metadata").mkdir(parents=True, exist_ok=True)

            sample_ids = [
                "train_h1-[1 0 0 1]",
                "train_h2-[1 1 0 0]",
            ]
            (root / "splits" / "train.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

            for sid in sample_ids:
                img = np.full((2, 2, 3), 120, dtype=np.uint8)
                Image.fromarray(img).save(root / "images" / f"{sid}.png")

            with (root / "metadata" / "train_weak_labels.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["sample_id", "tumor", "stroma", "lymphocyte", "necrosis"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": sample_ids[0],
                        "tumor": 1,
                        "stroma": 1,
                        "lymphocyte": 0,
                        "necrosis": 0,
                    }
                )
                writer.writerow(
                    {
                        "sample_id": sample_ids[1],
                        "tumor": 1,
                        "stroma": 0,
                        "lymphocyte": 0,
                        "necrosis": 1,
                    }
                )

            ds = SourceWeakLabelDataset(domain_root=root, split="train", image_size=224)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]["label"], 3)
            self.assertEqual(ds[1]["label"], 1)


if __name__ == "__main__":
    unittest.main()
