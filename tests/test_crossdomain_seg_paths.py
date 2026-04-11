from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(TORCH_AVAILABLE and TORCHVISION_AVAILABLE, "torch/torchvision is required")
class TestCrossDomainSegPaths(unittest.TestCase):
    def test_resolve_sample_id_with_brackets(self) -> None:
        from datasets.crossdomain_seg import CrossDomainSegDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir(parents=True, exist_ok=True)
            (root / "masks").mkdir(parents=True, exist_ok=True)
            (root / "splits").mkdir(parents=True, exist_ok=True)

            sample_id = "train_TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101]"
            (root / "splits" / "train.txt").write_text(sample_id + "\n", encoding="utf-8")
            (root / "images" / f"{sample_id}.png").write_bytes(b"dummy")

            dataset = CrossDomainSegDataset(domain_root=root, split="train", image_size=16)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.samples[0].sample_id, sample_id)
            self.assertEqual(dataset.samples[0].image_path.name, f"{sample_id}.png")


if __name__ == "__main__":
    unittest.main()
