from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.eval_crossdomain import CrossDomainEvaluator, EvalConfig


class TestCrossDomainEvaluator(unittest.TestCase):
    def test_evaluate_metrics_with_ignore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            domain_root = tmp_path / "domain"
            pred_root = tmp_path / "pred"
            (domain_root / "splits").mkdir(parents=True, exist_ok=True)
            (domain_root / "masks").mkdir(parents=True, exist_ok=True)
            pred_root.mkdir(parents=True, exist_ok=True)

            (domain_root / "splits" / "test.txt").write_text("s1\ns2\n", encoding="utf-8")

            gt1 = np.array([[0, 1], [2, 255]], dtype=np.uint8)
            gt2 = np.array([[0, 1], [2, 1]], dtype=np.uint8)
            pr1 = np.array([[0, 1], [2, 0]], dtype=np.uint8)
            pr2 = np.array([[0, 2], [2, 1]], dtype=np.uint8)

            Image.fromarray(gt1, mode="L").save(domain_root / "masks" / "s1.png")
            Image.fromarray(gt2, mode="L").save(domain_root / "masks" / "s2.png")
            Image.fromarray(pr1, mode="L").save(pred_root / "s1.png")
            Image.fromarray(pr2, mode="L").save(pred_root / "s2.png")

            evaluator = CrossDomainEvaluator(
                EvalConfig(
                    pred_dir=pred_root,
                    domain_root=domain_root,
                    split="test",
                    num_classes=3,
                    ignore_index=255,
                )
            )
            metrics = evaluator.evaluate()

            self.assertAlmostEqual(metrics["Tumor"], 2.0 / 3.0, places=6)
            self.assertAlmostEqual(metrics["Stroma"], 2.0 / 3.0, places=6)
            self.assertAlmostEqual(metrics["ACC"], 6.0 / 7.0, places=6)
            self.assertAlmostEqual(metrics["mIoU"], 7.0 / 9.0, places=6)
            self.assertAlmostEqual(metrics["FwIoU"], 16.0 / 21.0, places=6)

    def test_missing_prediction_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            domain_root = tmp_path / "domain"
            pred_root = tmp_path / "pred"
            (domain_root / "splits").mkdir(parents=True, exist_ok=True)
            (domain_root / "masks").mkdir(parents=True, exist_ok=True)
            pred_root.mkdir(parents=True, exist_ok=True)

            (domain_root / "splits" / "test.txt").write_text("s1\n", encoding="utf-8")
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(
                domain_root / "masks" / "s1.png"
            )

            evaluator = CrossDomainEvaluator(
                EvalConfig(
                    pred_dir=pred_root,
                    domain_root=domain_root,
                    split="test",
                    num_classes=3,
                    ignore_index=255,
                )
            )

            with self.assertRaises(FileNotFoundError):
                _ = evaluator.evaluate()


if __name__ == "__main__":
    unittest.main()
