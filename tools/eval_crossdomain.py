from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from damp_es.datasets.constants import STROMA_LABEL, TUMOR_LABEL


@dataclass
class EvalConfig:
    pred_dir: Path
    domain_root: Path
    split: str
    num_classes: int = 3
    ignore_index: int = 255


class CrossDomainEvaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def evaluate(self) -> Dict[str, float]:
        sample_ids = self._load_split_ids()
        hist = np.zeros((self.cfg.num_classes, self.cfg.num_classes), dtype=np.int64)

        for sample_id in sample_ids:
            pred_path = self.cfg.pred_dir / f"{sample_id}.png"
            gt_path = self.cfg.domain_root / "masks" / f"{sample_id}.png"
            if not pred_path.exists():
                raise FileNotFoundError(f"Prediction not found: {pred_path}")
            if not gt_path.exists():
                raise FileNotFoundError(f"Ground-truth mask not found: {gt_path}")

            pred_img = Image.open(pred_path)
            gt_img = Image.open(gt_path)
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.NEAREST)

            pred = np.array(pred_img, dtype=np.int64)
            gt = np.array(gt_img, dtype=np.int64)

            hist += self._fast_hist(gt.reshape(-1), pred.reshape(-1), self.cfg.num_classes)

        metrics = self._compute_metrics(hist)
        return metrics

    def _load_split_ids(self) -> List[str]:
        split_file = self.cfg.domain_root / "splits" / f"{self.cfg.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {split_file}")
        with split_file.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray, n_class: int) -> np.ndarray:
        mask = (
            (label_true >= 0)
            & (label_true < n_class)
            & (label_true != self.cfg.ignore_index)
            & (label_pred >= 0)
            & (label_pred < n_class)
        )
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def _compute_metrics(self, hist: np.ndarray) -> Dict[str, float]:
        acc = np.diag(hist).sum() / (hist.sum() + 1e-8)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-8)
        valid = hist.sum(axis=1) > 0
        miou = np.mean(iu[valid]) if valid.any() else 0.0
        freq = hist.sum(axis=1) / (hist.sum() + 1e-8)
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()

        tumor_iou = float(iu[TUMOR_LABEL]) if self.cfg.num_classes > TUMOR_LABEL else 0.0
        stroma_iou = float(iu[STROMA_LABEL]) if self.cfg.num_classes > STROMA_LABEL else 0.0

        return {
            "Tumor": tumor_iou,
            "Stroma": stroma_iou,
            "ACC": float(acc),
            "mIoU": float(miou),
            "FwIoU": float(fwiou),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-domain segmentation masks")
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--domain-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--ignore-index", type=int, default=255)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        pred_dir=Path(args.pred_dir),
        domain_root=Path(args.domain_root),
        split=args.split,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )

    evaluator = CrossDomainEvaluator(cfg)
    metrics = evaluator.evaluate()
    print(
        "[Eval] "
        f"split={args.split} "
        f"Tumor={metrics['Tumor']:.4f} "
        f"Stroma={metrics['Stroma']:.4f} "
        f"mIoU={metrics['mIoU']:.4f} "
        f"FwIoU={metrics['FwIoU']:.4f} "
        f"ACC={metrics['ACC']:.4f}"
    )


if __name__ == "__main__":
    main()
