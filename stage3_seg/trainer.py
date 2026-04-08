from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from damp_es.common.io import ensure_dir
from damp_es.datasets.constants import STROMA_LABEL, TUMOR_LABEL


@dataclass
class TrainerState:
    best_miou: float = 0.0
    best_epoch: int = -1


class SegmentationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str | Path,
        eval_every: int = 1,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = ensure_dir(save_dir)
        self.eval_every = eval_every
        self.state = TrainerState()

    def train(self, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            print(f"[Stage3] epoch={epoch} train_loss={train_loss:.6f}")

            if epoch % self.eval_every == 0:
                metrics = self.evaluate(self.val_loader)
                miou = metrics["mIoU"]
                print(
                    f"[Stage3] val epoch={epoch} mIoU={miou:.4f} "
                    f"FwIoU={metrics['FwIoU']:.4f} ACC={metrics['ACC']:.4f} "
                    f"Tumor={metrics['Tumor']:.4f} Stroma={metrics['Stroma']:.4f}"
                )
                self._save_checkpoint(epoch=epoch, metrics=metrics)

                if miou > self.state.best_miou:
                    self.state.best_miou = miou
                    self.state.best_epoch = epoch
                    self._save_checkpoint(epoch=epoch, metrics=metrics, is_best=True)

        print(
            f"[Stage3] best_mIoU={self.state.best_miou:.4f} "
            f"at epoch={self.state.best_epoch}"
        )

    def evaluate(self, loader: DataLoader, prediction_dir: str | Path | None = None) -> Dict[str, float]:
        self.model.eval()
        num_classes = 3
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        out_dir = ensure_dir(prediction_dir) if prediction_dir is not None else None

        with torch.no_grad():
            for batch in tqdm(loader, desc="Stage3 eval", leave=False):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                sample_ids = batch["sample_id"]

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)
                preds_np = preds.cpu().numpy().astype(np.uint8)

                confusion += self._fast_hist(
                    y_true=masks.cpu().numpy().reshape(-1),
                    y_pred=preds_np.reshape(-1),
                    n_class=num_classes,
                )

                if out_dir is not None:
                    for i, sample_id in enumerate(sample_ids):
                        Image.fromarray(preds_np[i], mode="L").save(out_dir / f"{sample_id}.png")

        return self._compute_metrics(confusion)

    def _train_one_epoch(self, epoch: int) -> float:
        del epoch
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Stage3 train", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            confidence = batch.get("confidence")
            if confidence is not None:
                confidence = confidence.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits=logits, target=masks, confidence=confidence)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())

        return running_loss / max(len(self.train_loader), 1)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        suffix = "best" if is_best else f"epoch_{epoch:03d}"
        ckpt_path = self.save_dir / f"{suffix}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            ckpt_path,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)

    @staticmethod
    def _fast_hist(y_true: np.ndarray, y_pred: np.ndarray, n_class: int) -> np.ndarray:
        mask = (y_true >= 0) & (y_true < n_class)
        hist = np.bincount(
            n_class * y_true[mask].astype(int) + y_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    @staticmethod
    def _compute_metrics(hist: np.ndarray) -> Dict[str, float]:
        acc = np.diag(hist).sum() / (hist.sum() + 1e-8)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-8)
        valid = hist.sum(axis=1) > 0
        miou = np.mean(iu[valid]) if valid.any() else 0.0
        freq = hist.sum(axis=1) / (hist.sum() + 1e-8)
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return {
            "Tumor": float(iu[TUMOR_LABEL]) if hist.shape[0] > TUMOR_LABEL else 0.0,
            "Stroma": float(iu[STROMA_LABEL]) if hist.shape[0] > STROMA_LABEL else 0.0,
            "ACC": float(acc),
            "mIoU": float(miou),
            "FwIoU": float(fwiou),
        }
