from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
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
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        channels_last: bool = True,
        consistency_enabled: bool = False,
        consistency_weight: float = 0.0,
        consistency_confidence_threshold: float = 0.6,
        consistency_noise_std: float = 0.05,
        consistency_drop_prob: float = 0.5,
        consistency_cutout_ratio: float = 0.2,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = ensure_dir(save_dir)
        self.eval_every = eval_every
        self.early_stop_patience = max(0, int(early_stop_patience))
        self.early_stop_min_delta = float(max(0.0, early_stop_min_delta))
        self.ignore_index = int(getattr(self.criterion, "ignore_index", 255))

        self.consistency_enabled = bool(consistency_enabled)
        self.consistency_weight = float(max(0.0, consistency_weight))
        self.consistency_confidence_threshold = float(max(0.0, consistency_confidence_threshold))
        self.consistency_noise_std = float(max(0.0, consistency_noise_std))
        self.consistency_drop_prob = float(min(1.0, max(0.0, consistency_drop_prob)))
        self.consistency_cutout_ratio = float(min(1.0, max(0.0, consistency_cutout_ratio)))
        self.state = TrainerState()

        self.channels_last = bool(channels_last and self.device.type == "cuda")
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        amp_name = str(amp_dtype).strip().lower()
        if amp_name in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16

        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        if self.use_grad_scaler:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.grad_scaler = None

    def train(self, epochs: int) -> None:
        no_improve_evals = 0
        for epoch in range(1, epochs + 1):
            train_stats = self._train_one_epoch(epoch)
            print(
                f"[Stage3] epoch={epoch} train_loss={train_stats['total_loss']:.6f} "
                f"sup_loss={train_stats['supervised_loss']:.6f} "
                f"cons_loss={train_stats['consistency_loss']:.6f}"
            )

            if epoch % self.eval_every == 0:
                metrics = self.evaluate(self.val_loader)
                miou = metrics["mIoU"]
                print(
                    f"[Stage3] val epoch={epoch} mIoU={miou:.4f} "
                    f"FwIoU={metrics['FwIoU']:.4f} ACC={metrics['ACC']:.4f} "
                    f"Tumor={metrics['Tumor']:.4f} Stroma={metrics['Stroma']:.4f}"
                )
                self._save_checkpoint(epoch=epoch, metrics=metrics)

                if miou > (self.state.best_miou + self.early_stop_min_delta):
                    self.state.best_miou = miou
                    self.state.best_epoch = epoch
                    self._save_checkpoint(epoch=epoch, metrics=metrics, is_best=True)
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1

                if self.early_stop_patience > 0 and no_improve_evals >= self.early_stop_patience:
                    print(
                        "[Stage3] early stopping triggered: "
                        f"no mIoU improvement > {self.early_stop_min_delta:.4f} "
                        f"for {self.early_stop_patience} evals"
                    )
                    break

        print(
            f"[Stage3] best_mIoU={self.state.best_miou:.4f} "
            f"at epoch={self.state.best_epoch}"
        )

    def evaluate(self, loader: DataLoader, prediction_dir: str | Path | None = None) -> Dict[str, float]:
        self.model.eval()
        num_classes = getattr(self.model, "num_classes", 3)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        out_dir = ensure_dir(prediction_dir) if prediction_dir is not None else None

        with torch.no_grad():
            for batch in tqdm(loader, desc="Stage3 eval", leave=False):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                sample_ids = batch["sample_id"]

                if self.channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                    if self.use_amp
                    else nullcontext()
                )
                with autocast_ctx:
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

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        del epoch
        self.model.train()
        running_total_loss = 0.0
        running_supervised_loss = 0.0
        running_consistency_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Stage3 train", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            confidence = batch.get("confidence")
            if confidence is not None:
                confidence = confidence.to(self.device)

            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                if self.use_amp
                else nullcontext()
            )
            with autocast_ctx:
                logits = self.model(images)
                supervised_loss = self.criterion(logits=logits, target=masks, confidence=confidence)
                consistency_loss = logits.sum() * 0.0
                if self.consistency_enabled and self.consistency_weight > 0.0:
                    consistency_loss = self._compute_consistency_loss(
                        weak_logits=logits,
                        weak_images=images,
                        masks=masks,
                    )
                loss = supervised_loss + self.consistency_weight * consistency_loss

            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_total_loss += float(loss.item())
            running_supervised_loss += float(supervised_loss.item())
            running_consistency_loss += float(consistency_loss.item())

        train_len = len(self.train_loader)
        if train_len == 0:
            print("[Warning] Train loader has 0 batches! Are you using drop_last=True with dataset size < batch size?")
            return {
                "total_loss": 0.0,
                "supervised_loss": 0.0,
                "consistency_loss": 0.0,
            }

        return {
            "total_loss": running_total_loss / train_len,
            "supervised_loss": running_supervised_loss / train_len,
            "consistency_loss": running_consistency_loss / train_len,
        }

    def _compute_consistency_loss(
        self,
        weak_logits: torch.Tensor,
        weak_images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        weak_probs = torch.softmax(weak_logits.detach(), dim=1)
        weak_confidence, weak_labels = torch.max(weak_probs, dim=1)

        valid = weak_confidence >= self.consistency_confidence_threshold
        valid = valid & (masks != self.ignore_index)

        strong_images = self._build_strong_view(weak_images)
        strong_logits = self.model(strong_images)
        per_pixel = F.cross_entropy(strong_logits, weak_labels, reduction="none")

        valid_float = valid.float()
        valid_count = valid_float.sum()
        if valid_count.item() == 0:
            return strong_logits.sum() * 0.0

        return (per_pixel * valid_float).sum() / valid_count.clamp_min(1.0)

    def _build_strong_view(self, weak_images: torch.Tensor) -> torch.Tensor:
        strong = weak_images.detach().clone()

        if self.consistency_noise_std > 0.0:
            strong = strong + torch.randn_like(strong) * self.consistency_noise_std

        batch_size, _, height, width = strong.shape
        if batch_size == 0:
            return strong

        flip_mask = torch.rand(batch_size, device=strong.device) < 0.5
        if bool(flip_mask.any().item()):
            strong[flip_mask] = torch.flip(strong[flip_mask], dims=[3])

        if self.consistency_drop_prob > 0.0 and self.consistency_cutout_ratio > 0.0 and height > 1 and width > 1:
            cut_h = max(1, int(height * self.consistency_cutout_ratio))
            cut_w = max(1, int(width * self.consistency_cutout_ratio))
            max_top = max(1, height - cut_h + 1)
            max_left = max(1, width - cut_w + 1)

            for batch_idx in range(batch_size):
                if torch.rand(1, device=strong.device).item() >= self.consistency_drop_prob:
                    continue
                top = int(torch.randint(0, max_top, (1,), device=strong.device).item())
                left = int(torch.randint(0, max_left, (1,), device=strong.device).item())
                strong[batch_idx, :, top : top + cut_h, left : left + cut_w] = 0.0

        return strong

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
