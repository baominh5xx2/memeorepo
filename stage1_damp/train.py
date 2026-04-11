from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import apply_overrides, load_yaml_config, parse_overrides
from common.io import ensure_dir
from stage1_damp.data import SourceWeakLabelDataset, SplitImageDataset
from stage1_damp.model import DAMPWrapper


@dataclass
class Stage1TrainConfig:
    backbone_name: str
    clip_weights: str | None
    source_domain: str
    target_domain: str
    dataset_root: Path
    output_dir: Path
    seed: int
    image_size: int
    batch_size: int
    num_workers: int
    n_ctx: int
    epochs: int
    warmup_epochs: int
    optimizer_name: str
    lr: float
    weight_decay: float
    save_every: int
    tau: float
    lambda_cls: float
    lambda_u: float
    lambda_ind: float
    lambda_im: float
    pin_memory: bool
    prefetch_factor: int
    persistent_workers: bool
    allow_tf32: bool
    cudnn_benchmark: bool
    matmul_precision: str
    amp_enabled: bool
    amp_dtype: str


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.bfloat16


def _build_grad_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def im_loss(outputs_target: torch.Tensor, mask_lt: torch.Tensor) -> torch.Tensor:
    if int(mask_lt.sum().item()) == 0:
        return outputs_target.sum() * 0.0

    outputs_target = outputs_target[mask_lt]
    batch_size = mask_lt.sum().float()
    softmax_outs_t = torch.softmax(outputs_target, dim=1)
    avg_softmax_outs_t = softmax_outs_t.sum(dim=0) / batch_size + 1e-5
    log_avg = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t + 1e-5)) / batch_size
    return item2 - item1


class InternalStage1Trainer:
    """Internal stage1 adaptation trainer (self-contained inside damp_es)."""

    def __init__(self, cfg: Stage1TrainConfig):
        self.cfg = cfg
        self.class_names = ["background", "tumor", "stroma", "tumor stroma"]
        self.wrapper = DAMPWrapper(
            backbone=cfg.backbone_name,
            clip_weights=cfg.clip_weights,
            device="cuda",
            feature_layer=-1,
            n_ctx=cfg.n_ctx,
            class_names=self.class_names,
            enable_mutual_prompting=True,
        )
        self.device = self.wrapper.device
        print(f"[Stage1] device={self.device}")

        self.use_amp = bool(cfg.amp_enabled and self.device.type == "cuda")
        self.amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16)
        self.grad_scaler = _build_grad_scaler(self.use_grad_scaler)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
            torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
            try:
                torch.set_float32_matmul_precision(str(cfg.matmul_precision))
            except Exception:
                pass

        print(
            f"[Stage1] amp={self.use_amp} amp_dtype={self.amp_dtype} "
            f"tf32={bool(cfg.allow_tf32)} cudnn_benchmark={bool(cfg.cudnn_benchmark)}"
        )

        self.source_root = cfg.dataset_root / cfg.source_domain
        self.target_root = cfg.dataset_root / cfg.target_domain

        optim_name = cfg.optimizer_name.lower()
        if optim_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.wrapper.stage1_parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif optim_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.wrapper.stage1_parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported Stage1 optimizer: {cfg.optimizer_name}")

        self.base_lr = float(cfg.lr)

    def run(self) -> None:
        source_dataset = SourceWeakLabelDataset(
            domain_root=self.source_root,
            split="train",
            image_size=self.cfg.image_size,
        )
        target_dataset = SplitImageDataset(
            domain_root=self.target_root,
            split="train",
            image_size=self.cfg.image_size,
        )

        if len(source_dataset) == 0:
            raise RuntimeError(f"Empty source dataset: {self.source_root}")
        if len(target_dataset) == 0:
            raise RuntimeError(f"Empty target dataset: {self.target_root}")

        source_loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "shuffle": True,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.cfg.pin_memory,
        }
        target_loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "shuffle": True,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.cfg.pin_memory,
        }

        if self.cfg.num_workers > 0:
            source_loader_kwargs["persistent_workers"] = bool(self.cfg.persistent_workers)
            source_loader_kwargs["prefetch_factor"] = max(1, int(self.cfg.prefetch_factor))
            target_loader_kwargs["persistent_workers"] = bool(self.cfg.persistent_workers)
            target_loader_kwargs["prefetch_factor"] = max(1, int(self.cfg.prefetch_factor))

        source_loader = DataLoader(source_dataset, **source_loader_kwargs)
        target_loader = DataLoader(target_dataset, **target_loader_kwargs)

        history: List[Dict[str, float | int]] = []
        best_loss = float("inf")

        for epoch in range(1, self.cfg.epochs + 1):
            self._current_epoch = epoch
            current_lr = self._set_epoch_lr(epoch)
            metrics = self._train_one_epoch(source_loader, target_loader)
            is_best = float(metrics["loss_total"]) <= best_loss
            if is_best:
                best_loss = float(metrics["loss_total"])

            should_save = (
                (epoch % max(int(self.cfg.save_every), 1) == 0)
                or is_best
                or (epoch == self.cfg.epochs)
            )
            if should_save:
                self.wrapper.save_stage1_checkpoint(
                    output_root=self.cfg.output_dir,
                    epoch=epoch,
                    is_best=is_best,
                )

            row = {
                "epoch": epoch,
                "loss_total": float(metrics["loss_total"]),
                "loss_cls": float(metrics["loss_cls"]),
                "loss_im": float(metrics["loss_im"]),
                "source_acc": float(metrics["source_acc"]),
            }
            history.append(row)

            print(
                f"[Stage1] epoch={epoch}/{self.cfg.epochs} "
                f"loss={row['loss_total']:.4f} "
                f"cls={row['loss_cls']:.4f} "
                f"im={row['loss_im']:.4f} "
                f"src_acc={row['source_acc']:.4f} "
                f"lr={current_lr:.6f}"
            )

        out_metrics = self.cfg.output_dir / "metrics.json"
        with out_metrics.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_domain": self.cfg.source_domain,
                    "target_domain": self.cfg.target_domain,
                    "best_loss": best_loss,
                    "history": history,
                },
                f,
                indent=2,
            )

        print(f"[Stage1] saved mutual-prompt checkpoints to: {self.cfg.output_dir / 'mutual_prompt'}")

    def _set_epoch_lr(self, epoch: int) -> float:
        warmup = max(int(self.cfg.warmup_epochs), 0)
        total = max(int(self.cfg.epochs), 1)

        if warmup > 0 and epoch <= warmup:
            lr = self.base_lr * (float(epoch) / float(warmup))
        elif total > warmup:
            progress = float(epoch - warmup) / float(total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        else:
            lr = self.base_lr

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _train_one_epoch(self, source_loader: DataLoader, target_loader: DataLoader) -> Dict[str, float]:
        self.wrapper.set_stage1_train(True)

        total_loss = 0.0
        total_cls = 0.0
        total_im = 0.0
        total_correct = 0
        total_seen = 0

        target_iter = iter(target_loader)

        n_steps = max(len(source_loader), 1)
        progress = tqdm(
            source_loader,
            total=len(source_loader),
            desc=f"[Stage1] epoch {self._current_epoch}/{self.cfg.epochs}",
            dynamic_ncols=True,
            leave=False,
        )

        for step_idx, source_batch in enumerate(progress, start=1):
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            src_images = source_batch["image"].to(self.device)
            src_labels = source_batch["label"].to(self.device).long()
            tgt_images = target_batch["image"].to(self.device)

            # Lightweight dual-view augmentation to mimic DAMP's weak/strong views.
            src_images2 = torch.flip(src_images, dims=[-1])
            tgt_images2 = torch.flip(tgt_images, dims=[-1])

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                if self.use_amp
                else nullcontext()
            )

            with autocast_ctx:
                output_x, output_x_ind = self.wrapper.forward_stage1(src_images, ind=True, pse=False)
                output_u, output_u_ind, pseudo_label_logits = self.wrapper.forward_stage1(
                    tgt_images,
                    ind=True,
                    pse=True,
                )

                output_x2 = self.wrapper.forward_stage1(src_images2)[0]
                output_u2 = self.wrapper.forward_stage1(tgt_images2)[0]

                mix_lambda = float(max(1, self._current_epoch)) / float(max(self.cfg.epochs, 1))
                pseudo_label = (
                    torch.softmax(output_u.reshape(-1, len(self.class_names)), dim=-1) * mix_lambda
                    + torch.softmax(pseudo_label_logits.reshape(-1, len(self.class_names)), dim=-1) * (1.0 - mix_lambda)
                ).detach()

                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask_ge = max_probs.ge(self.cfg.tau).float()
                mask_lt = max_probs.lt(1.0)

                loss_x = F.cross_entropy(output_x, src_labels)
                loss_x2 = F.cross_entropy(output_x2, src_labels)
                loss_cls = loss_x + loss_x2
                if int(mask_ge.sum().item()) > 0:
                    loss_u = (F.cross_entropy(output_u, label_p, reduction="none") * mask_ge).sum() / mask_ge.sum()
                    loss_u2 = (F.cross_entropy(output_u2, label_p, reduction="none") * mask_ge).sum() / mask_ge.sum()
                else:
                    loss_u = output_u.sum() * 0.0
                    loss_u2 = output_u2.sum() * 0.0

                x_ind_label = torch.arange(output_x_ind.shape[0], dtype=torch.long, device=self.device)
                loss_x_ind = (
                    F.cross_entropy(output_x_ind, x_ind_label)
                    + F.cross_entropy(output_x_ind.permute(1, 0), x_ind_label)
                ) / 2.0

                u_ind_label = torch.arange(output_u_ind.shape[0], dtype=torch.long, device=self.device)
                loss_u_ind = (
                    F.cross_entropy(output_u_ind, u_ind_label)
                    + F.cross_entropy(output_u_ind.permute(1, 0), u_ind_label)
                ) / 2.0

                loss_ind = loss_x_ind + loss_u_ind
                loss_im = im_loss(output_u, mask_lt)

                loss = (
                    self.cfg.lambda_cls * loss_cls
                    + self.cfg.lambda_u * (loss_u + loss_u2)
                    + self.cfg.lambda_ind * loss_ind
                    + self.cfg.lambda_im * loss_im
                )

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_grad_scaler and self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                pred_src = output_x.argmax(dim=-1)
                total_correct += int((pred_src == src_labels).sum().item())
                total_seen += int(src_labels.numel())

            total_loss += float(loss.item())
            total_cls += float(loss_cls.item())
            total_im += float(loss_im.item())

            if step_idx % 10 == 0 or step_idx == n_steps:
                avg_loss = total_loss / float(step_idx)
                avg_cls = total_cls / float(step_idx)
                avg_im = total_im / float(step_idx)
                src_acc = float(total_correct) / float(max(total_seen, 1))
                progress.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "cls": f"{avg_cls:.4f}",
                        "im": f"{avg_im:.4f}",
                        "src_acc": f"{src_acc:.3f}",
                    }
                )

        progress.close()
        return {
            "loss_total": total_loss / n_steps,
            "loss_cls": total_cls / n_steps,
            "loss_im": total_im / n_steps,
            "source_acc": float(total_correct) / float(max(total_seen, 1)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internal Stage 1 adaptation training")
    parser.add_argument("--config", type=str, default="configs/stage1_damp.yaml")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values using key=value format",
    )
    parser.add_argument(
        "--damp-opt",
        type=str,
        nargs="*",
        default=[],
        help="Reserved for compatibility. Ignored by internal stage1 trainer.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_config(config_dict: Dict) -> Stage1TrainConfig:
    backbone_cfg = config_dict.get("backbone", {})
    dataset_cfg = config_dict["dataset"]
    training_cfg = config_dict["training"]
    loss_cfg = config_dict.get("losses", {})
    mutual_cfg = config_dict.get("mutual_prompting", {})
    runtime_cfg = config_dict.get("runtime", {})

    return Stage1TrainConfig(
        backbone_name=str(backbone_cfg.get("name", "ViT-B/16")),
        clip_weights=backbone_cfg.get("pretrained_model"),
        source_domain=str(dataset_cfg["source_domain"]),
        target_domain=str(dataset_cfg["target_domain"]),
        dataset_root=Path(dataset_cfg["root"]),
        output_dir=ensure_dir(training_cfg["output_dir"]),
        seed=int(config_dict.get("seed", 42)),
        image_size=int(training_cfg.get("image_size", 224)),
        batch_size=int(training_cfg.get("batch_size", 16)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        n_ctx=int(mutual_cfg.get("n_ctx_text", mutual_cfg.get("n_ctx_visual", 4))),
        epochs=int(training_cfg.get("epochs", 10)),
        warmup_epochs=int(training_cfg.get("warmup_epochs", 1)),
        optimizer_name=str(training_cfg.get("optimizer", "adam")),
        lr=float(training_cfg.get("lr", 2e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        save_every=int(training_cfg.get("save_every", 1)),
        lambda_cls=float(loss_cfg.get("lambda_cls", 1.0)),
        lambda_u=float(loss_cfg.get("lambda_u", 1.0)),
        lambda_ind=float(loss_cfg.get("lambda_ind", 1.0)),
        lambda_im=float(loss_cfg.get("lambda_im", 1.0)),
        tau=float(loss_cfg.get("tau", 0.5)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        prefetch_factor=int(training_cfg.get("prefetch_factor", 4)),
        persistent_workers=bool(training_cfg.get("persistent_workers", True)),
        allow_tf32=bool(runtime_cfg.get("allow_tf32", True)),
        cudnn_benchmark=bool(runtime_cfg.get("cudnn_benchmark", True)),
        matmul_precision=str(runtime_cfg.get("matmul_precision", "high")),
        amp_enabled=bool(runtime_cfg.get("amp", True)),
        amp_dtype=str(runtime_cfg.get("amp_dtype", "bf16")),
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_overrides(cfg, parse_overrides(args.override))

    if args.damp_opt:
        print("[Stage1] note: --damp-opt is ignored by internal damp_es trainer")

    train_cfg = build_train_config(cfg)
    set_global_seed(train_cfg.seed)

    trainer = InternalStage1Trainer(train_cfg)
    trainer.run()


if __name__ == "__main__":
    main()
