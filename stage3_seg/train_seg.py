from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from damp_es.common.config import apply_overrides, load_yaml_config, parse_overrides
from damp_es.datasets.crossdomain_seg import CrossDomainSegDataset
from damp_es.stage3_seg.cgl import ConfidenceGuidedLoss
from damp_es.stage3_seg.deeplab import DeepLabSegModel
from damp_es.stage3_seg.trainer import SegmentationTrainer


class PlainSegmentationLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor | None = None) -> torch.Tensor:
        del confidence
        return torch.nn.functional.cross_entropy(logits, target, ignore_index=self.ignore_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 segmentation training")
    parser.add_argument("--config", type=str, default="configs/stage3_seg.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()


def build_dataloaders(cfg):
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

    domain_root = Path(dataset_cfg["root"]) / dataset_cfg["domain"]
    pseudo_mask_dir = dataset_cfg.get("pseudo_mask_dir")

    train_dataset = CrossDomainSegDataset(
        domain_root=domain_root,
        split=dataset_cfg["train_split"],
        image_size=int(train_cfg["image_size"]),
        pseudo_mask_dir=pseudo_mask_dir,
    )
    val_dataset = CrossDomainSegDataset(
        domain_root=domain_root,
        split=dataset_cfg["val_split"],
        image_size=int(train_cfg["image_size"]),
    )
    test_dataset = CrossDomainSegDataset(
        domain_root=domain_root,
        split=dataset_cfg["test_split"],
        image_size=int(train_cfg["image_size"]),
    )

    num_workers = int(train_cfg["num_workers"])
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", True))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 4))

    train_loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    eval_loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        eval_loader_kwargs["persistent_workers"] = persistent_workers
        eval_loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        **eval_loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_overrides(cfg, parse_overrides(args.override))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_cfg = cfg.get("runtime", {})

    if device.type == "cuda":
        allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        try:
            torch.set_float32_matmul_precision(str(runtime_cfg.get("matmul_precision", "high")))
        except Exception:
            pass

    print(f"[Stage3] device={device}")
    print(
        f"[Stage3] amp={bool(runtime_cfg.get('amp', True) and device.type == 'cuda')} "
        f"amp_dtype={str(runtime_cfg.get('amp_dtype', 'bf16'))} "
        f"tf32={bool(runtime_cfg.get('allow_tf32', True))} "
        f"channels_last={bool(runtime_cfg.get('channels_last', True))}"
    )
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    cgl_cfg = cfg["cgl"]

    if int(train_cfg["batch_size"]) < 2:
        raise ValueError(
            "training.batch_size must be >= 2 for deeplabv3_resnet101 training "
            "because BatchNorm is unstable with singleton batches."
        )

    architecture = str(model_cfg.get("architecture", "deeplabv3_resnet101")).lower()
    if architecture != "deeplabv3_resnet101":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = DeepLabSegModel(
        num_classes=int(model_cfg["num_classes"]),
        pretrained_backbone=bool(model_cfg.get("pretrained_backbone", True)),
    )

    optimizer_name = str(opt_cfg.get("name", "adamw")).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(opt_cfg["lr"]),
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    cgl_enabled = bool(cgl_cfg.get("enabled", True))
    if cgl_enabled:
        criterion = ConfidenceGuidedLoss(
            confidence_threshold=float(cgl_cfg.get("confidence_threshold", 0.25)),
            ignore_index=int(cgl_cfg.get("ignore_index", 255)),
        )
    else:
        criterion = PlainSegmentationLoss(ignore_index=int(cgl_cfg.get("ignore_index", 255)))

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=train_cfg["save_dir"],
        eval_every=int(train_cfg.get("eval_every", 1)),
        use_amp=bool(runtime_cfg.get("amp", True)),
        amp_dtype=str(runtime_cfg.get("amp_dtype", "bf16")),
        channels_last=bool(runtime_cfg.get("channels_last", True)),
    )

    trainer.train(epochs=int(train_cfg["epochs"]))

    best_ckpt = Path(train_cfg["save_dir"]) / "best.pth"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)

    pred_dir = Path(train_cfg["save_dir"]) / "test_predictions"
    test_metrics = trainer.evaluate(test_loader, prediction_dir=pred_dir)
    print(
        "[Stage3] test metrics: "
        f"Tumor={test_metrics['Tumor']:.4f} "
        f"Stroma={test_metrics['Stroma']:.4f} "
        f"mIoU={test_metrics['mIoU']:.4f} "
        f"FwIoU={test_metrics['FwIoU']:.4f} "
        f"ACC={test_metrics['ACC']:.4f}"
    )
    print(f"[Stage3] saved test predictions to: {pred_dir}")


if __name__ == "__main__":
    main()
