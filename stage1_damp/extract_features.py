from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from damp_es.common.config import apply_overrides, load_yaml_config, parse_overrides
from damp_es.common.io import ensure_dir
from damp_es.stage1_damp.data import SplitImageDataset
from damp_es.stage1_damp.model import DAMPWrapper


class FeatureExporter:
    def __init__(self, model: DAMPWrapper, output_dir: str | Path):
        self.model = model
        self.output_dir = ensure_dir(output_dir)

    @torch.no_grad()
    def export(self, dataloader: DataLoader) -> None:
        for batch in dataloader:
            images = batch["image"].to(self.model.device)
            sample_ids = batch["sample_id"]

            global_features = self.model.encode_image(images).cpu().numpy()
            # Hooked feature map from the selected CLIP transformer layer.
            feature_map = self.model.get_feature_map().cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                np.savez_compressed(
                    self.output_dir / f"{sample_id}.npz",
                    global_feature=global_features[i],
                    feature_map=feature_map[i],
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Stage 1 adapted features")
    parser.add_argument("--config", type=str, default="configs/stage1_damp.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/stage1/features")
    parser.add_argument(
        "--stage1-ckpt-root",
        type=str,
        default="",
        help="Optional stage1 checkpoint root (contains adapters/model-best.pth.tar)",
    )
    return parser.parse_args()


def build_domain_root(cfg: Dict) -> Path:
    data_root = Path(cfg["dataset"]["root"])
    source_domain = cfg["dataset"]["source_domain"]
    return data_root / source_domain


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_overrides(cfg, parse_overrides(args.override))

    model = DAMPWrapper(
        backbone=cfg["backbone"]["name"],
        clip_weights=cfg["backbone"].get("pretrained_model"),
        device="cuda",
        feature_layer=-1,
        enable_mutual_prompting=True,
    )

    ckpt_root = args.stage1_ckpt_root or str(cfg.get("training", {}).get("output_dir", ""))
    if ckpt_root:
        ckpt_path = Path(ckpt_root)
        if ckpt_path.exists():
            model.load_damp_prompt_checkpoints(ckpt_path)

    domain_root = build_domain_root(cfg)
    dataset = SplitImageDataset(domain_root=domain_root, split=args.split, image_size=224)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    exporter = FeatureExporter(model=model, output_dir=args.output_dir)
    exporter.export(dataloader)
    print(f"Saved Stage 1 features to: {args.output_dir}")


if __name__ == "__main__":
    main()
