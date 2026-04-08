from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from damp_es.common.config import apply_overrides, load_yaml_config, parse_overrides
from damp_es.common.io import ensure_dir
from damp_es.stage1_damp.model import DAMPWrapper
from damp_es.stage2_cam.caa import CAARefiner
from damp_es.stage2_cam.crf import CRFParams, DenseCRFRefiner
from damp_es.stage2_cam.prompts import PromptManager
from damp_es.stage2_cam.softmax_gradcam import SoftmaxGradCAM


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - PIL compatibility
    RESAMPLE_BILINEAR = Image.BILINEAR


@dataclass
class Stage2Paths:
    domain_root: Path
    split_file: Path
    cam_dir: Path
    pseudomask_dir: Path
    confidence_dir: Path


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.bfloat16


class TargetSplitReader:
    def __init__(self, domain_root: Path, split: str):
        self.domain_root = domain_root
        self.split = split
        self.image_dir = self.domain_root / "images"
        self.image_index = self._build_image_index()

    def _build_image_index(self) -> Dict[str, Path]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        index: Dict[str, Path] = {}
        for p in sorted(self.image_dir.iterdir()):
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name.startswith("._"):
                continue
            index.setdefault(p.stem, p)
        return index

    def list_samples(self) -> List[str]:
        split_file = self.domain_root / "splits" / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with split_file.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def resolve_image_path(self, sample_id: str) -> Path:
        image_dir = self.image_dir
        candidate_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        for ext in candidate_exts:
            candidate = image_dir / f"{sample_id}{ext}"
            if candidate.exists():
                return candidate

        indexed = self.image_index.get(sample_id)
        if indexed is not None:
            return indexed

        matches = [
            p
            for p in image_dir.iterdir()
            if p.is_file()
            and p.stem == sample_id
            and not p.name.startswith(".")
            and not p.name.startswith("._")
        ]
        if not matches:
            raise FileNotFoundError(f"Image not found for sample: {sample_id}")
        return sorted(matches)[0]


class PseudoMaskGenerator:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        runtime_cfg = cfg.get("runtime", {})

        if torch.cuda.is_available():
            allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
            try:
                torch.set_float32_matmul_precision(str(runtime_cfg.get("matmul_precision", "high")))
            except Exception:
                pass

        dataset_cfg = cfg["dataset"]
        output_cfg = cfg["output"]

        domain_root = Path(dataset_cfg["root"]) / dataset_cfg["target_domain"]
        self.paths = Stage2Paths(
            domain_root=domain_root,
            split_file=domain_root / "splits" / f"{dataset_cfg['split']}.txt",
            cam_dir=ensure_dir(output_cfg["cam_dir"]),
            pseudomask_dir=ensure_dir(output_cfg["pseudomask_dir"]),
            confidence_dir=ensure_dir(Path(output_cfg["pseudomask_dir"]) / "confidence"),
        )

        model_cfg = cfg["model"]
        use_damp_features = bool(model_cfg.get("use_damp_features", False))
        damp_ckpt = model_cfg.get("damp_ckpt")

        inferred_n_ctx = int(model_cfg.get("n_ctx", 4))
        inferred_class_names = None
        if use_damp_features and damp_ckpt:
            stage1_meta = self._load_stage1_meta(Path(damp_ckpt))
            if stage1_meta is not None:
                inferred_n_ctx = int(stage1_meta.get("n_ctx", inferred_n_ctx))
                class_names = stage1_meta.get("class_names")
                if isinstance(class_names, list) and class_names:
                    inferred_class_names = class_names

        self.wrapper = DAMPWrapper(
            backbone=model_cfg["clip_backbone"],
            clip_weights=model_cfg.get("clip_weights"),
            device=self.device,
            feature_layer=int(model_cfg.get("feature_layer", -1)),
            n_ctx=inferred_n_ctx,
            class_names=inferred_class_names,
            enable_mutual_prompting=use_damp_features,
        )
        if use_damp_features:
            if damp_ckpt:
                self.wrapper.load_damp_prompt_checkpoints(damp_ckpt)

        print(f"[Stage2] device={self.wrapper.device}")
        self.stage2_use_amp = bool(runtime_cfg.get("amp", True))
        self.stage2_amp_dtype = _resolve_amp_dtype(str(runtime_cfg.get("amp_dtype", "bf16")))
        print(
            f"[Stage2] amp={bool(self.stage2_use_amp and self.wrapper.device.type == 'cuda')} "
            f"amp_dtype={self.stage2_amp_dtype} tf32={bool(runtime_cfg.get('allow_tf32', True))}"
        )

        prompts_cfg = cfg["prompts"]
        self.prompt_manager = PromptManager(
            template=prompts_cfg["template"],
            class_synonyms=prompts_cfg["classes"],
            background_names=prompts_cfg["background"],
        )
        self.prompt_bundle = self.prompt_manager.build(self.wrapper)

        self.cam_generator = SoftmaxGradCAM(
            damp_wrapper=self.wrapper,
            use_softmax=bool(cfg["cam"].get("use_softmax_gradcam", True)),
            replace_cls_with_avg=bool(cfg["cam"].get("replace_cls_with_avg", True)),
            amp_enabled=self.stage2_use_amp,
            amp_dtype=self.stage2_amp_dtype,
        )

        self.caa_refiner = CAARefiner(
            threshold=float(cfg["caa"].get("threshold", 0.4)),
            n_iter=int(cfg["caa"].get("iterations", 2)),
        )

        crf_cfg = cfg["crf"]
        self.use_crf = bool(crf_cfg.get("enabled", True))
        self.crf_refiner = None
        if self.use_crf:
            self.crf_refiner = DenseCRFRefiner(
                CRFParams(
                    iter_max=int(crf_cfg.get("iter", 10)),
                    pos_w=int(crf_cfg.get("pos_w", 3)),
                    pos_xy_std=int(crf_cfg.get("pos_xy_std", 1)),
                    bi_w=int(crf_cfg.get("bi_w", 5)),
                    bi_xy_std=int(crf_cfg.get("bi_xy_std", 80)),
                    bi_rgb_std=int(crf_cfg.get("bi_rgb_std", 13)),
                )
            )

    @staticmethod
    def _load_stage1_meta(damp_ckpt_root: Path) -> Dict | None:
        mutual_ckpt = damp_ckpt_root / "mutual_prompt" / "model-best.pth.tar"
        if not mutual_ckpt.exists():
            return None

        try:
            checkpoint = torch.load(mutual_ckpt, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"Warning: failed to read Stage1 metadata from {mutual_ckpt}: {exc}")
            return None

        if not isinstance(checkpoint, dict):
            return None

        out: Dict = {}
        if "n_ctx" in checkpoint:
            out["n_ctx"] = checkpoint["n_ctx"]
        if "class_names" in checkpoint:
            out["class_names"] = checkpoint["class_names"]
        return out if out else None

    def run(self) -> None:
        split_reader = TargetSplitReader(
            domain_root=self.paths.domain_root,
            split=self.cfg["dataset"]["split"],
        )
        sample_ids = split_reader.list_samples()

        for sample_id in tqdm(sample_ids, desc="Stage2 pseudomask generation"):
            image_path = split_reader.resolve_image_path(sample_id)
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)
            image_tensor = self._preprocess_image(image_pil).unsqueeze(0).to(self.wrapper.device)
            orig_h, orig_w = int(image_np.shape[0]), int(image_np.shape[1])

            fg_cams = []
            for class_idx in self.prompt_bundle.class_to_full_indices:
                cam_result = self.cam_generator.compute(
                    image=image_tensor,
                    text_features=self.prompt_bundle.full_features,
                    class_index=class_idx,
                )

                cam = cam_result.cam
                if self.cfg["caa"].get("enabled", True):
                    affinity = cam_result.affinity
                    token_side = int(round(float(affinity.shape[0]) ** 0.5))
                    cam_low = F.interpolate(
                        cam.unsqueeze(0).unsqueeze(0),
                        size=(token_side, token_side),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                    cam_low = self.caa_refiner.refine(cam_2d=cam_low, affinity=affinity)
                    cam = F.interpolate(
                        cam_low.unsqueeze(0).unsqueeze(0),
                        size=cam.shape,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                fg_cams.append(cam.float().cpu().numpy())

            fg_prob = np.stack(fg_cams, axis=0)
            bg_prob = np.maximum(0.0, 1.0 - np.max(fg_prob, axis=0, keepdims=True))
            probs = np.concatenate([bg_prob, fg_prob], axis=0).astype(np.float32)
            probs = probs / (probs.sum(axis=0, keepdims=True) + 1e-8)

            if self.use_crf and self.crf_refiner is not None:
                probs = self.crf_refiner.refine(image_rgb=image_np, probs=probs)

            pred = np.argmax(probs, axis=0).astype(np.uint8)
            confidence = np.max(probs, axis=0)
            conf_threshold = float(self.cfg["confidence"].get("threshold", 0.25))
            pred[confidence < conf_threshold] = 255

            Image.fromarray(pred, mode="L").save(self.paths.pseudomask_dir / f"{sample_id}.png")
            np.save(self.paths.confidence_dir / f"{sample_id}.npy", confidence.astype(np.float32))
            np.save(
                self.paths.cam_dir / f"{sample_id}.npy",
                {
                    "class_names": self.prompt_bundle.class_names,
                    "probs": probs,
                    "confidence": confidence,
                },
                allow_pickle=True,
            )

    def _preprocess_image(self, image_pil: Image.Image) -> torch.Tensor:
        patch = 16
        h = int(image_pil.height)
        w = int(image_pil.width)
        new_h = int(np.ceil(h / patch) * patch)
        new_w = int(np.ceil(w / patch) * patch)
        image = image_pil.resize((new_w, new_h), RESAMPLE_BILINEAR)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_t = (image_t - mean) / std
        return image_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 2 pseudo-masks")
    parser.add_argument("--config", type=str, default="configs/stage2_cam.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg = apply_overrides(cfg, parse_overrides(args.override))

    generator = PseudoMaskGenerator(cfg)
    generator.run()
    print(f"Saved pseudo masks to: {generator.paths.pseudomask_dir}")


if __name__ == "__main__":
    main()
