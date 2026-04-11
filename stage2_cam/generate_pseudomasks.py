from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from damp_es.common.config import apply_overrides, load_yaml_config, parse_overrides
from damp_es.common.io import ensure_dir
from damp_es.stage1_damp.model import DAMPWrapper
from damp_es.stage2_cam.caa import CAARefiner
from damp_es.stage2_cam.co_attention import BidirectionalCoAttentionRefiner
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


def preprocess_image_for_stage2(image_pil: Image.Image) -> torch.Tensor:
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


class Stage2ImageDataset(Dataset):
    def __init__(self, split_reader: TargetSplitReader, sample_ids: List[str]):
        self.split_reader = split_reader
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id = self.sample_ids[index]
        image_path = self.split_reader.resolve_image_path(sample_id)
        with Image.open(image_path) as image_file:
            image_pil = image_file.convert("RGB")
            orig_h, orig_w = int(image_pil.height), int(image_pil.width)
            image_tensor = preprocess_image_for_stage2(image_pil)

        return {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "orig_h": orig_h,
            "orig_w": orig_w,
            "image_tensor": image_tensor,
        }


def stage2_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    # Keep tensors as a list because each sample can have different HxW.
    return {
        "sample_id": [str(item["sample_id"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
        "orig_h": [int(item["orig_h"]) for item in batch],
        "orig_w": [int(item["orig_w"]) for item in batch],
        "image_tensor": [item["image_tensor"] for item in batch],
    }


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
        inferred_context_layers = int(model_cfg.get("context_decoder_layers", 2))
        inferred_class_names = None
        if use_damp_features and damp_ckpt:
            stage1_meta = self._load_stage1_meta(Path(damp_ckpt))
            if stage1_meta is not None:
                inferred_n_ctx = int(stage1_meta.get("n_ctx", inferred_n_ctx))
                inferred_context_layers = int(
                    stage1_meta.get("context_decoder_layers", inferred_context_layers)
                )
                class_names = stage1_meta.get("class_names")
                if isinstance(class_names, list) and class_names:
                    inferred_class_names = class_names

        self.wrapper = DAMPWrapper(
            backbone=model_cfg["clip_backbone"],
            clip_weights=model_cfg.get("clip_weights"),
            device=self.device,
            feature_layer=int(model_cfg.get("feature_layer", -1)),
            n_ctx=inferred_n_ctx,
            context_decoder_layers=inferred_context_layers,
            class_names=inferred_class_names,
            enable_mutual_prompting=use_damp_features,
        )
        if use_damp_features:
            if damp_ckpt:
                self.wrapper.load_damp_prompt_checkpoints(damp_ckpt)

        print(f"[Stage2] device={self.wrapper.device}")
        self.stage2_use_amp = bool(runtime_cfg.get("amp", True))
        self.stage2_amp_dtype = _resolve_amp_dtype(str(runtime_cfg.get("amp_dtype", "bf16")))
        self.io_batch_size = max(1, int(runtime_cfg.get("batch_size", 8)))
        self.io_workers = max(0, int(runtime_cfg.get("io_workers", 32)))
        self.io_pin_memory = bool(runtime_cfg.get("pin_memory", True))
        self.io_prefetch_factor = max(1, int(runtime_cfg.get("prefetch_factor", 4)))
        self.io_persistent_workers = bool(runtime_cfg.get("persistent_workers", True))
        self.skip_existing = bool(output_cfg.get("skip_existing", True))
        print(
            f"[Stage2] amp={bool(self.stage2_use_amp and self.wrapper.device.type == 'cuda')} "
            f"amp_dtype={self.stage2_amp_dtype} tf32={bool(runtime_cfg.get('allow_tf32', True))} "
            f"io_batch_size={self.io_batch_size} io_workers={self.io_workers}"
        )

        prompts_cfg = cfg["prompts"]
        prompt_strategy_cfg = prompts_cfg.get("strategy", {})
        self.prompt_manager = PromptManager(
            template=prompts_cfg["template"],
            class_synonyms=prompts_cfg["classes"],
            background_names=prompts_cfg["background"],
            use_sharpness_selection=bool(prompt_strategy_cfg.get("use_sharpness_selection", True)),
            use_synonym_fusion=bool(prompt_strategy_cfg.get("use_synonym_fusion", True)),
            extra_templates=list(prompt_strategy_cfg.get("extra_templates", [])),
            sharpness_eps=float(prompt_strategy_cfg.get("sharpness_eps", 1e-6)),
        )
        self.prompt_bundle = self.prompt_manager.build(self.wrapper)
        if self.prompt_bundle.class_prompt_map:
            prompt_log = ", ".join(
                [f"{k}: '{v}'" for k, v in self.prompt_bundle.class_prompt_map.items()]
            )
            print(f"[Stage2] selected class prompts -> {prompt_log}")

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

        refinement_cfg = cfg.get("refinement", {})
        self.refinement_mode = str(refinement_cfg.get("mode", "caa")).strip().lower()
        coattn_cfg = refinement_cfg.get("co_attention", {})
        self.co_attention_refiner = BidirectionalCoAttentionRefiner(
            alpha=float(coattn_cfg.get("alpha", 0.6)),
            beta=float(coattn_cfg.get("beta", 0.4)),
            blend=float(coattn_cfg.get("blend", 0.5)),
            temperature=float(coattn_cfg.get("temperature", 0.07)),
        )
        if self.refinement_mode not in {"caa", "co_attention", "hybrid"}:
            raise ValueError(
                f"Unsupported refinement.mode='{self.refinement_mode}'. Expected one of: caa, co_attention, hybrid"
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

        context_state = checkpoint.get("context_decoder")
        if isinstance(context_state, dict):
            decoder_layer_ids: List[int] = []
            for key in context_state.keys():
                parts = key.split(".")
                if len(parts) >= 3 and parts[0] == "decoder" and parts[1].isdigit():
                    decoder_layer_ids.append(int(parts[1]))
            if decoder_layer_ids:
                out["context_decoder_layers"] = max(decoder_layer_ids) + 1

        return out if out else None

    def run(self) -> None:
        split_reader = TargetSplitReader(
            domain_root=self.paths.domain_root,
            split=self.cfg["dataset"]["split"],
        )
        sample_ids = split_reader.list_samples()
        sample_ids = self._filter_pending_samples(sample_ids)

        if not sample_ids:
            print("[Stage2] all samples already processed; nothing to do.")
            return

        dataset = Stage2ImageDataset(split_reader=split_reader, sample_ids=sample_ids)
        loader_kwargs = {
            "batch_size": self.io_batch_size,
            "shuffle": False,
            "num_workers": self.io_workers,
            "pin_memory": self.io_pin_memory,
            "collate_fn": stage2_collate_fn,
        }
        if self.io_workers > 0:
            loader_kwargs["persistent_workers"] = self.io_persistent_workers
            loader_kwargs["prefetch_factor"] = self.io_prefetch_factor
        loader = DataLoader(dataset, **loader_kwargs)

        for batch in tqdm(loader, total=len(loader), desc="Stage2 pseudomask generation"):
            for sample_id, image_path, orig_h, orig_w, image_tensor in zip(
                batch["sample_id"],
                batch["image_path"],
                batch["orig_h"],
                batch["orig_w"],
                batch["image_tensor"],
            ):
                image_path = Path(image_path)
                orig_h = int(orig_h)
                orig_w = int(orig_w)
                image_tensor = image_tensor.unsqueeze(0).to(self.wrapper.device, non_blocking=True)

                image_np = None
                if self.use_crf and self.crf_refiner is not None:
                    with Image.open(image_path) as image_file:
                        image_np = np.array(image_file.convert("RGB"))

                fg_cams = []
                cam_results = self.cam_generator.compute_for_classes(
                    image=image_tensor,
                    class_indices=self.prompt_bundle.class_to_full_indices,
                    tokenized_prompts=self.prompt_bundle.tokenized_full,
                    use_mutual_text=True,
                )

                for cam_result in cam_results:
                    cam = cam_result.cam
                    affinity = cam_result.affinity
                    token_side = int(round(float(affinity.shape[0]) ** 0.5))
                    cam_low = F.interpolate(
                        cam.unsqueeze(0).unsqueeze(0),
                        size=(token_side, token_side),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

                    use_caa = self.refinement_mode in {"caa", "hybrid"} and bool(
                        self.cfg["caa"].get("enabled", True)
                    )
                    use_coattn = self.refinement_mode in {"co_attention", "hybrid"}

                    if use_caa:
                        affinity = cam_result.affinity
                        cam_low = self.caa_refiner.refine(cam_2d=cam_low, affinity=affinity)

                    if use_coattn:
                        cam_low = self.co_attention_refiner.refine(
                            cam_2d=cam_low,
                            patch_features=cam_result.patch_features,
                            text_feature=cam_result.text_feature,
                        )

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
                max_fg = np.max(fg_prob, axis=0)
                cam_confidence = np.maximum(max_fg, 1.0 - max_fg).astype(np.float32)
                bg_prob = np.maximum(0.0, 1.0 - np.max(fg_prob, axis=0, keepdims=True))
                probs = np.concatenate([bg_prob, fg_prob], axis=0).astype(np.float32)
                probs = probs / (probs.sum(axis=0, keepdims=True) + 1e-8)

                if self.use_crf and self.crf_refiner is not None:
                    if image_np is None:
                        with Image.open(image_path) as image_file:
                            image_np = np.array(image_file.convert("RGB"))
                    probs = self.crf_refiner.refine(image_rgb=image_np, probs=probs)

                pred = np.argmax(probs, axis=0).astype(np.uint8)
                confidence = np.max(probs, axis=0).astype(np.float32)
                conf_threshold = float(self.cfg["confidence"].get("threshold", 0.40))
                pred[confidence < conf_threshold] = 255

                Image.fromarray(pred, mode="L").save(self.paths.pseudomask_dir / f"{sample_id}.png")
                np.save(self.paths.confidence_dir / f"{sample_id}.npy", confidence)
                np.save(
                    self.paths.cam_dir / f"{sample_id}.npy",
                    {
                        "class_names": self.prompt_bundle.class_names,
                        "prompts": self.prompt_bundle.full_phrases,
                        "probs": probs,
                        "confidence": confidence,
                        "cam_confidence": cam_confidence,
                    },
                    allow_pickle=True,
                )

    def _filter_pending_samples(self, sample_ids: List[str]) -> List[str]:
        if not self.skip_existing:
            return sample_ids

        pending: List[str] = []
        skipped = 0
        for sample_id in sample_ids:
            mask_path = self.paths.pseudomask_dir / f"{sample_id}.png"
            conf_path = self.paths.confidence_dir / f"{sample_id}.npy"
            cam_path = self.paths.cam_dir / f"{sample_id}.npy"
            if mask_path.exists() and conf_path.exists() and cam_path.exists():
                skipped += 1
            else:
                pending.append(sample_id)

        if skipped > 0:
            print(f"[Stage2] skip_existing enabled: skipped {skipped} completed samples")
        print(f"[Stage2] pending samples: {len(pending)}")
        return pending

    def _preprocess_image(self, image_pil: Image.Image) -> torch.Tensor:
        return preprocess_image_for_stage2(image_pil)


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
