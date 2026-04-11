from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - PIL compatibility
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


@dataclass(frozen=True)
class CrossDomainSample:
    sample_id: str
    image_path: Path
    mask_path: Optional[Path]
    confidence_path: Optional[Path]


class CrossDomainSegDataset(Dataset):
    """Dataset for CrossDomainSeg splits using sample ids from split files."""

    def __init__(
        self,
        domain_root: str | Path,
        split: str,
        image_size: int = 512,
        pseudo_mask_dir: str | Path | None = None,
        aug_cfg: dict | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        self.domain_root = Path(domain_root)
        self.split = split
        self.image_size = image_size
        self.pseudo_mask_dir = Path(pseudo_mask_dir) if pseudo_mask_dir else None
        self.aug_cfg = aug_cfg or {}
        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        
        if sample.mask_path is not None:
            if not sample.mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {sample.mask_path}")
            mask = Image.open(sample.mask_path)
        else:
            mask = Image.fromarray(np.full((image.height, image.width), 255, dtype=np.uint8))

        if sample.confidence_path is not None and sample.confidence_path.exists():
            conf_np = np.load(sample.confidence_path).astype(np.float32)
            confidence = Image.fromarray(conf_np)
        else:
            confidence = Image.fromarray(np.ones((image.height, image.width), dtype=np.float32))

        if self.split == "train" and self.aug_cfg:
            from torchvision import transforms as T
            from torchvision.transforms import ColorJitter
            import torchvision.transforms.functional as F_v1
            
            # Color jitter (only image)
            if self.aug_cfg.get("color_jitter"):
                jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                image = jitter(image)
                
            # Random horizontal flip
            if self.aug_cfg.get("random_flip") and torch.rand(1).item() > 0.5:
                image = F_v1.hflip(image)
                mask = F_v1.hflip(mask)
                confidence = F_v1.hflip(confidence)
                
            # Random crop
            if self.aug_cfg.get("random_crop"):
                scale_size = int(self.image_size * 1.1)
                image = F_v1.resize(image, (scale_size, scale_size), interpolation=RESAMPLE_BILINEAR)
                mask = F_v1.resize(mask, (scale_size, scale_size), interpolation=RESAMPLE_NEAREST)
                confidence = F_v1.resize(confidence, (scale_size, scale_size), interpolation=RESAMPLE_BILINEAR)
                
                i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.image_size, self.image_size))
                image = F_v1.crop(image, i, j, h, w)
                mask = F_v1.crop(mask, i, j, h, w)
                confidence = F_v1.crop(confidence, i, j, h, w)
            else:
                image = image.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)
                mask = mask.resize((self.image_size, self.image_size), RESAMPLE_NEAREST)
                confidence = confidence.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)
        else:
            image = image.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), RESAMPLE_NEAREST)
            confidence = confidence.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        confidence_tensor = torch.from_numpy(np.array(confidence, dtype=np.float32))

        return {
            "sample_id": sample.sample_id,
            "image": image_tensor,
            "mask": mask_tensor,
            "confidence": confidence_tensor,
        }

    def _load_samples(self) -> List[CrossDomainSample]:
        split_path = self.domain_root / "splits" / f"{self.split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        sample_ids: List[str] = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                sample_id = line.strip()
                if sample_id:
                    sample_ids.append(sample_id)

        images_dir = self.domain_root / "images"
        masks_dir = self.domain_root / "masks"

        samples: List[CrossDomainSample] = []
        for sample_id in sample_ids:
            image_path = self._resolve_image_path(images_dir, sample_id)

            mask_path = self._resolve_mask_path(sample_id, masks_dir)
            confidence_path = self._resolve_confidence_path(sample_id)
            samples.append(
                CrossDomainSample(
                    sample_id=sample_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    confidence_path=confidence_path,
                )
            )

        return samples

    def _resolve_image_path(self, images_dir: Path, sample_id: str) -> Path:
        # Avoid glob patterns because sample ids can include "[" and "]".
        candidate_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        for ext in candidate_exts:
            candidate = images_dir / f"{sample_id}{ext}"
            if candidate.exists():
                return candidate

        candidates = [
            p
            for p in images_dir.iterdir()
            if p.is_file()
            and p.stem == sample_id
            and not p.name.startswith(".")
            and not p.name.startswith("._")
        ]
        if candidates:
            return sorted(candidates)[0]

        raise FileNotFoundError(f"Image not found for sample id: {sample_id}")

    def _resolve_mask_path(self, sample_id: str, masks_dir: Path) -> Optional[Path]:
        if self.split == "train" and self.pseudo_mask_dir is not None:
            pseudo_path = self.pseudo_mask_dir / f"{sample_id}.png"
            return pseudo_path

        gt_mask_path = masks_dir / f"{sample_id}.png"
        if gt_mask_path.exists():
            return gt_mask_path
        return None

    def _resolve_confidence_path(self, sample_id: str) -> Optional[Path]:
        if self.split != "train" or self.pseudo_mask_dir is None:
            return None
        confidence_path = self.pseudo_mask_dir / "confidence" / f"{sample_id}.npy"
        if confidence_path.exists():
            return confidence_path
        return None
