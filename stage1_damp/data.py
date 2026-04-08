from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - PIL compatibility
    RESAMPLE_BILINEAR = Image.BILINEAR


WEAK_PATTERN = re.compile(r"\[([01])\s*([01])\s*([01])\s*([01])\]$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_transformed_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), RESAMPLE_BILINEAR)
    image_tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(
        image_tensor,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return image_tensor


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    index: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("._"):
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index.setdefault(path.stem, path)

    if not index:
        raise RuntimeError(f"No valid image files found in: {images_dir}")
    return index


def resolve_image_path(images_dir: Path, sample_id: str, image_index: Dict[str, Path] | None = None) -> Path:
    if image_index is not None:
        image_path = image_index.get(sample_id)
        if image_path is not None:
            return image_path

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("._"):
            continue
        if path.stem == sample_id:
            return path

    raise FileNotFoundError(f"Cannot find image for sample id: {sample_id}")


def load_split_ids(split_path: Path) -> List[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")

    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_weak_label_map(csv_path: Path) -> Dict[str, Tuple[int, int]]:
    if not csv_path.exists():
        return {}

    weak_map: Dict[str, Tuple[int, int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id")
            if not sample_id:
                continue
            tumor = int(row.get("tumor", 0))
            stroma = int(row.get("stroma", 0))
            weak_map[sample_id] = (tumor, stroma)
    return weak_map


def parse_flags_from_sample_id(sample_id: str) -> Tuple[int, int] | None:
    match = WEAK_PATTERN.search(sample_id)
    if match is None:
        return None

    bits = [int(v) for v in match.groups()]
    has_spaces = " " in sample_id
    tumor = bits[0]
    # LUAD names carry [a b c d] and preprocessing maps stroma from d.
    stroma = bits[3] if has_spaces else bits[1]
    return tumor, stroma


def flags_to_label(tumor: int, stroma: int) -> int:
    tumor = int(tumor > 0)
    stroma = int(stroma > 0)
    if tumor and stroma:
        return 3
    if tumor:
        return 1
    if stroma:
        return 2
    return 0


class SplitImageDataset(Dataset):
    """Read images from CrossDomainSeg split file."""

    def __init__(self, domain_root: str | Path, split: str, image_size: int = 224):
        self.domain_root = Path(domain_root)
        self.split = split
        self.image_size = image_size
        self.ids = self._load_ids()
        self.images_dir = self.domain_root / "images"
        self.image_index = build_image_index(self.images_dir)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, str | torch.Tensor]:
        sample_id = self.ids[index]
        image_path = self._resolve_image(sample_id)
        image_tensor = build_transformed_image(image_path=image_path, image_size=self.image_size)
        return {"sample_id": sample_id, "image": image_tensor}

    def _load_ids(self) -> List[str]:
        split_path = self.domain_root / "splits" / f"{self.split}.txt"
        return load_split_ids(split_path)

    def _resolve_image(self, sample_id: str) -> Path:
        return resolve_image_path(
            images_dir=self.images_dir,
            sample_id=sample_id,
            image_index=self.image_index,
        )


class SourceWeakLabelDataset(Dataset):
    """Source split loader with weak class labels for Stage 1 adaptation."""

    def __init__(self, domain_root: str | Path, split: str = "train", image_size: int = 224):
        self.domain_root = Path(domain_root)
        self.split = split
        self.image_size = image_size
        self.images_dir = self.domain_root / "images"

        split_path = self.domain_root / "splits" / f"{self.split}.txt"
        self.ids = load_split_ids(split_path)
        weak_csv = self.domain_root / "metadata" / "train_weak_labels.csv"
        self.weak_map = load_weak_label_map(weak_csv)
        self.image_index = build_image_index(self.images_dir)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, str | torch.Tensor | int]:
        sample_id = self.ids[index]
        image_path = resolve_image_path(
            images_dir=self.images_dir,
            sample_id=sample_id,
            image_index=self.image_index,
        )
        image_tensor = build_transformed_image(image_path=image_path, image_size=self.image_size)
        label = self._resolve_label(sample_id)
        return {"sample_id": sample_id, "image": image_tensor, "label": label}

    def _resolve_label(self, sample_id: str) -> int:
        if sample_id in self.weak_map:
            tumor, stroma = self.weak_map[sample_id]
            return flags_to_label(tumor=tumor, stroma=stroma)

        parsed = parse_flags_from_sample_id(sample_id)
        if parsed is None:
            raise ValueError(f"Cannot resolve weak label for sample: {sample_id}")

        tumor, stroma = parsed
        return flags_to_label(tumor=tumor, stroma=stroma)
