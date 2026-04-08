from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from damp_es.common.io import copy_file, ensure_dir, list_images, write_lines
from damp_es.datasets.constants import BCSS_SPEC, LUAD_SPEC
from damp_es.datasets.label_mappers import MaskRemapper


@dataclass(frozen=True)
class DomainPaths:
    name: str
    raw_root: Path
    output_root: Path

    @property
    def train_dir(self) -> Path:
        return self.raw_root / "training"

    @property
    def val_img_dir(self) -> Path:
        return self.raw_root / "val" / "img"

    @property
    def val_mask_dir(self) -> Path:
        return self.raw_root / "val" / "mask"

    @property
    def test_img_dir(self) -> Path:
        return self.raw_root / "test" / "img"

    @property
    def test_mask_dir(self) -> Path:
        return self.raw_root / "test" / "mask"

    @property
    def out_images_dir(self) -> Path:
        return self.output_root / "images"

    @property
    def out_masks_dir(self) -> Path:
        return self.output_root / "masks"

    @property
    def out_splits_dir(self) -> Path:
        return self.output_root / "splits"

    @property
    def out_meta_dir(self) -> Path:
        return self.output_root / "metadata"


class WeakLabelParser:
    _bcss_re = re.compile(r"\[([01]{4})\]$")
    _luad_re = re.compile(r"\[([01])\s+([01])\s+([01])\s+([01])\]$")

    def parse(self, filename_stem: str) -> Tuple[int, int, int, int]:
        if " " in filename_stem:
            return self._parse_luad(filename_stem)
        return self._parse_bcss(filename_stem)

    def _parse_bcss(self, filename_stem: str) -> Tuple[int, int, int, int]:
        match = self._bcss_re.search(filename_stem)
        if match is None:
            raise ValueError(f"Cannot parse BCSS weak label from: {filename_stem}")
        bits = match.group(1)
        return tuple(int(v) for v in bits)

    def _parse_luad(self, filename_stem: str) -> Tuple[int, int, int, int]:
        match = self._luad_re.search(filename_stem)
        if match is None:
            raise ValueError(f"Cannot parse LUAD weak label from: {filename_stem}")
        return tuple(int(v) for v in match.groups())


class DomainPreprocessor:
    def __init__(self, paths: DomainPaths, remapper: MaskRemapper):
        self.paths = paths
        self.remapper = remapper
        self.weak_parser = WeakLabelParser()

    def run(self) -> None:
        ensure_dir(self.paths.output_root)
        ensure_dir(self.paths.out_images_dir)
        ensure_dir(self.paths.out_masks_dir)
        ensure_dir(self.paths.out_splits_dir)
        ensure_dir(self.paths.out_meta_dir)

        train_ids, weak_rows = self._process_train_split()
        val_ids = self._process_labeled_split(split_name="val")
        test_ids = self._process_labeled_split(split_name="test")

        write_lines(self.paths.out_splits_dir / "train.txt", train_ids)
        write_lines(self.paths.out_splits_dir / "val.txt", val_ids)
        write_lines(self.paths.out_splits_dir / "test.txt", test_ids)
        self._write_weak_labels_csv(weak_rows)

        print(
            f"[{self.paths.name}] prepared: "
            f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )

    def _process_train_split(self) -> Tuple[List[str], List[Dict[str, int]]]:
        if not self.paths.train_dir.exists():
            raise FileNotFoundError(f"Training dir not found: {self.paths.train_dir}")

        sample_ids: List[str] = []
        weak_rows: List[Dict[str, int]] = []

        for img_path in list_images(self.paths.train_dir):
            source_stem = img_path.stem
            sample_id = f"train_{source_stem}"
            out_image = self.paths.out_images_dir / f"{sample_id}{img_path.suffix.lower()}"
            copy_file(img_path, out_image)
            sample_ids.append(sample_id)

            a, b, c, d = self.weak_parser.parse(source_stem)
            weak_rows.append(
                {
                    "sample_id": sample_id,
                    "tumor": a,
                    "stroma": d if self.paths.name == "Hist" else b,
                    "lymphocyte": c,
                    "necrosis": b if self.paths.name == "Hist" else d,
                }
            )

        return sample_ids, weak_rows

    def _process_labeled_split(self, split_name: str) -> List[str]:
        if split_name not in {"val", "test"}:
            raise ValueError(f"Unsupported split: {split_name}")

        image_dir = self.paths.val_img_dir if split_name == "val" else self.paths.test_img_dir
        mask_dir = self.paths.val_mask_dir if split_name == "val" else self.paths.test_mask_dir

        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Missing labeled split directories: image={image_dir} mask={mask_dir}"
            )

        sample_ids: List[str] = []
        for image_path in list_images(image_dir):
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

            sample_id = f"{split_name}_{image_path.stem}"
            out_image = self.paths.out_images_dir / f"{sample_id}{image_path.suffix.lower()}"
            out_mask = self.paths.out_masks_dir / f"{sample_id}.png"
            copy_file(image_path, out_image)
            self._remap_and_save_mask(mask_path, out_mask)
            sample_ids.append(sample_id)

        return sample_ids

    def _remap_and_save_mask(self, raw_mask_path: Path, out_mask_path: Path) -> None:
        mask = np.array(Image.open(raw_mask_path))
        remapped = self.remapper.remap(mask)
        Image.fromarray(remapped, mode="L").save(out_mask_path)

    def _write_weak_labels_csv(self, rows: Sequence[Dict[str, int]]) -> None:
        out_csv = self.paths.out_meta_dir / "train_weak_labels.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "tumor", "stroma", "lymphocyte", "necrosis"],
            )
            writer.writeheader()
            writer.writerows(rows)


def _build_domain_paths(raw_data_root: Path, out_root: Path) -> List[DomainPaths]:
    return [
        DomainPaths(
            name="Hist",
            raw_root=raw_data_root / "LUAD-HistoSeg" / "LUAD-HistoSeg",
            output_root=out_root / "Hist",
        ),
        DomainPaths(
            name="BCSS",
            raw_root=raw_data_root / "BCSS-WSSS" / "BCSS-WSSS",
            output_root=out_root / "BCSS",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LUAD-HistoSeg and BCSS-WSSS into CrossDomainSeg format"
    )
    parser.add_argument(
        "--raw-data-root",
        type=str,
        default="data",
        help="Root containing raw LUAD-HistoSeg and BCSS-WSSS folders",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/CrossDomainSeg",
        help="Output root in CrossDomainSeg format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_data_root = Path(args.raw_data_root)
    output_root = Path(args.output_root)

    domain_paths = _build_domain_paths(raw_data_root=raw_data_root, out_root=output_root)
    spec_map = {"Hist": LUAD_SPEC, "BCSS": BCSS_SPEC}

    for paths in domain_paths:
        spec = spec_map[paths.name]
        preprocessor = DomainPreprocessor(paths=paths, remapper=MaskRemapper(spec))
        preprocessor.run()

    print("CrossDomainSeg preparation completed.")


if __name__ == "__main__":
    main()
