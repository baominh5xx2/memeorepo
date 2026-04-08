from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from damp_es.common.io import copy_file, ensure_dir, list_images


@dataclass(frozen=True)
class DomainSpec:
    name: str
    raw_root: Path
    out_root: Path


def _take_first_n(paths: List[Path], n: int) -> List[Path]:
    if n < 0:
        return paths
    return paths[:n]


def _copy_training(src_dir: Path, dst_dir: Path, n_train: int) -> int:
    ensure_dir(dst_dir)
    images = _take_first_n(list_images(src_dir), n_train)
    for image_path in images:
        copy_file(image_path, dst_dir / image_path.name)
    return len(images)


def _copy_labeled_split(src_split_dir: Path, dst_split_dir: Path, n_split: int) -> int:
    src_img_dir = src_split_dir / "img"
    src_mask_dir = src_split_dir / "mask"
    dst_img_dir = dst_split_dir / "img"
    dst_mask_dir = dst_split_dir / "mask"

    if not src_img_dir.exists() or not src_mask_dir.exists():
        raise FileNotFoundError(
            f"Missing split folders: img={src_img_dir} mask={src_mask_dir}"
        )

    ensure_dir(dst_img_dir)
    ensure_dir(dst_mask_dir)

    images = _take_first_n(list_images(src_img_dir), n_split)
    for image_path in images:
        mask_path = src_mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image {image_path.name}: {mask_path}"
            )
        copy_file(image_path, dst_img_dir / image_path.name)
        copy_file(mask_path, dst_mask_dir / mask_path.name)

    return len(images)


def _iter_domains(raw_data_root: Path, output_root: Path) -> Iterable[DomainSpec]:
    yield DomainSpec(
        name="Hist",
        raw_root=raw_data_root / "LUAD-HistoSeg" / "LUAD-HistoSeg",
        out_root=output_root / "LUAD-HistoSeg" / "LUAD-HistoSeg",
    )
    yield DomainSpec(
        name="BCSS",
        raw_root=raw_data_root / "BCSS-WSSS" / "BCSS-WSSS",
        out_root=output_root / "BCSS-WSSS" / "BCSS-WSSS",
    )


def build_raw_smoke(
    raw_data_root: Path,
    output_root: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    overwrite: bool,
) -> None:
    for domain in _iter_domains(raw_data_root=raw_data_root, output_root=output_root):
        if not domain.raw_root.exists():
            raise FileNotFoundError(f"Raw domain root not found: {domain.raw_root}")

        if overwrite and domain.out_root.exists():
            shutil.rmtree(domain.out_root)

        train_count = _copy_training(
            src_dir=domain.raw_root / "training",
            dst_dir=domain.out_root / "training",
            n_train=n_train,
        )
        val_count = _copy_labeled_split(
            src_split_dir=domain.raw_root / "val",
            dst_split_dir=domain.out_root / "val",
            n_split=n_val,
        )
        test_count = _copy_labeled_split(
            src_split_dir=domain.raw_root / "test",
            dst_split_dir=domain.out_root / "test",
            n_split=n_test,
        )

        print(
            f"[{domain.name}] raw_smoke created: "
            f"train={train_count} val={val_count} test={test_count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small raw smoke subset for LUAD-HistoSeg and BCSS-WSSS"
    )
    parser.add_argument(
        "--raw-data-root",
        type=str,
        default="data",
        help="Root containing full LUAD-HistoSeg and BCSS-WSSS raw datasets",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw_smoke",
        help="Destination root for the smoke raw subset",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=8,
        help="Number of training images per domain (-1 for all)",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=8,
        help="Number of validation image-mask pairs per domain (-1 for all)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=8,
        help="Number of test image-mask pairs per domain (-1 for all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output domain folders before copying",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_raw_smoke(
        raw_data_root=Path(args.raw_data_root),
        output_root=Path(args.output_root),
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()