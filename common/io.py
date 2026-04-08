from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(root: str | Path) -> List[Path]:
    root_path = Path(root)
    files = [
        p
        for p in root_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    ]
    return sorted(files)


def read_lines(path: str | Path) -> List[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def write_lines(path: str | Path, lines: Sequence[str]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def copy_file(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    shutil.copy2(src_path, dst_path)
