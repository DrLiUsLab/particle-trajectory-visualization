from pathlib import Path
from typing import List
import cv2
import numpy as np


def list_image_files(input_dir: str, extensions: List[str]) -> List[Path]:
    folder = Path(input_dir)
    if not folder.exists():
        raise FileNotFoundError(f"输入图像文件夹不存在: {folder.resolve()}")
    exts = {e.lower() for e in extensions}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    files = sorted(files, key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"未在 {folder.resolve()} 中找到图像文件")
    return files


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
