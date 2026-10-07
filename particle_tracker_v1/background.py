from typing import List
from pathlib import Path
import numpy as np
from .io_utils import read_gray


def estimate_background(image_files: List[Path], sample_count: int = 80) -> np.ndarray:
    """Median background estimation from uniformly sampled frames."""
    n = len(image_files)
    if n == 0:
        raise ValueError("image_files 为空")
    sample_count = max(1, min(sample_count, n))
    idx = np.linspace(0, n - 1, sample_count).astype(int)
    stack = []
    for i in idx:
        stack.append(read_gray(image_files[i]).astype(np.float32))
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)


def make_pseudo_exposure(image_files: List[Path], background: np.ndarray | None = None) -> np.ndarray:
    """Create pseudo-exposure image. For dark spots, use max(background-frame)."""
    accum = None
    for p in image_files:
        frame = read_gray(p).astype(np.float32)
        if background is None:
            enhanced = 255.0 - frame
        else:
            enhanced = np.clip(background - frame, 0, 255)
        if accum is None:
            accum = enhanced
        else:
            accum = np.maximum(accum, enhanced)
    if accum is None:
        raise ValueError("无法生成伪曝光图")
    accum = np.clip(accum, 0, 255).astype(np.uint8)
    return accum
