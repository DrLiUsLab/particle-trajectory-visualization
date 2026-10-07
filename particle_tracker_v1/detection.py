from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd


@dataclass
class Detection:
    frame: int
    x: float
    z: float
    area: float
    mean_intensity: float
    circularity: float
    aspect_ratio: float
    confidence: float


def _safe_roi_mask(shape: Tuple[int, int], roi: Optional[Union[Tuple[int, int, int, int], np.ndarray]], padding: int = 0) -> np.ndarray:
    h, w = shape
    if roi is None:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, :] = 255
        return mask
    if isinstance(roi, np.ndarray):
        mask = roi.astype(np.uint8)
        if mask.shape[:2] != (h, w):
            raise ValueError(f"ROI mask shape {mask.shape[:2]} does not match image shape {(h, w)}")
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if padding and padding > 0:
            k = int(padding) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, rw, rh = roi
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w, x + rw + padding)
    y1 = min(h, y + rh + padding)
    mask[y0:y1, x0:x1] = 255
    return mask


def detect_dark_spots(
    frame: np.ndarray,
    background: np.ndarray,
    frame_index: int,
    roi: Optional[Union[Tuple[int, int, int, int], np.ndarray]],
    cfg: dict,
) -> List[Detection]:
    """Detect dark particle spots by background - frame enhancement."""
    blur_ksize = int(cfg.get("blur_ksize", 3))
    if blur_ksize > 1 and blur_ksize % 2 == 1:
        frame_f = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0).astype(np.float32)
    else:
        frame_f = frame.astype(np.float32)

    enhanced = np.clip(background.astype(np.float32) - frame_f, 0, 255).astype(np.uint8)
    roi_mask = _safe_roi_mask(enhanced.shape, roi, int(cfg.get("detection_roi_padding_px", 0)))
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask)

    method = cfg.get("threshold_method", "otsu").lower()
    if method == "manual":
        th = int(cfg.get("manual_threshold", 25))
        _, binary = cv2.threshold(enhanced, th, 255, cv2.THRESH_BINARY)
    else:
        nonzero = enhanced[roi_mask > 0]
        if nonzero.size == 0:
            return []
        # Otsu on ROI pixels; then apply threshold globally.
        otsu_img = nonzero.reshape(-1, 1).astype(np.uint8)
        th, _ = cv2.threshold(otsu_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary = cv2.threshold(enhanced, max(3, int(th)), 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dets: List[Detection] = []
    min_area = float(cfg.get("min_area_px", 4))
    max_area = float(cfg.get("max_area_px", 500))
    min_circ = float(cfg.get("min_circularity", 0.20))
    max_ar = float(cfg.get("max_aspect_ratio", 4.0))
    min_dark = float(cfg.get("min_mean_dark_intensity", 8.0))

    for lab in range(1, num):
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x0 = stats[lab, cv2.CC_STAT_LEFT]
        y0 = stats[lab, cv2.CC_STAT_TOP]
        ww = stats[lab, cv2.CC_STAT_WIDTH]
        hh = stats[lab, cv2.CC_STAT_HEIGHT]
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        if aspect > max_ar:
            continue

        comp = (labels == lab).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perim = cv2.arcLength(contours[0], True)
        circ = 4.0 * np.pi * area / (perim * perim + 1e-9)
        if circ < min_circ:
            continue

        cx, cy = centroids[lab]
        pix = enhanced[labels == lab]
        mean_int = float(np.mean(pix)) if pix.size else 0.0
        if mean_int < min_dark:
            continue
        conf = float(min(1.0, (mean_int / 255.0) * (circ / max(min_circ, 1e-6))))
        dets.append(Detection(frame_index, float(cx), float(cy), area, mean_int, circ, aspect, conf))

    return dets


def detections_to_dataframe(dets_by_frame: dict[int, List[Detection]]) -> pd.DataFrame:
    rows = []
    for frame, dets in dets_by_frame.items():
        for j, d in enumerate(dets):
            rows.append({
                "frame": frame,
                "candidate_id": j,
                "x_px": d.x,
                "z_px": d.z,
                "area_px": d.area,
                "mean_dark_intensity": d.mean_intensity,
                "circularity": d.circularity,
                "aspect_ratio": d.aspect_ratio,
                "confidence": d.confidence,
            })
    return pd.DataFrame(rows)
