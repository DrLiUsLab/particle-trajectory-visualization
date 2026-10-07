from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import cv2
import numpy as np
import yaml

from .io_utils import read_gray
from .detection import detect_dark_spots, Detection
from .auto_seed import generate_auto_seed_records, _parse_frame_indices


def _get_trackbar(win: str, name: str, default: int) -> int:
    try:
        return int(cv2.getTrackbarPos(name, win))
    except Exception:
        return int(default)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _cfg_from_trackbars(win: str, base_cfg: dict, frame_count: int) -> dict:
    cfg = copy.deepcopy(base_cfg)

    start_pct = _get_trackbar(win, "start_%", int(float(cfg.get("auto_seed_frame_start_ratio", 0.05)) * 100))
    end_pct = _get_trackbar(win, "end_%", int(float(cfg.get("auto_seed_frame_end_ratio", 0.95)) * 100))
    if end_pct < start_pct:
        start_pct, end_pct = end_pct, start_pct
    if end_pct == start_pct:
        end_pct = min(100, start_pct + 1)

    frame_count_slider = _get_trackbar(win, "seed_frames", int(cfg.get("auto_seed_frame_count", 9)))
    frame_count_slider = _clamp(frame_count_slider, 2, 50)

    min_area = _get_trackbar(win, "min_area", int(cfg.get("auto_seed_min_area_px", cfg.get("min_area_px", 5))))
    max_area = _get_trackbar(win, "max_area", int(cfg.get("auto_seed_max_area_px", cfg.get("max_area_px", 350))))
    if max_area <= min_area:
        max_area = min_area + 1

    min_dark = _get_trackbar(win, "min_dark", int(cfg.get("auto_seed_min_mean_dark_intensity", cfg.get("min_mean_dark_intensity", 8))))
    min_circ = _get_trackbar(win, "min_circ_x100", int(float(cfg.get("auto_seed_min_circularity", cfg.get("min_circularity", 0.2))) * 100)) / 100.0
    min_conf = _get_trackbar(win, "min_conf_x1000", int(float(cfg.get("auto_seed_min_confidence", 0.005)) * 1000)) / 1000.0
    min_dist = _get_trackbar(win, "min_dist", int(cfg.get("auto_seed_min_distance_px", 8)))
    max_per_frame = _get_trackbar(win, "max_per_frame", int(cfg.get("auto_seed_max_per_frame", 80)))
    max_total = _get_trackbar(win, "max_total_x10", int(cfg.get("auto_seed_max_total", 350)) // 10) * 10
    max_total = max(10, max_total)

    blur = _get_trackbar(win, "blur", int(cfg.get("blur_ksize", 3)))
    if blur <= 1:
        blur = 1
    elif blur % 2 == 0:
        blur += 1

    method_flag = _get_trackbar(win, "method 0O1M", 0 if str(cfg.get("threshold_method", "otsu")).lower() == "otsu" else 1)
    manual_threshold = _get_trackbar(win, "manual_th", int(cfg.get("manual_threshold", 25)))
    max_ar_x10 = _get_trackbar(win, "max_AR_x10", int(float(cfg.get("max_aspect_ratio", 4.0)) * 10))

    cfg["auto_seed_frame_indices"] = "auto"
    cfg["auto_seed_frame_start_ratio"] = float(start_pct) / 100.0
    cfg["auto_seed_frame_end_ratio"] = float(end_pct) / 100.0
    cfg["auto_seed_frame_count"] = int(frame_count_slider)

    # Use the same candidate-quality thresholds for the seed filter.
    cfg["auto_seed_min_area_px"] = float(min_area)
    cfg["auto_seed_max_area_px"] = float(max_area)
    cfg["auto_seed_min_mean_dark_intensity"] = float(min_dark)
    cfg["auto_seed_min_circularity"] = float(min_circ)
    cfg["auto_seed_min_confidence"] = float(min_conf)
    cfg["auto_seed_min_distance_px"] = float(min_dist)
    cfg["auto_seed_max_per_frame"] = int(max_per_frame)
    cfg["auto_seed_max_total"] = int(max_total)

    # Synchronize with frame-level detection, otherwise the seed filter may never see
    # candidates that were filtered out earlier by detect_dark_spots().
    cfg["min_area_px"] = float(min_area)
    cfg["max_area_px"] = float(max_area)
    cfg["min_mean_dark_intensity"] = float(min_dark)
    cfg["min_circularity"] = float(min_circ)
    cfg["max_aspect_ratio"] = float(max(1, max_ar_x10)) / 10.0
    cfg["blur_ksize"] = int(blur)
    cfg["threshold_method"] = "manual" if method_flag == 1 else "otsu"
    cfg["manual_threshold"] = int(manual_threshold)
    return cfg


def _draw_seed_preview(
    image_gray: np.ndarray,
    frame_id: int,
    detections: Sequence[Detection],
    seed_records: Sequence[object],
    roi_mask: Optional[np.ndarray],
    cfg: dict,
    current_index: int,
    preview_frames: Sequence[int],
) -> np.ndarray:
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR) if image_gray.ndim == 2 else image_gray.copy()
    if roi_mask is not None:
        color = np.zeros_like(base)
        color[:, :, 1] = 255
        tint = cv2.addWeighted(base, 0.84, color, 0.16, 0)
        base = np.where(roi_mask[:, :, None] > 0, tint, base).astype(np.uint8)

    # Show all accepted detections on this frame as faint blue circles.
    for d in detections:
        x, y = int(round(d.x)), int(round(d.z))
        cv2.circle(base, (x, y), 4, (255, 120, 0), 1, cv2.LINE_AA)

    seeds_here = [s for s in seed_records if int(getattr(s, "seed_frame")) == int(frame_id)]
    for i, s in enumerate(seeds_here, start=1):
        x, y = int(round(float(getattr(s, "x")))), int(round(float(getattr(s, "z"))))
        cv2.circle(base, (x, y), 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(base, (x, y), 10, (255, 255, 255), 1, cv2.LINE_AA)

    txt_lines = [
        "Auto-seed tuner | sliders update preview | N/P switch frame | Enter accept | Esc cancel",
        f"Frame {frame_id} ({current_index+1}/{len(preview_frames)}) | det={len(detections)} | seeds_here={len(seeds_here)} | seeds_total={len(seed_records)}",
        f"range={cfg['auto_seed_frame_start_ratio']:.2f}-{cfg['auto_seed_frame_end_ratio']:.2f}, frames={cfg['auto_seed_frame_count']}, area={cfg['auto_seed_min_area_px']:.0f}-{cfg['auto_seed_max_area_px']:.0f}, dark>={cfg['auto_seed_min_mean_dark_intensity']:.0f}, circ>={cfg['auto_seed_min_circularity']:.2f}",
        f"method={cfg['threshold_method']}, manual_th={cfg['manual_threshold']}, blur={cfg['blur_ksize']}, min_dist={cfg['auto_seed_min_distance_px']:.0f}, max/frame={cfg['auto_seed_max_per_frame']}",
    ]
    y = 24
    for line in txt_lines:
        cv2.putText(base, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(base, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 23
    return base


def tune_auto_seed_parameters_interactive(
    image_files: Sequence[Path],
    background: np.ndarray,
    roi_mask: Optional[np.ndarray],
    cfg: dict,
    out_dir: Path,
    window_name: str = "Auto seed parameter tuner",
) -> dict:
    """Interactive OpenCV trackbar panel for auto-seed parameter tuning.

    The function returns an updated config dict. It only previews seed extraction on the
    selected seed frames; the full all-frame detection still runs later in main.py.
    """
    if len(image_files) == 0:
        return cfg

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cfg.get("tuner_window_width", 1280)), int(cfg.get("tuner_window_height", 850)))

    def noop(_):
        pass

    cv2.createTrackbar("start_%", window_name, int(float(cfg.get("auto_seed_frame_start_ratio", 0.05)) * 100), 100, noop)
    cv2.createTrackbar("end_%", window_name, int(float(cfg.get("auto_seed_frame_end_ratio", 0.95)) * 100), 100, noop)
    cv2.createTrackbar("seed_frames", window_name, int(cfg.get("auto_seed_frame_count", 9)), 50, noop)
    cv2.createTrackbar("min_area", window_name, int(cfg.get("auto_seed_min_area_px", cfg.get("min_area_px", 5))), 500, noop)
    cv2.createTrackbar("max_area", window_name, int(cfg.get("auto_seed_max_area_px", cfg.get("max_area_px", 350))), 1000, noop)
    cv2.createTrackbar("min_dark", window_name, int(cfg.get("auto_seed_min_mean_dark_intensity", cfg.get("min_mean_dark_intensity", 8))), 100, noop)
    cv2.createTrackbar("min_circ_x100", window_name, int(float(cfg.get("auto_seed_min_circularity", cfg.get("min_circularity", 0.2))) * 100), 100, noop)
    cv2.createTrackbar("min_conf_x1000", window_name, int(float(cfg.get("auto_seed_min_confidence", 0.005)) * 1000), 100, noop)
    cv2.createTrackbar("min_dist", window_name, int(cfg.get("auto_seed_min_distance_px", 8)), 100, noop)
    cv2.createTrackbar("max_per_frame", window_name, int(cfg.get("auto_seed_max_per_frame", 80)), 300, noop)
    cv2.createTrackbar("max_total_x10", window_name, int(cfg.get("auto_seed_max_total", 350)) // 10, 100, noop)
    cv2.createTrackbar("blur", window_name, int(cfg.get("blur_ksize", 3)), 15, noop)
    cv2.createTrackbar("method 0O1M", window_name, 0 if str(cfg.get("threshold_method", "otsu")).lower() == "otsu" else 1, 1, noop)
    cv2.createTrackbar("manual_th", window_name, int(cfg.get("manual_threshold", 25)), 100, noop)
    cv2.createTrackbar("max_AR_x10", window_name, int(float(cfg.get("max_aspect_ratio", 4.0)) * 10), 100, noop)

    current_frame_list: List[int] = []
    current_idx = 0
    last_signature = None
    preview_dets: Dict[int, List[Detection]] = {}
    preview_seeds = []
    accepted_cfg = copy.deepcopy(cfg)

    print("自动种子滑块调参：N/P 切换预览帧；Enter 接受当前参数；Esc 取消并保留原参数。")
    print("蓝色小圈=当前检测到的暗斑候选；绿色实心点=最终作为种子的点。")

    while True:
        tuned = _cfg_from_trackbars(window_name, cfg, len(image_files))
        frames = _parse_frame_indices(len(image_files), tuned)
        if not frames:
            frames = [0]
        signature = (
            tuple(frames),
            tuned.get("min_area_px"), tuned.get("max_area_px"), tuned.get("min_mean_dark_intensity"),
            tuned.get("min_circularity"), tuned.get("max_aspect_ratio"), tuned.get("blur_ksize"),
            tuned.get("threshold_method"), tuned.get("manual_threshold"),
            tuned.get("auto_seed_min_confidence"), tuned.get("auto_seed_min_distance_px"),
            tuned.get("auto_seed_max_per_frame"), tuned.get("auto_seed_max_total"),
        )
        if signature != last_signature:
            current_frame_list = list(frames)
            current_idx = min(current_idx, len(current_frame_list) - 1)
            preview_dets = {}
            for f in current_frame_list:
                img = read_gray(image_files[int(f)])
                preview_dets[int(f)] = detect_dark_spots(img, background, int(f), roi_mask, tuned)
            preview_seeds, audit = generate_auto_seed_records(preview_dets, len(image_files), tuned)
            if out_dir is not None:
                try:
                    audit.to_csv(out_dir / "auto_seed_tuner_preview_audit.csv", index=False, encoding="utf-8-sig")
                except Exception:
                    pass
            last_signature = signature

        frame_id = current_frame_list[current_idx]
        img = read_gray(image_files[int(frame_id)])
        canvas = _draw_seed_preview(img, int(frame_id), preview_dets.get(int(frame_id), []), preview_seeds, roi_mask, tuned, current_idx, current_frame_list)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(80) & 0xFF
        if key in (13, 10):
            accepted_cfg = tuned
            cv2.destroyWindow(window_name)
            try:
                with open(out_dir / "auto_seed_tuned_config.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(accepted_cfg, f, allow_unicode=True, sort_keys=False)
            except Exception:
                pass
            return accepted_cfg
        if key == 27:
            cv2.destroyWindow(window_name)
            return cfg
        if key in (ord("n"), ord("N"), 83):
            current_idx = (current_idx + 1) % len(current_frame_list)
        if key in (ord("p"), ord("P"), 81):
            current_idx = (current_idx - 1) % len(current_frame_list)
