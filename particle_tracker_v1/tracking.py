from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Any
import math
import numpy as np
import pandas as pd
from .detection import Detection
from .kalman import KalmanCA2D


@dataclass
class TrackPoint:
    frame: int
    x: float
    z: float
    vx: float
    vz: float
    ax: float
    az: float
    detected: bool
    confidence: float
    area: float | None = None
    mean_intensity: float | None = None
    missed_count: int = 0
    parabola_dist_px: float | None = None


def _inside_mask(x: float, z: float, roi_mask: Optional[np.ndarray]) -> bool:
    """Hard ROI constraint. Coordinates outside the polygon mask are invalid."""
    if roi_mask is None:
        return True
    h, w = roi_mask.shape[:2]
    xi = int(round(float(x)))
    zi = int(round(float(z)))
    if xi < 0 or xi >= w or zi < 0 or zi >= h:
        return False
    return bool(roi_mask[zi, xi] > 0)


def _nearest_detection(
    dets: List[Detection],
    seed_xy: Tuple[float, float],
    max_dist: Optional[float] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> Optional[Detection]:
    valid = [d for d in dets if _inside_mask(d.x, d.z, roi_mask)]
    if not valid:
        return None
    xy = np.array([[d.x, d.z] for d in valid], dtype=float)
    dist = np.linalg.norm(xy - np.array(seed_xy, dtype=float), axis=1)
    j = int(np.argmin(dist))
    if max_dist is not None and dist[j] > float(max_dist):
        return None
    return valid[j]


def _robust_poly_predict(
    history: List[Tuple[int, float, float]],
    frame: int,
    cfg: dict,
) -> Optional[Tuple[np.ndarray, float]]:
    """Predict x,z at frame using a local time-parametric parabola.

    This is the key V1.1.2 change: we no longer reject a point because of a large
    local turning angle. A parabolic trajectory naturally turns around near the apex.
    Instead, a candidate is rejected only when it is inconsistent with the local
    quadratic prediction built from recent real detections.
    """
    if not bool(cfg.get("enable_local_parabola_gate", True)):
        return None

    min_pts = int(cfg.get("local_parabola_min_points", 4))
    win = int(cfg.get("local_parabola_window_points", 7))
    pts = [(int(f), float(x), float(z)) for f, x, z in history if np.isfinite(x) and np.isfinite(z)]
    if len(pts) < min_pts:
        return None
    pts = pts[-win:]

    frames = np.array([p[0] for p in pts], dtype=float)
    # Need at least 3 distinct frame numbers for quadratic fit.
    if len(np.unique(frames)) < 3:
        return None
    x = np.array([p[1] for p in pts], dtype=float)
    z = np.array([p[2] for p in pts], dtype=float)

    t0 = frames[-1]
    tau = frames - t0
    target_tau = float(frame) - t0

    # Use quadratic if possible. If the window is still weak, fall back to linear.
    degree = 2 if len(pts) >= 3 else 1
    try:
        px = np.polyfit(tau, x, degree)
        pz = np.polyfit(tau, z, degree)
        pred = np.array([np.polyval(px, target_tau), np.polyval(pz, target_tau)], dtype=float)
        x_fit = np.polyval(px, tau)
        z_fit = np.polyval(pz, tau)
        rmse = float(np.sqrt(np.mean((x - x_fit) ** 2 + (z - z_fit) ** 2)))
        return pred, rmse
    except Exception:
        return None


def _angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _candidate_cost(
    det: Detection,
    pred_xy: np.ndarray,
    last_xy: np.ndarray,
    last_area: Optional[float],
    last_intensity: Optional[float],
    cfg: dict,
    parabola_pred: Optional[np.ndarray] = None,
    parabola_rmse: Optional[float] = None,
) -> float:
    det_xy = np.array([det.x, det.z], dtype=float)
    kalman_dist = float(np.linalg.norm(det_xy - pred_xy))

    parabola_cost = 0.0
    if parabola_pred is not None:
        parabola_dist = float(np.linalg.norm(det_xy - parabola_pred))
        # Normalize by local fit uncertainty so that noisy but consistent tracks are not over-penalized.
        norm = max(float(cfg.get("local_parabola_cost_norm_px", 8.0)), float(parabola_rmse or 0.0) + 1.0)
        parabola_cost = parabola_dist / norm

    # Keep step-length as a soft preference only. No hard turning-angle rejection is used in V1.1.2.
    new_motion = det_xy - last_xy
    step_len = float(np.linalg.norm(new_motion))
    expected_step = float(cfg.get("expected_step_px", 0.0))
    step_cost = 0.0
    if expected_step > 0:
        step_cost = abs(step_len - expected_step) / max(expected_step, 1.0)

    area_cost = 0.0
    if last_area and last_area > 0:
        area_cost = abs(det.area - last_area) / last_area
    intensity_cost = 0.0
    if last_intensity and last_intensity > 0:
        intensity_cost = abs(det.mean_intensity - last_intensity) / last_intensity

    return (
        kalman_dist
        + float(cfg.get("local_parabola_weight", 18.0)) * parabola_cost
        + float(cfg.get("step_change_weight", 1.0)) * step_cost
        + float(cfg.get("area_weight", 0.2)) * area_cost
        + float(cfg.get("intensity_weight", 0.2)) * intensity_cost
    )


def _candidate_passes_hard_gates(
    det: Detection,
    pred_xy: np.ndarray,
    last_xy: np.ndarray,
    roi_mask: Optional[np.ndarray],
    gate: float,
    cfg: dict,
    parabola_pred: Optional[np.ndarray] = None,
    parabola_rmse: Optional[float] = None,
) -> tuple[bool, str, float | None]:
    """Hard constraints: ROI, Kalman prediction gate, local parabolic consistency.

    Important: V1.1.2 intentionally removes the previous hard turn-angle/reverse-motion gates.
    A true ballistic/parabolic track can have a large apparent direction change near its top.
    """
    if not _inside_mask(det.x, det.z, roi_mask):
        return False, "outside_roi", None

    det_xy = np.array([det.x, det.z], dtype=float)
    pred_dist = float(np.linalg.norm(det_xy - pred_xy))
    if pred_dist > gate:
        return False, "outside_kalman_gate", None

    # Keep a loose single-frame jump gate only to reject remote noise/other particles.
    # It is deliberately not a turn-angle gate.
    step_len = float(np.linalg.norm(det_xy - last_xy))
    max_step = float(cfg.get("max_step_px_per_frame", 0.0))
    if max_step > 0 and step_len > max_step:
        return False, "step_too_large", None

    parabola_dist = None
    if parabola_pred is not None:
        parabola_dist = float(np.linalg.norm(det_xy - parabola_pred))
        base_gate = float(cfg.get("local_parabola_gate_px", 18.0))
        # Allow slightly wider gate when the recent local fit itself has measurable uncertainty.
        dyn_gate = base_gate + float(cfg.get("local_parabola_rmse_factor", 1.5)) * float(parabola_rmse or 0.0)
        dyn_gate = min(dyn_gate, float(cfg.get("local_parabola_max_gate_px", 45.0)))
        if parabola_dist > dyn_gate:
            return False, "outside_local_parabola_gate", parabola_dist

    return True, "ok", parabola_dist


def _estimate_initial_velocity(
    dets_by_frame: Dict[int, List[Detection]],
    seed_frame: int,
    seed_det: Detection,
    direction: int,
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Estimate velocity from adjacent frame if a plausible nearby detection exists."""
    next_frame = seed_frame + direction
    dets = dets_by_frame.get(next_frame, [])
    if not dets:
        return 0.0, 0.0
    seed_xy = np.array([seed_det.x, seed_det.z], dtype=float)
    gate = float(cfg.get("initial_velocity_gate_px", cfg.get("initial_gate_px", 25)))
    candidates = []
    for d in dets:
        if not _inside_mask(d.x, d.z, roi_mask):
            continue
        dxy = np.array([d.x, d.z], dtype=float)
        dist = float(np.linalg.norm(dxy - seed_xy))
        if dist <= gate:
            candidates.append((dist, d))
    if not candidates:
        return 0.0, 0.0
    _, d = min(candidates, key=lambda item: item[0])
    return (d.x - seed_det.x) * direction, (d.z - seed_det.z) * direction


def _init_filter_from_seed(
    dets_by_frame: Dict[int, List[Detection]],
    seed_frame: int,
    seed_det: Detection,
    direction: int,
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> KalmanCA2D:
    kf = KalmanCA2D(
        dt=1.0,
        process_noise=float(cfg.get("kalman_process_noise", 0.3)),
        measurement_noise=float(cfg.get("kalman_measurement_noise", 6.0)),
    )
    vx, vz = _estimate_initial_velocity(dets_by_frame, seed_frame, seed_det, direction, cfg, roi_mask)
    kf.initialize(seed_det.x, seed_det.z, vx=vx, vz=vz)
    return kf


def track_one_direction(
    dets_by_frame: Dict[int, List[Detection]],
    start_frame: int,
    end_frame_exclusive: int,
    direction: int,
    seed_det: Detection,
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> List[TrackPoint]:
    assert direction in (-1, 1)
    kf = _init_filter_from_seed(dets_by_frame, start_frame, seed_det, direction, cfg, roi_mask)
    points: List[TrackPoint] = []
    last_area = seed_det.area
    last_intensity = seed_det.mean_intensity
    missed = 0
    max_missed = int(cfg.get("max_missed_frames", 2))
    gate = float(cfg.get("initial_gate_px", 25))
    max_gate = float(cfg.get("max_gate_px", 45))
    stop_when_leave_roi = bool(cfg.get("stop_when_prediction_leaves_roi", True))
    allow_predicted_points = bool(cfg.get("allow_predicted_points_inside_roi", True))

    last_xy = np.array([seed_det.x, seed_det.z], dtype=float)
    # Local parabola is trained only from real detections, not predicted points.
    detected_history: List[Tuple[int, float, float]] = [(start_frame, float(seed_det.x), float(seed_det.z))]

    frame = start_frame
    while frame != end_frame_exclusive:
        if frame == start_frame:
            vx, vz = kf.velocity
            ax, az = kf.acceleration
            points.append(TrackPoint(frame, seed_det.x, seed_det.z, vx, vz, ax, az, True, seed_det.confidence, seed_det.area, seed_det.mean_intensity, 0, None))
        else:
            pred_xy = kf.predict()

            if stop_when_leave_roi and not _inside_mask(pred_xy[0], pred_xy[1], roi_mask):
                break

            local_pred = _robust_poly_predict(detected_history, frame, cfg)
            parabola_pred = local_pred[0] if local_pred is not None else None
            parabola_rmse = local_pred[1] if local_pred is not None else None

            dets = dets_by_frame.get(frame, [])
            candidates = []
            for det in dets:
                ok, reason, parabola_dist = _candidate_passes_hard_gates(
                    det, pred_xy, last_xy, roi_mask, gate, cfg, parabola_pred, parabola_rmse
                )
                if not ok:
                    continue
                cost = _candidate_cost(det, pred_xy, last_xy, last_area, last_intensity, cfg, parabola_pred, parabola_rmse)
                candidates.append((det, cost, parabola_dist))

            if candidates:
                det, cost, parabola_dist = min(candidates, key=lambda item: item[1])
                kf.update(det.x, det.z)
                last_area = det.area
                last_intensity = det.mean_intensity
                missed = 0
                gate = float(cfg.get("initial_gate_px", 25))
                detected = True
                conf = max(0.0, min(1.0, 1.0 - cost / max(max_gate * 2.5, 1.0)))
                x, z = float(det.x), float(det.z)
                last_xy = np.array([x, z], dtype=float)
                detected_history.append((frame, x, z))
                # Keep history bounded.
                max_hist = int(cfg.get("local_parabola_window_points", 7)) + 3
                if len(detected_history) > max_hist:
                    detected_history = detected_history[-max_hist:]
                area = det.area
                intensity = det.mean_intensity
            else:
                missed += 1
                gate = min(max_gate, gate * float(cfg.get("gate_growth_factor", 1.10)))
                detected = False
                conf = max(0.0, 0.42 - 0.12 * missed)

                # Prefer local parabola prediction for missing-point compensation once enough data exist.
                if parabola_pred is not None and bool(cfg.get("use_parabola_for_missing_prediction", True)):
                    x, z = float(parabola_pred[0]), float(parabola_pred[1])
                    parabola_dist = 0.0
                else:
                    x, z = float(pred_xy[0]), float(pred_xy[1])
                    parabola_dist = None

                area = None
                intensity = None

                if not allow_predicted_points:
                    break
                if not _inside_mask(x, z, roi_mask):
                    break
                if missed > max_missed:
                    break
                if bool(cfg.get("predicted_points_update_motion", False)):
                    last_xy = np.array([x, z], dtype=float)

            vx, vz = kf.velocity
            ax, az = kf.acceleration
            points.append(TrackPoint(frame, float(x), float(z), float(vx) * direction, float(vz) * direction, float(ax), float(az), detected, conf, area, intensity, missed, parabola_dist))
        frame += direction
    return points


def track_bidirectional(
    dets_by_frame: Dict[int, List[Detection]],
    seed_frame: int,
    seed_xy: Tuple[float, float],
    frame_count: int,
    cfg: dict,
    trajectory_id: int | None = None,
    roi_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if not _inside_mask(seed_xy[0], seed_xy[1], roi_mask):
        raise RuntimeError("种子点位于 ROI 外，已拒绝追踪。")

    seed_gate = float(cfg.get("seed_snap_gate_px", cfg.get("initial_gate_px", 25)))
    seed_det = _nearest_detection(dets_by_frame.get(seed_frame, []), seed_xy, max_dist=seed_gate, roi_mask=roi_mask)
    if seed_det is None:
        if bool(cfg.get("allow_seed_without_detection", False)):
            seed_det = Detection(seed_frame, float(seed_xy[0]), float(seed_xy[1]), 0.0, 0.0, 1.0, 1.0, 0.3)
        else:
            raise RuntimeError("种子点附近没有检测到颗粒，请调整阈值、ROI 或重新选择种子点。")

    backward = track_one_direction(dets_by_frame, seed_frame, -1, -1, seed_det, cfg, roi_mask)
    forward = track_one_direction(dets_by_frame, seed_frame, frame_count, 1, seed_det, cfg, roi_mask)

    pts = list(reversed(backward))[:-1] + forward
    rows = []
    for p in pts:
        if not _inside_mask(p.x, p.z, roi_mask):
            continue
        row = {
            "frame": p.frame,
            "x_px": p.x,
            "z_px": p.z,
            "vx_px_per_frame": p.vx,
            "vz_px_per_frame": p.vz,
            "ax_px_per_frame2": p.ax,
            "az_px_per_frame2": p.az,
            "detected": p.detected,
            "predicted": not p.detected,
            "confidence": p.confidence,
            "area_px": p.area,
            "mean_dark_intensity": p.mean_intensity,
            "missed_count": p.missed_count,
            "local_parabola_dist_px": p.parabola_dist_px,
        }
        if trajectory_id is not None:
            row = {"trajectory_id": int(trajectory_id), **row}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def track_multiple_seeds(
    dets_by_frame: Dict[int, List[Detection]],
    seed_frame: int,
    seed_points: List[Tuple[float, float]],
    frame_count: int,
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Track multiple manually selected seeds independently.

    V1.1.2 uses Kalman prediction + local parabolic consistency. It does not use
    a hard turning-angle/reverse-motion rule because the apex of a true parabola
    naturally creates a large apparent turning angle.
    """
    tracks: List[pd.DataFrame] = []
    status_rows = []
    min_len = int(cfg.get("min_track_points_to_save", 3))

    for idx, seed_xy in enumerate(seed_points, start=1):
        try:
            df = track_bidirectional(dets_by_frame, seed_frame, seed_xy, frame_count, cfg, trajectory_id=idx, roi_mask=roi_mask)
            if len(df) < min_len:
                raise RuntimeError(f"轨迹点数过少，仅 {len(df)} 点；已丢弃，避免噪声轨迹。")
            tracks.append(df)
            status_rows.append({
                "trajectory_id": idx,
                "seed_x_px": float(seed_xy[0]),
                "seed_z_px": float(seed_xy[1]),
                "status": "tracked_raw",
                "points": int(len(df)),
                "detected_points": int(df["detected"].sum()) if len(df) else 0,
                "predicted_points": int((~df["detected"].astype(bool)).sum()) if len(df) else 0,
                "mean_confidence": float(df["confidence"].mean()) if len(df) else 0.0,
                "message": "raw track before global parabolic validation",
            })
            print(f"  轨迹 {idx:03d}: 原始追踪完成，点数={len(df)}")
        except Exception as e:
            status_rows.append({
                "trajectory_id": idx,
                "seed_x_px": float(seed_xy[0]),
                "seed_z_px": float(seed_xy[1]),
                "status": "failed",
                "points": 0,
                "detected_points": 0,
                "predicted_points": 0,
                "mean_confidence": 0.0,
                "message": str(e),
            })
            print(f"  轨迹 {idx:03d}: 失败：{e}")
    # Avoid pandas FutureWarning caused by concatenating empty/all-NA frames.
    tracks = [df for df in tracks if df is not None and len(df) > 0]
    if tracks:
        all_df = pd.concat(tracks, ignore_index=True)
        all_df = all_df.sort_values(["trajectory_id", "frame"]).reset_index(drop=True)
    else:
        all_df = pd.DataFrame()
    return all_df, pd.DataFrame(status_rows)



def track_multiple_seed_records(
    dets_by_frame: Dict[int, List[Detection]],
    seed_records: Sequence[Any],
    frame_count: int,
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Track seeds that may come from different seed frames.

    Each seed record is expected to have: seed_id, seed_frame, x, z, source.
    This is the V2 entry point for auto seeds sampled from multiple time frames.
    """
    tracks: List[pd.DataFrame] = []
    status_rows = []
    min_len = int(cfg.get("min_track_points_to_save", 3))

    for j, seed in enumerate(seed_records, start=1):
        seed_id = int(getattr(seed, "seed_id", j))
        seed_frame = int(getattr(seed, "seed_frame"))
        seed_xy = (float(getattr(seed, "x")), float(getattr(seed, "z")))
        seed_source = str(getattr(seed, "source", "unknown"))
        try:
            df = track_bidirectional(
                dets_by_frame,
                seed_frame,
                seed_xy,
                frame_count,
                cfg,
                trajectory_id=seed_id,
                roi_mask=roi_mask,
            )
            if len(df) < min_len:
                raise RuntimeError(f"轨迹点数过少，仅 {len(df)} 点；已丢弃，避免噪声轨迹。")
            df["seed_id"] = seed_id
            df["seed_frame"] = seed_frame
            df["seed_source"] = seed_source
            tracks.append(df)
            status_rows.append({
                "trajectory_id": seed_id,
                "seed_id": seed_id,
                "seed_frame": seed_frame,
                "seed_x_px": seed_xy[0],
                "seed_z_px": seed_xy[1],
                "seed_source": seed_source,
                "status": "tracked_raw",
                "points": int(len(df)),
                "detected_points": int(df["detected"].sum()) if len(df) else 0,
                "predicted_points": int((~df["detected"].astype(bool)).sum()) if len(df) else 0,
                "mean_confidence": float(df["confidence"].mean()) if len(df) else 0.0,
                "message": "raw track before global parabolic validation",
            })
            print(f"  轨迹 {seed_id:03d}: seed_frame={seed_frame}, 原始追踪完成，点数={len(df)}")
        except Exception as e:
            status_rows.append({
                "trajectory_id": seed_id,
                "seed_id": seed_id,
                "seed_frame": seed_frame,
                "seed_x_px": seed_xy[0],
                "seed_z_px": seed_xy[1],
                "seed_source": seed_source,
                "status": "failed",
                "points": 0,
                "detected_points": 0,
                "predicted_points": 0,
                "mean_confidence": 0.0,
                "message": str(e),
            })
            print(f"  轨迹 {seed_id:03d}: seed_frame={seed_frame}, 失败：{e}")

    tracks = [df for df in tracks if df is not None and len(df) > 0]
    if tracks:
        all_df = pd.concat(tracks, ignore_index=True)
        all_df = all_df.sort_values(["trajectory_id", "frame"]).reset_index(drop=True)
    else:
        all_df = pd.DataFrame()
    return all_df, pd.DataFrame(status_rows)





def prune_tracks_by_parabola_residual(
    tracks_df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove isolated trajectory points that are inconsistent with the global parabolic trend.

    This is a conservative pre-validation denoising step for V2.1. It does not invent new
    points; it only removes points whose distance to a time-parametric quadratic fit is
    too large. It mainly suppresses cases where a background noise spot or nearby particle
    was accidentally linked into an otherwise valid trajectory.
    """
    if tracks_df is None or len(tracks_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    if not bool(cfg.get("enable_track_outlier_pruning", True)):
        rows = [{"trajectory_id": int(t), "pruning_enabled": False, "removed_points": 0}
                for t in sorted(tracks_df["trajectory_id"].unique())]
        return tracks_df.copy(), pd.DataFrame(rows)

    min_points = int(cfg.get("outlier_prune_min_points", 7))
    max_res = float(cfg.get("outlier_prune_max_residual_px", 18.0))
    rmse_factor = float(cfg.get("outlier_prune_rmse_factor", 2.5))
    max_iter = int(cfg.get("outlier_prune_iterations", 2))
    preserve_predicted = bool(cfg.get("outlier_prune_preserve_predicted", False))

    kept_groups = []
    summary = []
    for tid, group in tracks_df.groupby("trajectory_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        if len(group) < min_points:
            kept_groups.append(group)
            summary.append({
                "trajectory_id": int(tid),
                "original_points": int(len(group)),
                "kept_points": int(len(group)),
                "removed_points": 0,
                "reason": "too_short_for_pruning",
            })
            continue

        keep_mask = np.ones(len(group), dtype=bool)
        reason = "ok"
        for _ in range(max_iter):
            work = group[keep_mask].copy()
            # Fit primarily to detected points; predicted points are less reliable.
            fit_src = work[work["detected"].astype(bool)] if "detected" in work.columns else work
            if len(fit_src) < min_points or len(fit_src["frame"].unique()) < 3:
                reason = "too_few_detected_after_iteration"
                break
            t = fit_src["frame"].to_numpy(dtype=float)
            tau = t - float(t.min())
            x = fit_src["x_px"].to_numpy(dtype=float)
            z = fit_src["z_px"].to_numpy(dtype=float)
            try:
                px = np.polyfit(tau, x, 2)
                pz = np.polyfit(tau, z, 2)
            except Exception:
                reason = "polyfit_failed"
                break

            all_t = group["frame"].to_numpy(dtype=float)
            all_tau = all_t - float(t.min())
            xf = np.polyval(px, all_tau)
            zf = np.polyval(pz, all_tau)
            residual = np.sqrt((group["x_px"].to_numpy(dtype=float) - xf) ** 2 + (group["z_px"].to_numpy(dtype=float) - zf) ** 2)
            current_res = residual[keep_mask]
            rmse = float(np.sqrt(np.mean(current_res ** 2))) if len(current_res) else 0.0
            dyn_threshold = max_res if rmse <= 0 else max(max_res, rmse_factor * rmse)

            new_keep = residual <= dyn_threshold
            if preserve_predicted and "detected" in group.columns:
                # Optional: keep predicted points unless they are extremely far away.
                predicted = ~group["detected"].astype(bool).to_numpy()
                new_keep = new_keep | (predicted & (residual <= dyn_threshold * 1.5))
            # Never allow pruning below minimum points.
            if int(new_keep.sum()) < min_points:
                reason = "would_be_too_short"
                break
            if np.array_equal(new_keep, keep_mask):
                break
            keep_mask = new_keep

        pruned = group[keep_mask].copy().reset_index(drop=True)
        if len(pruned) > 0:
            kept_groups.append(pruned)
        summary.append({
            "trajectory_id": int(tid),
            "original_points": int(len(group)),
            "kept_points": int(len(pruned)),
            "removed_points": int(len(group) - len(pruned)),
            "reason": reason,
        })

    out = pd.concat(kept_groups, ignore_index=True) if kept_groups else pd.DataFrame(columns=tracks_df.columns)
    return out, pd.DataFrame(summary)

def _track_quality_score(group: pd.DataFrame) -> float:
    if group is None or len(group) == 0:
        return -1e9
    detected = float(group["detected"].astype(bool).sum()) if "detected" in group.columns else float(len(group))
    predicted = float((~group["detected"].astype(bool)).sum()) if "detected" in group.columns else 0.0
    conf = float(group["confidence"].mean()) if "confidence" in group.columns and len(group) else 0.0
    duration = float(group["frame"].max() - group["frame"].min() + 1) if "frame" in group.columns else float(len(group))
    return detected * 3.0 + duration * 0.5 + conf * 20.0 - predicted * 1.2


def _pair_track_overlap_distance(g1: pd.DataFrame, g2: pd.DataFrame) -> tuple[int, float]:
    a = g1[["frame", "x_px", "z_px"]].copy()
    b = g2[["frame", "x_px", "z_px"]].copy()
    m = a.merge(b, on="frame", suffixes=("_a", "_b"))
    if len(m) == 0:
        return 0, float("inf")
    dist = np.sqrt((m["x_px_a"] - m["x_px_b"]) ** 2 + (m["z_px_a"] - m["z_px_b"]) ** 2)
    return int(len(m)), float(dist.mean())


def remove_duplicate_tracks(
    tracks_df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate tracks generated by auto seeds from multiple frames.

    Two tracks are duplicates when they share enough frames and their same-frame
    positions are closer than duplicate_track_distance_px. The better-quality one
    is retained.
    """
    if tracks_df is None or len(tracks_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    if not bool(cfg.get("enable_duplicate_track_filter", True)):
        rows = [{"trajectory_id": int(t), "duplicate_status": "kept", "reason": "duplicate_filter_disabled"}
                for t in sorted(tracks_df["trajectory_id"].unique())]
        return tracks_df.copy(), pd.DataFrame(rows)

    min_overlap = int(cfg.get("duplicate_min_overlap_frames", 5))
    max_dist = float(cfg.get("duplicate_track_distance_px", 8.0))
    min_overlap_ratio = float(cfg.get("duplicate_min_overlap_ratio", 0.35))

    groups = {int(t): g.sort_values("frame").reset_index(drop=True) for t, g in tracks_df.groupby("trajectory_id")}
    tids = sorted(groups)
    keep = set(tids)
    rows = []

    for i, ti in enumerate(tids):
        if ti not in keep:
            continue
        for tj in tids[i + 1:]:
            if tj not in keep:
                continue
            g1, g2 = groups[ti], groups[tj]
            overlap, mean_dist = _pair_track_overlap_distance(g1, g2)
            shorter = max(1, min(len(g1), len(g2)))
            overlap_ratio = overlap / shorter
            is_dup = overlap >= min_overlap and overlap_ratio >= min_overlap_ratio and mean_dist <= max_dist
            if not is_dup:
                continue
            q1, q2 = _track_quality_score(g1), _track_quality_score(g2)
            if q1 >= q2:
                loser, winner = tj, ti
            else:
                loser, winner = ti, tj
            keep.discard(loser)
            rows.append({
                "trajectory_id": int(loser),
                "duplicate_status": "removed",
                "kept_as": int(winner),
                "overlap_frames": int(overlap),
                "overlap_ratio": float(overlap_ratio),
                "mean_overlap_distance_px": float(mean_dist),
                "quality_loser": float(_track_quality_score(groups[loser])),
                "quality_winner": float(_track_quality_score(groups[winner])),
            })
            if loser == ti:
                break

    for tid in tids:
        if tid in keep:
            rows.append({"trajectory_id": int(tid), "duplicate_status": "kept", "kept_as": int(tid)})
    out = tracks_df[tracks_df["trajectory_id"].isin(keep)].copy().reset_index(drop=True)
    return out, pd.DataFrame(rows).sort_values(["duplicate_status", "trajectory_id"]).reset_index(drop=True)

def estimate_bottom_boundary_z(roi_mask: Optional[np.ndarray], cfg: dict) -> float:
    """Estimate the lower wall/boundary in pixel z coordinates.

    For the current images z usually increases downward, so the lower boundary is max z
    in the selected ROI. If wall_z_px is given as a number, it overrides auto estimation.
    """
    wall = cfg.get("wall_z_px", "auto")
    if wall is not None and str(wall).lower() != "auto":
        return float(wall)
    if roi_mask is not None:
        ys = np.where(roi_mask > 0)[0]
        if len(ys):
            return float(np.max(ys))
    return float(cfg.get("image_bottom_z_px", 0.0))


def validate_global_parabolic_flight(
    track_df: pd.DataFrame,
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> tuple[bool, dict]:
    """Validate complete flight: bottom -> rise -> apex -> fall -> bottom.

    This is a post-tracking filter. It rejects tracks that are only rising, only falling,
    or cannot be explained by a time-parametric parabola with an internal apex.
    """
    if track_df is None or len(track_df) == 0:
        return False, {"reason": "empty_track"}

    min_detected = int(cfg.get("global_min_detected_points", 6))
    valid = track_df[track_df["detected"].astype(bool)].copy()
    if len(valid) < min_detected:
        return False, {"reason": "too_few_detected_points", "detected_points": int(len(valid))}

    t = valid["frame"].to_numpy(dtype=float)
    x = valid["x_px"].to_numpy(dtype=float)
    z = valid["z_px"].to_numpy(dtype=float)
    t0 = float(t.min())
    tau = t - t0
    if len(np.unique(tau)) < 3:
        return False, {"reason": "too_few_unique_frames"}

    try:
        px = np.polyfit(tau, x, 2)
        pz = np.polyfit(tau, z, 2)
    except Exception as e:
        return False, {"reason": "polyfit_failed", "error": str(e)}

    xfit = np.polyval(px, tau)
    zfit = np.polyval(pz, tau)
    rmse = float(np.sqrt(np.mean((x - xfit) ** 2 + (z - zfit) ** 2)))
    max_rmse = float(cfg.get("global_max_parabola_rmse_px", 18.0))
    if rmse > max_rmse:
        return False, {"reason": "global_parabola_rmse_too_large", "rmse_px": rmse, "max_allowed_px": max_rmse}

    z_down = str(cfg.get("z_axis_direction", "down")).lower() == "down"
    a = float(pz[0])
    b = float(pz[1])
    duration = float(tau.max() - tau.min())
    if duration <= 0:
        return False, {"reason": "zero_duration"}

    # For image coordinates with z downward, a positive quadratic means upward launch then downward fall.
    # For z upward, the sign is reversed.
    min_abs_a = float(cfg.get("global_min_abs_z_curvature", 1e-6))
    if z_down:
        if a <= min_abs_a:
            return False, {"reason": "wrong_curvature_for_downward_z", "z_quad_a": a}
    else:
        if a >= -min_abs_a:
            return False, {"reason": "wrong_curvature_for_upward_z", "z_quad_a": a}

    tau_apex = -b / (2.0 * a) if abs(a) > 1e-12 else float("nan")
    margin = float(cfg.get("global_apex_margin_frames", 1.0))
    if not np.isfinite(tau_apex) or tau_apex <= tau.min() + margin or tau_apex >= tau.max() - margin:
        return False, {"reason": "apex_not_inside_track", "tau_apex": float(tau_apex), "duration_frames": duration}

    z_apex = float(np.polyval(pz, tau_apex))
    z_start = float(np.polyval(pz, tau.min()))
    z_end = float(np.polyval(pz, tau.max()))
    if z_down:
        flight_height = min(z_start, z_end) - z_apex
    else:
        flight_height = z_apex - max(z_start, z_end)
    min_height = float(cfg.get("global_min_flight_height_px", 6.0))
    if flight_height < min_height:
        return False, {"reason": "flight_height_too_small", "flight_height_px": float(flight_height)}

    # The fitted vertical velocity should change sign: rising before apex, falling after apex.
    dz_start = float(2.0 * a * tau.min() + b)
    dz_end = float(2.0 * a * tau.max() + b)
    if z_down:
        if not (dz_start < 0 and dz_end > 0):
            return False, {"reason": "not_rise_then_fall", "dz_start": dz_start, "dz_end": dz_end}
    else:
        if not (dz_start > 0 and dz_end < 0):
            return False, {"reason": "not_rise_then_fall", "dz_start": dz_start, "dz_end": dz_end}

    bottom_z = estimate_bottom_boundary_z(roi_mask, cfg)
    boundary_tol = float(cfg.get("bottom_boundary_tolerance_px", 25.0))
    # Use fitted roots with the lower boundary. A complete flight should have two crossings
    # close to or outside the observed time range. This handles hidden start/end near the wall.
    pz_root = pz.copy()
    pz_root[-1] -= bottom_z
    roots = np.roots(pz_root)
    real_roots = sorted([float(r.real) for r in roots if abs(r.imag) < 1e-5])
    root_ok = False
    if len(real_roots) >= 2:
        r1, r2 = real_roots[0], real_roots[-1]
        start_ok = r1 <= tau.min() + float(cfg.get("boundary_root_frame_margin", 8.0))
        end_ok = r2 >= tau.max() - float(cfg.get("boundary_root_frame_margin", 8.0))
        root_ok = bool(start_ok and end_ok)

    endpoint_near_bottom = abs(z_start - bottom_z) <= boundary_tol and abs(z_end - bottom_z) <= boundary_tol
    if bool(cfg.get("require_bottom_boundary_consistency", True)) and not (root_ok or endpoint_near_bottom):
        return False, {
            "reason": "not_consistent_with_lower_boundary_start_end",
            "bottom_z_px": float(bottom_z),
            "z_start_fit_px": z_start,
            "z_end_fit_px": z_end,
            "boundary_roots_tau": real_roots,
        }

    info = {
        "reason": "ok",
        "global_parabola_rmse_px": rmse,
        "z_quad_a": a,
        "tau_apex": float(tau_apex),
        "frame_apex": float(t0 + tau_apex),
        "z_apex_fit_px": z_apex,
        "z_start_fit_px": z_start,
        "z_end_fit_px": z_end,
        "flight_height_px": float(flight_height),
        "bottom_z_px": float(bottom_z),
        "boundary_roots_tau": real_roots,
        "poly_x_desc": px.tolist(),
        "poly_z_desc": pz.tolist(),
    }
    return True, info


def filter_tracks_by_global_flight_model(
    tracks_df: pd.DataFrame,
    cfg: dict,
    roi_mask: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split raw tracks into accepted and rejected using global parabolic flight validation."""
    if tracks_df is None or len(tracks_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not bool(cfg.get("enable_global_flight_validation", True)):
        summary = []
        for tid, g in tracks_df.groupby("trajectory_id"):
            summary.append({"trajectory_id": int(tid), "flight_valid": True, "reason": "validation_disabled"})
        return tracks_df.copy(), pd.DataFrame(), pd.DataFrame(summary)

    accepted = []
    rejected = []
    rows = []
    for tid, group in tracks_df.groupby("trajectory_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        ok, info = validate_global_parabolic_flight(group, cfg, roi_mask)
        row = {"trajectory_id": int(tid), "flight_valid": bool(ok), **info}
        rows.append(row)
        if ok:
            accepted.append(group)
        else:
            rejected.append(group.assign(reject_reason=info.get("reason", "unknown")))
    accepted_df = pd.concat(accepted, ignore_index=True) if accepted else pd.DataFrame(columns=tracks_df.columns)
    rejected_df = pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame()
    return accepted_df, rejected_df, pd.DataFrame(rows)
