from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd


def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return float(default)
        f = float(v)
        return f if np.isfinite(f) else float(default)
    except Exception:
        return float(default)


def _resolve_wall_z_px(track_df: pd.DataFrame, cfg: dict) -> float:
    wall_z_raw = cfg.get("wall_z_px", "auto")
    if isinstance(wall_z_raw, str) and wall_z_raw.lower() == "auto":
        roi_points = cfg.get("_runtime_roi_points", None)
        if roi_points:
            return float(max(float(p[1]) for p in roi_points))
        if "z_px" in track_df.columns and len(track_df) > 0:
            return float(track_df["z_px"].max())
        return 0.0
    return float(wall_z_raw)


def _detected_or_all(group: pd.DataFrame, min_points: int = 2) -> pd.DataFrame:
    if "detected" in group.columns:
        det = group[group["detected"].astype(bool)].copy()
        if len(det) >= min_points:
            return det
    return group.copy()


def _first_last_valid(group: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    g = group.sort_values("frame").reset_index(drop=True)
    src = _detected_or_all(g, min_points=2).sort_values("frame").reset_index(drop=True)
    return src.iloc[0], src.iloc[-1], src


def _finite_v0_from_first_two(src: pd.DataFrame, pixel_size_mm: float, fps: float) -> dict:
    if len(src) < 2:
        return {"v0_x_mm_s": np.nan, "v0_z_up_mm_s": np.nan, "v0_mm_s": np.nan,
                "theta0_deg_signed": np.nan, "theta0_deg_abs": np.nan,
                "v0_dt_s": np.nan, "v0_method": "failed_less_than_2_points"}
    p0 = src.iloc[0]
    p1 = src.iloc[1]
    dt_frame = float(p1["frame"] - p0["frame"])
    if dt_frame <= 0:
        return {"v0_x_mm_s": np.nan, "v0_z_up_mm_s": np.nan, "v0_mm_s": np.nan,
                "theta0_deg_signed": np.nan, "theta0_deg_abs": np.nan,
                "v0_dt_s": np.nan, "v0_method": "failed_nonpositive_dt"}
    dt = dt_frame / fps
    dx_mm = (float(p1["x_px"]) - float(p0["x_px"])) * pixel_size_mm
    # Image z grows downward. Physical upward displacement is -dz.
    dz_up_mm = -(float(p1["z_px"]) - float(p0["z_px"])) * pixel_size_mm
    vx = dx_mm / dt
    vz_up = dz_up_mm / dt
    v = math.hypot(vx, vz_up)
    theta_signed = math.degrees(math.atan2(vz_up, vx)) if v > 0 else np.nan
    theta_abs = math.degrees(math.atan2(vz_up, abs(vx))) if v > 0 else np.nan
    return {"v0_x_mm_s": vx, "v0_z_up_mm_s": vz_up, "v0_mm_s": v,
            "theta0_deg_signed": theta_signed, "theta0_deg_abs": theta_abs,
            "v0_dt_s": dt, "v0_method": "first_two_detected_points"}


def _a0_from_first_three(src: pd.DataFrame, pixel_size_mm: float, fps: float) -> dict:
    if len(src) < 3:
        return {"a0_x_mm_s2": np.nan, "a0_z_up_mm_s2": np.nan, "a0_mm_s2": np.nan,
                "a0_method": "failed_less_than_3_points"}
    p0, p1, p2 = src.iloc[0], src.iloc[1], src.iloc[2]
    f0, f1, f2 = float(p0["frame"]), float(p1["frame"]), float(p2["frame"])
    if f2 <= f0 or f1 <= f0 or f2 <= f1:
        return {"a0_x_mm_s2": np.nan, "a0_z_up_mm_s2": np.nan, "a0_mm_s2": np.nan,
                "a0_method": "failed_nonpositive_dt"}
    t0, t1, t2 = f0 / fps, f1 / fps, f2 / fps
    x = np.array([float(p0["x_px"]), float(p1["x_px"]), float(p2["x_px"])]) * pixel_size_mm
    zup = -np.array([float(p0["z_px"]), float(p1["z_px"]), float(p2["z_px"])]) * pixel_size_mm
    t = np.array([t0, t1, t2], dtype=float)
    try:
        cx = np.polyfit(t - t0, x, 2)
        cz = np.polyfit(t - t0, zup, 2)
        ax = 2.0 * float(cx[0])
        az = 2.0 * float(cz[0])
        return {"a0_x_mm_s2": ax, "a0_z_up_mm_s2": az, "a0_mm_s2": math.hypot(ax, az),
                "a0_method": "quadratic_first_three_detected_points"}
    except Exception:
        return {"a0_x_mm_s2": np.nan, "a0_z_up_mm_s2": np.nan, "a0_mm_s2": np.nan,
                "a0_method": "failed_polyfit"}


def _energy_v0_star(height_mm: float, theta_abs_deg: float, cfg: dict) -> dict:
    g = float(cfg.get("stats_gravity_m_s2", 9.80665))
    h_m = max(0.0, float(height_mm)) / 1000.0
    v_z_m_s = math.sqrt(2.0 * g * h_m) if h_m > 0 else 0.0
    out = {
        "v0_star_z_mm_s": v_z_m_s * 1000.0,
        "v0_star_method": "sqrt(2*g*jump_height), vertical component from potential energy",
    }
    theta = _safe_float(theta_abs_deg)
    if np.isfinite(theta) and abs(math.sin(math.radians(theta))) > 1e-6:
        out["v0_star_total_mm_s_from_theta"] = v_z_m_s * 1000.0 / abs(math.sin(math.radians(theta)))
    else:
        out["v0_star_total_mm_s_from_theta"] = np.nan

    # Optional particle mass/energy estimate. Not needed for V0*, but useful for EDS statistics.
    diameter_um = cfg.get("particle_diameter_um", None)
    density = cfg.get("particle_density_kg_m3", None)
    if diameter_um is not None and density is not None:
        d_m = float(diameter_um) * 1e-6
        rho = float(density)
        mass = rho * (math.pi / 6.0) * d_m ** 3
        out["particle_mass_kg"] = mass
        out["potential_energy_J"] = mass * g * h_m
        out["kinetic_energy_star_vertical_J"] = 0.5 * mass * v_z_m_s ** 2
    return out


def make_eds_track_summary(tracks_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if tracks_df is None or len(tracks_df) == 0:
        return pd.DataFrame()
    fps = float(cfg.get("fps", 1000.0))
    px = float(cfg.get("pixel_size_mm", 0.01))
    wall_z_px = _resolve_wall_z_px(tracks_df, cfg)
    rows = []
    for tid, group in tracks_df.groupby("trajectory_id"):
        g = group.sort_values("frame").reset_index(drop=True)
        start, end, src = _first_last_valid(g)
        apex = g.loc[g["z_px"].idxmin()]
        start_x_px = float(start["x_px"])
        end_x_px = float(end["x_px"])
        start_z_px = float(start["z_px"])
        end_z_px = float(end["z_px"])
        apex_z_px = float(apex["z_px"])
        start_frame = int(start["frame"])
        end_frame = int(end["frame"])
        duration_s = max(0.0, (end_frame - start_frame) / fps)
        jump_height_mm = max(0.0, (wall_z_px - apex_z_px) * px)
        launch_height_offset_mm = (wall_z_px - start_z_px) * px
        landing_height_offset_mm = (wall_z_px - end_z_px) * px
        dx_mm = (end_x_px - start_x_px) * px
        horizontal_displacement_abs_mm = abs(dx_mm)
        v0 = _finite_v0_from_first_two(src, px, fps)
        a0 = _a0_from_first_three(src, px, fps)
        vstar = _energy_v0_star(jump_height_mm, v0.get("theta0_deg_abs", np.nan), cfg)
        detected_points = int(g["detected"].astype(bool).sum()) if "detected" in g.columns else int(len(g))
        predicted_points = int((~g["detected"].astype(bool)).sum()) if "detected" in g.columns else 0
        rows.append({
            "trajectory_id": int(tid),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": start_frame / fps,
            "end_time_s": end_frame / fps,
            "start_time_ms": start_frame / fps * 1000.0,
            "end_time_ms": end_frame / fps * 1000.0,
            "transfer_time_s": duration_s,
            "transfer_time_ms": duration_s * 1000.0,
            "launch_x_px": start_x_px,
            "launch_z_px": start_z_px,
            "landing_x_px": end_x_px,
            "landing_z_px": end_z_px,
            "apex_x_px": float(apex["x_px"]),
            "apex_z_px": apex_z_px,
            "launch_x_mm": start_x_px * px,
            "launch_z_mm": start_z_px * px,
            "landing_x_mm": end_x_px * px,
            "landing_z_mm": end_z_px * px,
            "apex_x_mm": float(apex["x_px"]) * px,
            "apex_z_mm": apex_z_px * px,
            "jump_height_mm": jump_height_mm,
            "horizontal_displacement_mm_signed": dx_mm,
            "horizontal_displacement_mm_abs": horizontal_displacement_abs_mm,
            "launch_height_offset_from_wall_mm": launch_height_offset_mm,
            "landing_height_offset_from_wall_mm": landing_height_offset_mm,
            "detected_points": detected_points,
            "predicted_points": predicted_points,
            "predicted_ratio": predicted_points / max(len(g), 1),
            "mean_confidence": float(g["confidence"].mean()) if "confidence" in g.columns else np.nan,
            **v0,
            **vstar,
            **a0,
        })
    return pd.DataFrame(rows).sort_values("trajectory_id").reset_index(drop=True)


def make_motion_point_table(tracks_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-point table for acceleration vs x and later statistical analysis.

    Acceleration is computed from accepted trajectory coordinates using finite differences
    after sorting by time. This table is intentionally independent from Kalman internal
    acceleration, because the latter is a state estimate and can be less transparent.
    """
    if tracks_df is None or len(tracks_df) == 0:
        return pd.DataFrame()
    fps = float(cfg.get("fps", 1000.0))
    px = float(cfg.get("pixel_size_mm", 0.01))
    rows = []
    for tid, group in tracks_df.groupby("trajectory_id"):
        g = group.sort_values("frame").reset_index(drop=True)
        if len(g) < 3:
            continue
        t = g["frame"].to_numpy(dtype=float) / fps
        x = g["x_px"].to_numpy(dtype=float) * px
        zup = -g["z_px"].to_numpy(dtype=float) * px
        # np.gradient handles nonuniform frame gaps if t is supplied.
        try:
            vx = np.gradient(x, t)
            vz = np.gradient(zup, t)
            ax = np.gradient(vx, t)
            az = np.gradient(vz, t)
        except Exception:
            continue
        v = np.sqrt(vx ** 2 + vz ** 2)
        a = np.sqrt(ax ** 2 + az ** 2)
        for i, r in g.iterrows():
            rows.append({
                "trajectory_id": int(tid),
                "frame": int(r["frame"]),
                "time_s": float(t[i]),
                "time_ms": float(t[i] * 1000.0),
                "track_time_s": float(t[i] - t[0]),
                "track_time_ms": float((t[i] - t[0]) * 1000.0),
                "x_mm": float(x[i]),
                "z_up_mm": float(zup[i]),
                "vx_mm_s_fd": float(vx[i]),
                "vz_up_mm_s_fd": float(vz[i]),
                "v_mm_s_fd": float(v[i]),
                "ax_mm_s2_fd": float(ax[i]),
                "az_up_mm_s2_fd": float(az[i]),
                "a_mm_s2_fd": float(a[i]),
                "detected": bool(r["detected"]) if "detected" in g.columns else True,
                "confidence": float(r["confidence"]) if "confidence" in g.columns else np.nan,
            })
    return pd.DataFrame(rows)


def _bin_by_x(df: pd.DataFrame, x_col: str, cfg: dict, prefix: str = "x") -> pd.DataFrame:
    if df is None or len(df) == 0 or x_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    width = cfg.get("stats_x_bin_width_mm", "auto")
    if isinstance(width, str) and width.lower() == "auto":
        bins_n = int(cfg.get("stats_x_bin_count", 20))
        xmin = float(work[x_col].min())
        xmax = float(work[x_col].max())
        if xmax <= xmin:
            xmax = xmin + 1e-6
        bins = np.linspace(xmin, xmax, bins_n + 1)
    else:
        w = float(width)
        xmin = math.floor(float(work[x_col].min()) / w) * w
        xmax = math.ceil(float(work[x_col].max()) / w) * w
        bins = np.arange(xmin, xmax + w, w)
        if len(bins) < 2:
            bins = np.array([xmin, xmin + w])
    work["x_bin_id"] = pd.cut(work[x_col], bins=bins, include_lowest=True, labels=False)
    rows = []
    for bid, g in work.groupby("x_bin_id"):
        if pd.isna(bid):
            continue
        bid = int(bid)
        left = float(bins[bid])
        right = float(bins[bid + 1])
        row = {"x_bin_id": bid, "x_left_mm": left, "x_right_mm": right, "x_center_mm": 0.5 * (left + right), "count": int(len(g))}
        for col in ["ax_mm_s2_fd", "az_up_mm_s2_fd", "a_mm_s2_fd", "theta0_deg_signed", "theta0_deg_abs", "v0_mm_s", "jump_height_mm", "horizontal_displacement_mm_abs", "transfer_time_ms"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
                row[f"{col}_median"] = float(g[col].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("x_bin_id").reset_index(drop=True) if rows else pd.DataFrame()


def write_eds_statistics(tracks_df: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    """Write compact EDS-oriented statistics for one image sequence.

    Main outputs:
      - EDS_track_summary.csv: one row per particle transfer event.
      - EDS_motion_points.csv: per-point kinematics with finite-difference acceleration.
      - EDS_acceleration_by_x.csv: binned acceleration distribution along x.
      - EDS_launch_angle_by_x.csv: binned launch-angle distribution along x.
      - EDS_sequence_summary.csv: one-row summary for V3 batch aggregation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = make_eds_track_summary(tracks_df, cfg)
    points = make_motion_point_table(tracks_df, cfg)
    acc_by_x = _bin_by_x(points, "x_mm", cfg) if len(points) else pd.DataFrame()
    angle_by_x = _bin_by_x(summary, "launch_x_mm", cfg) if len(summary) else pd.DataFrame()

    summary.to_csv(out_dir / "EDS_track_summary.csv", index=False, encoding="utf-8-sig")
    points.to_csv(out_dir / "EDS_motion_points.csv", index=False, encoding="utf-8-sig")
    acc_by_x.to_csv(out_dir / "EDS_acceleration_by_x.csv", index=False, encoding="utf-8-sig")
    angle_by_x.to_csv(out_dir / "EDS_launch_angle_by_x.csv", index=False, encoding="utf-8-sig")

    seq = make_sequence_summary(summary, points, cfg)
    seq.to_csv(out_dir / "EDS_sequence_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "track_summary": summary,
        "motion_points": points,
        "acceleration_by_x": acc_by_x,
        "launch_angle_by_x": angle_by_x,
        "sequence_summary": seq,
    }


def make_sequence_summary(track_summary: pd.DataFrame, motion_points: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n = int(len(track_summary)) if track_summary is not None else 0
    row = {
        "sequence_id": str(cfg.get("sequence_id", "sequence_001")),
        "input_dir": str(cfg.get("input_dir", "")),
        "particle_count_valid_tracks": n,
        "fps": float(cfg.get("fps", 1000.0)),
        "pixel_size_mm": float(cfg.get("pixel_size_mm", 0.01)),
    }
    if n > 0:
        for col in ["jump_height_mm", "horizontal_displacement_mm_abs", "horizontal_displacement_mm_signed", "transfer_time_ms", "v0_mm_s", "v0_star_z_mm_s", "v0_star_total_mm_s_from_theta", "a0_mm_s2", "theta0_deg_abs"]:
            if col in track_summary.columns:
                row[f"{col}_mean"] = float(track_summary[col].mean())
                row[f"{col}_std"] = float(track_summary[col].std(ddof=1)) if n > 1 else 0.0
                row[f"{col}_median"] = float(track_summary[col].median())
                row[f"{col}_min"] = float(track_summary[col].min())
                row[f"{col}_max"] = float(track_summary[col].max())
    if motion_points is not None and len(motion_points) > 0:
        row["motion_point_count"] = int(len(motion_points))
        for col in ["ax_mm_s2_fd", "az_up_mm_s2_fd", "a_mm_s2_fd"]:
            if col in motion_points.columns:
                row[f"{col}_mean"] = float(motion_points[col].mean())
                row[f"{col}_std"] = float(motion_points[col].std(ddof=1)) if len(motion_points) > 1 else 0.0
    else:
        row["motion_point_count"] = 0
    return pd.DataFrame([row])
