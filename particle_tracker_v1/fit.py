from typing import Tuple, Optional
import numpy as np
import pandas as pd


def add_physical_units(df: pd.DataFrame, fps: float, pixel_size_mm: float) -> pd.DataFrame:
    out = df.copy()
    out["time_s"] = out["frame"] / float(fps)
    out["x_mm"] = out["x_px"] * float(pixel_size_mm)
    out["z_mm"] = out["z_px"] * float(pixel_size_mm)
    out["vx_mm_s"] = out["vx_px_per_frame"] * float(pixel_size_mm) * float(fps)
    out["vz_mm_s"] = out["vz_px_per_frame"] * float(pixel_size_mm) * float(fps)
    out["ax_mm_s2"] = out["ax_px_per_frame2"] * float(pixel_size_mm) * float(fps) ** 2
    out["az_mm_s2"] = out["az_px_per_frame2"] * float(pixel_size_mm) * float(fps) ** 2
    out["v_mm_s"] = np.sqrt(out["vx_mm_s"] ** 2 + out["vz_mm_s"] ** 2)
    return out


def fit_time_polynomial(track_df: pd.DataFrame, degree: int = 2, min_points: int = 6) -> Tuple[pd.DataFrame, dict]:
    valid = track_df[track_df["detected"] == True].copy()
    if len(valid) < min_points:
        valid = track_df.copy()
    if len(valid) < max(degree + 1, 3):
        raise RuntimeError("轨迹点数量不足，无法进行稳定拟合。")

    t = valid["frame"].to_numpy(dtype=float)
    x = valid["x_px"].to_numpy(dtype=float)
    z = valid["z_px"].to_numpy(dtype=float)
    t0 = t.min()
    tau = t - t0

    px = np.polyfit(tau, x, degree)
    pz = np.polyfit(tau, z, degree)

    all_t = track_df["frame"].to_numpy(dtype=float)
    all_tau = all_t - t0
    x_fit = np.polyval(px, all_tau)
    z_fit = np.polyval(pz, all_tau)

    fit_df = track_df[["frame"]].copy()
    fit_df["x_fit_px"] = x_fit
    fit_df["z_fit_px"] = z_fit
    fit_df["fit_residual_px"] = np.sqrt((track_df["x_px"].to_numpy() - x_fit) ** 2 + (track_df["z_px"].to_numpy() - z_fit) ** 2)

    info = {
        "t0_frame": float(t0),
        "degree": degree,
        "poly_x_desc": px.tolist(),
        "poly_z_desc": pz.tolist(),
        "mean_residual_px": float(fit_df["fit_residual_px"].mean()),
        "max_residual_px": float(fit_df["fit_residual_px"].max()),
        "detected_points_used": int(len(valid)),
    }
    return fit_df, info


def extrapolate_z0_crossings(poly_z_desc: list, poly_x_desc: list, t0_frame: float, wall_z_px: float = 0.0) -> dict:
    """Find roots for z(tau)=wall_z_px and evaluate x. tau is in frame units."""
    pz = np.array(poly_z_desc, dtype=float).copy()
    pz[-1] -= float(wall_z_px)
    roots = np.roots(pz)
    real_roots = sorted([float(r.real) for r in roots if abs(r.imag) < 1e-6])
    px = np.array(poly_x_desc, dtype=float)
    crossings = []
    for tau in real_roots:
        crossings.append({
            "frame_float": float(t0_frame + tau),
            "tau_frame": float(tau),
            "x_px": float(np.polyval(px, tau)),
            "z_px": float(wall_z_px),
        })
    return {"z0_crossings": crossings}
