from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _color_for_id(tid: int) -> tuple[int, int, int]:
    colors = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 0, 0), (255, 0, 255), (255, 255, 0), (128, 0, 255),
        (255, 128, 0), (0, 128, 255), (128, 255, 0), (255, 0, 128),
    ]
    return colors[(int(tid) - 1) % len(colors)]


def _draw_text_with_halo(img, text, org, font_scale, color, thickness=1):
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def save_trajectory_overlay(
    base_gray: np.ndarray,
    track_df: pd.DataFrame,
    out_path: Path,
    cfg: dict | None = None,
    fit_df: pd.DataFrame | None = None,
):
    """Save trajectory overlay.

    V2.1 improvements:
      - optional use of fitted/smoothed trajectory curves for cleaner reading;
      - optional small labels, endpoint labels, or no labels;
      - optional no-label image for publication/debugging.
    """
    cfg = cfg or {}
    img = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    if track_df is None or len(track_df) == 0:
        cv2.imwrite(str(out_path), img)
        return

    if "trajectory_id" not in track_df.columns:
        track_df = track_df.copy()
        track_df["trajectory_id"] = 1

    use_fit_curve = bool(cfg.get("overlay_use_fit_curve", True)) and fit_df is not None and len(fit_df) > 0
    draw_points = bool(cfg.get("overlay_draw_points", True))
    draw_raw_polyline = bool(cfg.get("overlay_draw_raw_polyline", not use_fit_curve))
    line_thickness = int(cfg.get("overlay_line_thickness", 2))
    point_radius = int(cfg.get("overlay_point_radius", 2))
    label_mode = str(cfg.get("overlay_label_mode", "end")).lower()  # none/start/end/both
    label_every_n = max(1, int(cfg.get("overlay_label_every_n", 1)))
    label_font_scale = float(cfg.get("overlay_label_font_scale", 0.38))
    label_offset_x = int(cfg.get("overlay_label_offset_x", 8))
    label_offset_y = int(cfg.get("overlay_label_offset_y", -8))

    # Draw smoothed fit first when available.
    if use_fit_curve:
        if "trajectory_id" not in fit_df.columns:
            fit_df = fit_df.copy()
            fit_df["trajectory_id"] = 1
        for tid, group in fit_df.groupby("trajectory_id"):
            color = _color_for_id(int(tid))
            g = group.sort_values("frame")
            if "x_fit_px" in g.columns and "z_fit_px" in g.columns:
                pts = g[["x_fit_px", "z_fit_px"]].dropna().to_numpy(dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=line_thickness)

    # Raw polyline and measured points.
    for tid, group in track_df.groupby("trajectory_id"):
        group = group.sort_values("frame")
        color = _color_for_id(int(tid))
        pts = group[["x_px", "z_px"]].dropna().to_numpy(dtype=np.int32)
        if draw_raw_polyline and len(pts) >= 2:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=max(1, line_thickness - 1))
        if draw_points:
            for _, row in group.iterrows():
                x, z = int(round(row["x_px"])), int(round(row["z_px"]))
                c = color if bool(row.get("detected", True)) else (170, 170, 170)
                cv2.circle(img, (x, z), point_radius, c, -1)

        # Labels: deliberately sparse and small to avoid occlusion.
        if label_mode != "none" and (int(tid) - 1) % label_every_n == 0 and len(pts) > 0:
            label_positions = []
            if label_mode in ("start", "both"):
                label_positions.append(pts[0])
            if label_mode in ("end", "both"):
                label_positions.append(pts[-1])
            if label_mode not in ("start", "end", "both", "none"):
                label_positions.append(pts[-1])
            for pos in label_positions:
                x0, z0 = int(pos[0]), int(pos[1])
                _draw_text_with_halo(img, str(int(tid)), (x0 + label_offset_x, z0 + label_offset_y), label_font_scale, color, 1)

    cv2.imwrite(str(out_path), img)


def save_trajectory_plot(track_df: pd.DataFrame, fit_df: pd.DataFrame | None, out_path: Path):
    plt.figure(figsize=(6, 5))
    if track_df is None or len(track_df) == 0:
        plt.xlabel("x / px")
        plt.ylabel("z / px")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return

    if "trajectory_id" not in track_df.columns:
        track_df = track_df.copy()
        track_df["trajectory_id"] = 1

    for tid, group in track_df.groupby("trajectory_id"):
        group = group.sort_values("frame")
        detected = group[group["detected"] == True]
        predicted = group[group["detected"] == False]
        if len(detected):
            plt.scatter(detected["x_px"], detected["z_px"], s=14, label=f"T{int(tid)} detected")
        if len(predicted):
            plt.scatter(predicted["x_px"], predicted["z_px"], s=14, marker="x", label=f"T{int(tid)} predicted")

    if fit_df is not None and len(fit_df):
        if "trajectory_id" not in fit_df.columns:
            fit_df = fit_df.copy()
            fit_df["trajectory_id"] = 1
        for tid, group in fit_df.groupby("trajectory_id"):
            group = group.sort_values("frame")
            plt.plot(group["x_fit_px"], group["z_fit_px"], linewidth=1.5, label=f"T{int(tid)} fit")

    plt.gca().invert_yaxis()
    plt.xlabel("x / px")
    plt.ylabel("z / px")
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) <= 20:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
