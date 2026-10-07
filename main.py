from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import yaml
import pandas as pd

from particle_tracker_v1.io_utils import list_image_files, read_gray, ensure_dir
from particle_tracker_v1.background import estimate_background, make_pseudo_exposure
from particle_tracker_v1.detection import detect_dark_spots, detections_to_dataframe
from particle_tracker_v1.tracking import track_multiple_seed_records, filter_tracks_by_global_flight_model, remove_duplicate_tracks, prune_tracks_by_parabola_residual
from particle_tracker_v1.fit import add_physical_units, fit_time_polynomial, extrapolate_z0_crossings
from particle_tracker_v1.visualize import save_trajectory_overlay, save_trajectory_plot
from particle_tracker_v1.statistics import write_eds_statistics
from particle_tracker_v1.ui import select_polygon_interactive, select_multi_seeds_interactive, polygon_to_mask, review_auto_seeds_interactive
from particle_tracker_v1.auto_seed import generate_auto_seed_records, manual_seed_records, seed_records_to_dataframe, reviewed_seed_tuples_to_records
from particle_tracker_v1.auto_seed_tuner import tune_auto_seed_parameters_interactive


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_dump(path: Path, data: dict | list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fit_multiple_tracks(track_df: pd.DataFrame, cfg: dict, out_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame, list[dict]]:
    fit_frames = []
    summary_rows = []
    fit_infos: list[dict] = []
    degree = int(cfg.get("fit_degree", 2))
    min_points = int(cfg.get("min_points_for_fit", 6))
    if track_df is None or len(track_df) == 0:
        return None, pd.DataFrame(), []

    # wall_z_px can be a number or "auto".
    # "auto" uses the bottom edge of the polygon ROI when available;
    # otherwise it falls back to the largest z value in accepted tracks.
    wall_z_raw = cfg.get("wall_z_px", 0.0)
    if isinstance(wall_z_raw, str) and wall_z_raw.lower() == "auto":
        roi_points_runtime = cfg.get("_runtime_roi_points", None)
        if roi_points_runtime is not None and len(roi_points_runtime) > 0:
            wall_z = float(max(float(p[1]) for p in roi_points_runtime))
        elif "z_px" in track_df.columns and len(track_df) > 0:
            wall_z = float(track_df["z_px"].max())
        else:
            wall_z = 0.0
    else:
        wall_z = float(wall_z_raw)

    for tid, group in track_df.groupby("trajectory_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        detected_points = int(group["detected"].sum())
        predicted_points = int((~group["detected"].astype(bool)).sum())
        info = {
            "trajectory_id": int(tid),
            "status": "ok",
            "start_frame": int(group["frame"].min()),
            "end_frame": int(group["frame"].max()),
            "start_time_s": float(group["time_s"].min()) if "time_s" in group.columns else None,
            "end_time_s": float(group["time_s"].max()) if "time_s" in group.columns else None,
            "start_time_ms": float(group["time_ms"].min()) if "time_ms" in group.columns else None,
            "end_time_ms": float(group["time_ms"].max()) if "time_ms" in group.columns else None,
            "duration_s": float(group["time_s"].max() - group["time_s"].min()) if "time_s" in group.columns else None,
            "duration_ms": float(group["time_ms"].max() - group["time_ms"].min()) if "time_ms" in group.columns else None,
            "points": int(len(group)),
            "detected_points": detected_points,
            "predicted_points": predicted_points,
            "mean_confidence": float(group["confidence"].mean()) if len(group) else 0.0,
        }
        try:
            fit_df, fit_info = fit_time_polynomial(group, degree=degree, min_points=min_points)
            fit_df.insert(0, "trajectory_id", int(tid))
            fps = float(cfg.get("fps", 1000.0))
            fit_df["time_s"] = fit_df["frame"] / fps
            fit_df["time_ms"] = fit_df["time_s"] * 1000.0
            fit_frames.append(fit_df)
            crossings = extrapolate_z0_crossings(
                fit_info["poly_z_desc"],
                fit_info["poly_x_desc"],
                fit_info["t0_frame"],
                wall_z,
            )
            fit_info.update(crossings)
            info.update(fit_info)
            fit_infos.append({"trajectory_id": int(tid), **fit_info})
        except Exception as e:
            info["status"] = "fit_failed"
            info["fit_error"] = str(e)
            fit_infos.append({"trajectory_id": int(tid), "fit_error": str(e)})
        summary_rows.append(info)

    fit_all = pd.concat(fit_frames, ignore_index=True) if fit_frames else None
    summary_df = pd.DataFrame(summary_rows)
    return fit_all, summary_df, fit_infos


def main():
    parser = argparse.ArgumentParser(description="Particle Tracker V2.3 - compact EDS statistics + V3-ready sequence summary")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    save_debug = bool(cfg.get("save_debug_outputs", False))
    compact = str(cfg.get("output_profile", "compact")).lower() == "compact"

    image_files = list_image_files(cfg["input_dir"], cfg.get("file_extensions", [".png", ".jpg", ".tif"]))
    out_dir = ensure_dir(cfg.get("output_dir", "output"))

    print(f"读取图像数量: {len(image_files)}")
    print("估计背景...")
    background = estimate_background(image_files, int(cfg.get("background_sample_count", 80)))
    if save_debug:
        cv2.imwrite(str(out_dir / "background.png"), background.clip(0, 255).astype("uint8"))

    print("生成伪曝光图...")
    pseudo = make_pseudo_exposure(image_files, background)
    cv2.imwrite(str(out_dir / "pseudo_exposure.png"), pseudo)  # kept as a lightweight QC image

    seed_frame = cfg.get("seed_frame_index", None)
    if seed_frame is None:
        seed_frame = len(image_files) // 2
    seed_frame = int(seed_frame)
    seed_frame = max(0, min(seed_frame, len(image_files) - 1))
    seed_img = read_gray(image_files[seed_frame])
    print(f"种子帧: {seed_frame}")

    roi_points = None
    roi_mask = None

    # V3: batch processing must run without interactive ROI selection.
    # If interactive_roi=false, users can provide roi_points_px in config.yaml, e.g.
    # roi_points_px: [[120, 680], [950, 650], [980, 420], [130, 440]]
    preset_roi = cfg.get("roi_points_px", None)
    if preset_roi is not None and len(preset_roi) >= 3:
        roi_points = [(int(p[0]), int(p[1])) for p in preset_roi]
        print(f"使用配置文件中的 ROI 顶点，顶点数: {len(roi_points)}")
    elif bool(cfg.get("interactive_roi", True)):
        print("请在伪曝光图上点击四个点，形成多边形 ROI。")
        roi_points = select_polygon_interactive(
            pseudo,
            window_name="Select polygon trajectory ROI on pseudo-exposure",
            max_points=int(cfg.get("roi_max_points", 4)),
        )

    if roi_points is not None:
        # Store ROI points in runtime config so later fitting can resolve wall_z_px: "auto".
        cfg["_runtime_roi_points"] = roi_points
        roi_mask = polygon_to_mask(pseudo.shape, roi_points, padding_px=int(cfg.get("roi_display_padding_px", 0)))
        cv2.imwrite(str(out_dir / "roi_mask.png"), roi_mask)
        _json_dump(out_dir / "roi_points.json", [{"x": x, "z": y} for x, y in roi_points])
        print(f"已选择多边形 ROI，顶点数: {len(roi_points)}")
    else:
        print("未选择 ROI，将使用全图。")

    seed_mode = str(cfg.get("seed_mode", "manual")).lower()
    seed_records = []

    if seed_mode == "manual":
        seed_points = []
        if bool(cfg.get("interactive_seed", True)):
            seed_points = select_multi_seeds_interactive(
                seed_img,
                roi_points=roi_points,
                roi_mask=roi_mask,
                window_name="Select multiple seed particles",
                max_seeds=int(cfg.get("max_manual_seeds", 100)),
            )
        else:
            raise RuntimeError("manual 模式需要 interactive_seed=true。")
        seed_records = manual_seed_records(seed_frame, seed_points)
        print(f"已手动选择种子颗粒数量: {len(seed_records)}")

    elif seed_mode == "auto":
        print("自动种子模式：将先完成所有帧暗斑检测，再从多张时间帧中抽取种子。")
        if bool(cfg.get("auto_seed_tuning_enabled", False)):
            print("打开自动种子实时滑块调参窗口。确认后，当前参数会用于后续全帧检测和自动种子生成。")
            cfg = tune_auto_seed_parameters_interactive(image_files, background, roi_mask, cfg, out_dir)
    else:
        raise RuntimeError("seed_mode 只支持 manual 或 auto。")

    print("逐帧检测暗斑候选颗粒...")
    dets_by_frame = {}
    for i, p in enumerate(image_files):
        frame = read_gray(p)
        dets_by_frame[i] = detect_dark_spots(frame, background, i, roi_mask, cfg)
        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{len(image_files)} 帧")

    det_df = detections_to_dataframe(dets_by_frame)
    if len(det_df) > 0:
        fps = float(cfg.get("fps", 1000.0))
        det_df["time_s"] = det_df["frame"] / fps
        det_df["time_ms"] = det_df["time_s"] * 1000.0
    if save_debug:
        det_df.to_csv(out_dir / "detections.csv", index=False, encoding="utf-8-sig")
    print(f"候选暗斑总数: {len(det_df)}")

    if seed_mode == "auto":
        print("从多张时间帧中自动抽取种子颗粒...")
        seed_records, auto_seed_audit = generate_auto_seed_records(dets_by_frame, len(image_files), cfg)
        if len(auto_seed_audit) > 0:
            fps = float(cfg.get("fps", 1000.0))
            auto_seed_audit["seed_time_s"] = auto_seed_audit["seed_frame"] / fps
            auto_seed_audit["seed_time_ms"] = auto_seed_audit["seed_time_s"] * 1000.0
        if save_debug:
            auto_seed_audit.to_csv(out_dir / "auto_seed_audit.csv", index=False, encoding="utf-8-sig")
        print(f"自动种子数量: {len(seed_records)}")

        # V2.1: optional interactive review window for auto seeds.
        # It is useful for parameter tuning: the user can immediately see whether
        # auto_seed_* settings are too strict, too loose, or biased toward noise.
        if bool(cfg.get("auto_seed_review_enabled", True)) and len(seed_records) > 0:
            print("弹出自动种子确认窗口。可左键补种子、右键删除误种子、Enter 接受。")
            seed_frames = sorted(set(int(s.seed_frame) for s in seed_records))
            review_images = {int(f): read_gray(image_files[int(f)]) for f in seed_frames}
            reviewed = review_auto_seeds_interactive(
                review_images,
                seed_records,
                roi_mask=roi_mask,
                window_name="Review auto-selected seed particles",
                max_extra_seeds=int(cfg.get("auto_seed_review_max_total", cfg.get("auto_seed_max_total", 300))),
            )
            seed_records = reviewed_seed_tuples_to_records(reviewed, seed_records)
            print(f"确认后种子数量: {len(seed_records)}")

    if len(seed_records) == 0:
        print("没有可用种子。manual 模式请重新选点；auto 模式请放宽 auto_seed_* 或检测阈值。")
        return

    seeds_df = seed_records_to_dataframe(seed_records)
    fps = float(cfg.get("fps", 1000.0))
    if len(seeds_df) > 0:
        seeds_df["seed_time_s"] = seeds_df["seed_frame"] / fps
        seeds_df["seed_time_ms"] = seeds_df["seed_time_s"] * 1000.0
    seeds_df.to_csv(out_dir / "seed_points.csv", index=False, encoding="utf-8-sig")
    _json_dump(out_dir / "seed_points.json", seeds_df.to_dict(orient="records"))

    print("执行多种子双向追踪：Kalman 短期预测 + 局部抛物线相容性判别...")
    raw_tracks_df, tracking_status = track_multiple_seed_records(dets_by_frame, seed_records, len(image_files), cfg, roi_mask=roi_mask)
    if len(tracking_status) > 0:
        tracking_status["seed_time_s"] = tracking_status["seed_frame"] / fps
        tracking_status["seed_time_ms"] = tracking_status["seed_time_s"] * 1000.0
    if save_debug:
        tracking_status.to_csv(out_dir / "tracking_status.csv", index=False, encoding="utf-8-sig")

    if raw_tracks_df is None or len(raw_tracks_df) == 0:
        print("没有成功追踪到原始轨迹。请检查 ROI、种子点、暗斑阈值或 gate 参数。")
        return

    raw_tracks_df = add_physical_units(raw_tracks_df, float(cfg.get("fps", 1000.0)), float(cfg.get("pixel_size_mm", 0.01)))
    if "time_s" in raw_tracks_df.columns:
        raw_tracks_df["time_ms"] = raw_tracks_df["time_s"] * 1000.0
    # Motion timestamps relative to this trajectory and relative to its seed frame.
    raw_tracks_df["track_time_s"] = raw_tracks_df.groupby("trajectory_id")["time_s"].transform(lambda v: v - v.min())
    raw_tracks_df["track_time_ms"] = raw_tracks_df["track_time_s"] * 1000.0
    raw_tracks_df["seed_time_s"] = raw_tracks_df["seed_frame"] / fps
    raw_tracks_df["seed_time_ms"] = raw_tracks_df["seed_time_s"] * 1000.0
    raw_tracks_df["time_from_seed_s"] = raw_tracks_df["time_s"] - raw_tracks_df["seed_time_s"]
    raw_tracks_df["time_from_seed_ms"] = raw_tracks_df["time_from_seed_s"] * 1000.0
    if save_debug:
        raw_tracks_df.to_csv(out_dir / "trajectories_raw_all.csv", index=False, encoding="utf-8-sig")

    print("执行轨迹异常点剔除：删除偏离整体抛物线趋势的孤立噪声点...")
    pruned_tracks_df, pruning_summary = prune_tracks_by_parabola_residual(raw_tracks_df, cfg)
    if save_debug:
        pruning_summary.to_csv(out_dir / "track_outlier_pruning_summary.csv", index=False, encoding="utf-8-sig")
    if pruned_tracks_df is not None and len(pruned_tracks_df) > 0:
        if save_debug:
            pruned_tracks_df.to_csv(out_dir / "trajectories_pruned_all.csv", index=False, encoding="utf-8-sig")
    else:
        pruned_tracks_df = raw_tracks_df

    print("执行全局抛物线飞行验证：下边界 → 上升 → 顶点 → 下降 → 下边界...")
    tracks_df, rejected_df, flight_validation = filter_tracks_by_global_flight_model(pruned_tracks_df, cfg, roi_mask=roi_mask)
    if save_debug:
        flight_validation.to_csv(out_dir / "flight_validation.csv", index=False, encoding="utf-8-sig")
    if rejected_df is not None and len(rejected_df) > 0:
        if save_debug:
            rejected_df.to_csv(out_dir / "trajectories_rejected.csv", index=False, encoding="utf-8-sig")

    if tracks_df is None or len(tracks_df) == 0:
        print("所有原始轨迹均未通过完整抛物线飞行验证。")
        print("可查看 output/trajectories_raw_all.csv 和 output/flight_validation.csv 判断被拒原因。")
        return

    print("执行重复轨迹合并/删除：避免多时间帧自动种子追踪到同一颗粒...")
    tracks_before_duplicate_filter = tracks_df.copy()
    if save_debug:
        tracks_before_duplicate_filter.to_csv(out_dir / "trajectories_before_duplicate_filter.csv", index=False, encoding="utf-8-sig")
    tracks_df, duplicate_summary = remove_duplicate_tracks(tracks_df, cfg)
    if save_debug:
        duplicate_summary.to_csv(out_dir / "duplicate_track_summary.csv", index=False, encoding="utf-8-sig")

    if tracks_df is None or len(tracks_df) == 0:
        print("通过飞行验证的轨迹均被重复过滤移除。请检查 duplicate_track_summary.csv。")
        return

    # Final accepted detailed trajectory table is optional in V2.3 compact mode.

    # Save detailed accepted trajectories only when requested.
    if bool(cfg.get("save_detailed_trajectories", True)):
        tracks_df.to_csv(out_dir / "trajectories_all.csv", index=False, encoding="utf-8-sig")
    if save_debug:
        indiv_dir = ensure_dir(out_dir / "individual_trajectories")
        for tid, group in tracks_df.groupby("trajectory_id"):
            group.to_csv(indiv_dir / f"trajectory_{int(tid):03d}.csv", index=False, encoding="utf-8-sig")

    print("进行每条轨迹的时间参数二次拟合与 z=0 外推...")
    fit_df, summary_df, fit_infos = fit_multiple_tracks(tracks_df, cfg, out_dir)
    if save_debug:
        if fit_df is not None:
            fit_df.to_csv(out_dir / "trajectory_fit_all.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(out_dir / "summary_fit_legacy.csv", index=False, encoding="utf-8-sig")
        _json_dump(out_dir / "fit_summary_all.json", fit_infos)

    print("生成 EDS 统计分析输出...")
    eds_outputs = write_eds_statistics(tracks_df, cfg, out_dir)

    print("保存多轨迹可视化结果...")
    # Compact mode keeps the no-label overlay as the primary QC image.
    if save_debug:
        save_trajectory_overlay(pseudo, tracks_df, out_dir / "trajectory_overlay_all.png", cfg=cfg, fit_df=fit_df)
    cfg_no_label = dict(cfg)
    cfg_no_label["overlay_label_mode"] = "none"
    save_trajectory_overlay(pseudo, tracks_df, out_dir / "trajectory_overlay_no_labels.png", cfg=cfg_no_label, fit_df=fit_df)
    if save_debug or bool(cfg.get("save_trajectory_plot", False)):
        save_trajectory_plot(tracks_df, fit_df, out_dir / "trajectory_plot_all.png")

    print("完成。V2.3 主要输出文件：")
    print(f"  {out_dir / 'EDS_track_summary.csv'}")
    print(f"  {out_dir / 'EDS_motion_points.csv'}")
    print(f"  {out_dir / 'EDS_acceleration_by_x.csv'}")
    print(f"  {out_dir / 'EDS_launch_angle_by_x.csv'}")
    print(f"  {out_dir / 'EDS_sequence_summary.csv'}")
    print(f"  {out_dir / 'trajectory_overlay_no_labels.png'}")
    print(f"  {out_dir / 'seed_points.csv'}")
    if save_debug:
        print("  Debug outputs are enabled; extra intermediate CSV files were also saved.")


if __name__ == "__main__":
    main()
