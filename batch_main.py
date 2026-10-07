from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None

from particle_tracker_v1.io_utils import list_image_files
from particle_tracker_v1.background import estimate_background, make_pseudo_exposure
from particle_tracker_v1.ui import select_polygon_interactive
import cv2


TRACK_FILE = "EDS_track_summary.csv"
POINT_FILE = "EDS_motion_points.csv"
SEQ_FILE = "EDS_sequence_summary.csv"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(path: str | Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_pressure_pa(folder_name: str) -> float | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*pa", folder_name, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_segment_id(name: str) -> str:
    m = re.search(r"segment[_-]?([0-9]+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).zfill(3)
    return name


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def discover_tasks(batch_cfg: dict) -> pd.DataFrame:
    b = batch_cfg.get("batch", {})
    root = Path(b.get("root_dir", "."))
    particle_size_group = str(b.get("particle_size_group") or root.name)
    pressure_pattern = str(b.get("pressure_folder_pattern", f"{particle_size_group}-*pa"))
    segments_subdir = str(b.get("segments_subdir", "segments"))
    frames_subdir = str(b.get("frames_subdir", "frames_enhanced"))
    include_segments = b.get("include_segments", "all")
    exclude_segments = set(str(s).zfill(3) for s in _as_list(b.get("exclude_segments", [])))

    if include_segments == "all" or include_segments is None:
        include_set = None
    else:
        include_set = set(str(s).zfill(3) for s in _as_list(include_segments))

    rows = []
    for pdir in sorted(root.glob(pressure_pattern)):
        if not pdir.is_dir():
            continue
        pressure = parse_pressure_pa(pdir.name)
        seg_root = pdir / segments_subdir
        if not seg_root.exists():
            continue
        for seg_dir in sorted(seg_root.glob("segment*")):
            if not seg_dir.is_dir():
                continue
            seg_id = parse_segment_id(seg_dir.name)
            if include_set is not None and seg_id not in include_set:
                continue
            if seg_id in exclude_segments:
                continue
            frames_dir = seg_dir / frames_subdir
            if not frames_dir.exists():
                continue
            rows.append({
                "particle_size_group": particle_size_group,
                "pressure_folder": pdir.name,
                "pressure_Pa": pressure,
                "segment_id": seg_id,
                "segment_folder": seg_dir.name,
                "frames_dir": str(frames_dir),
            })
    return pd.DataFrame(rows)


def _infer_particle_size_mid(group: str) -> float | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)[-_]([0-9]+(?:\.[0-9]+)?)", group)
    if not m:
        return None
    return 0.5 * (float(m.group(1)) + float(m.group(2)))


def make_sequence_config(base_cfg: dict, batch_cfg: dict, task: dict, out_dir: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    single = batch_cfg.get("single_sequence_config", {})
    # Single-sequence defaults for batch mode: disable all interactive windows.
    cfg.update({
        "input_dir": task["frames_dir"],
        "output_dir": str(out_dir),
        "seed_mode": single.get("seed_mode", "auto"),
        "interactive_roi": False,
        "interactive_seed": False,
        "auto_seed_tuning_enabled": False,
        "auto_seed_review_enabled": False,
        "output_profile": "compact",
        "save_debug_outputs": False,
        "save_trajectory_plot": False,
    })
    # User overrides from batch_config.yaml.
    for k, v in single.items():
        if k == "base_config":
            continue
        cfg[k] = v

    meta = batch_cfg.get("metadata", {})
    particle_group = task.get("particle_size_group", meta.get("particle_size_group", ""))
    cfg.update({
        "sequence_id": f"{particle_group}_{int(task['pressure_Pa']) if pd.notna(task.get('pressure_Pa')) else 'unknown'}Pa_segment_{task['segment_id']}",
        "experiment_group": meta.get("experiment_group", particle_group),
        "particle_size_group": particle_group,
        "pressure_Pa": None if pd.isna(task.get("pressure_Pa")) else float(task.get("pressure_Pa")),
        "segment_id": task.get("segment_id"),
        "pressure_folder": task.get("pressure_folder"),
        "frames_dir": task.get("frames_dir"),
    })
    # Optional particle metadata.
    for k, v in meta.items():
        cfg.setdefault(k, v)
    if cfg.get("particle_diameter_um", None) is None:
        mid = meta.get("particle_diameter_um_mid", None)
        if mid is None:
            mid = _infer_particle_size_mid(str(particle_group))
        if mid is not None:
            cfg["particle_diameter_um"] = float(mid)
    return cfg


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def add_task_columns(df: pd.DataFrame, task: dict) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    for k in ["particle_size_group", "pressure_folder", "pressure_Pa", "segment_id", "frames_dir"]:
        df.insert(0, k, task.get(k, None))
    return df


def success_done(out_dir: Path) -> bool:
    seq_path = out_dir / SEQ_FILE
    trk_path = out_dir / TRACK_FILE
    return seq_path.exists() and trk_path.exists()


def _normalize_roi_points(points: Any) -> list[list[int]]:
    if not points:
        return []
    out = []
    for p in points:
        if isinstance(p, dict):
            x = p.get("x", p.get("X", None))
            y = p.get("z", p.get("y", p.get("Y", None)))
        else:
            x, y = p[0], p[1]
        out.append([int(round(float(x))), int(round(float(y)))])
    return out


def _load_roi_points_json(path: Path) -> list[list[int]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "roi_points_px" in data:
            data = data["roi_points_px"]
        return _normalize_roi_points(data)
    except Exception:
        return []


def _save_roi_points_json(path: Path, points: list[list[int]], meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "roi_points_px": points,
        "metadata": meta or {},
        "saved_at": now_str(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_batch_roi(batch_cfg: dict, tasks: pd.DataFrame, project_dir: Path, out_root: Path) -> dict:
    """V3.1: choose ROI once on the first available sequence and reuse it for all tasks.

    Priority:
      1) single_sequence_config.roi_points_px if provided;
      2) existing batch_summary/batch_roi_points.json if reuse_existing=true;
      3) interactive selection on the pseudo-exposure image of the first task if batch_roi_select_once=true;
      4) full image ROI in main.py if all above are disabled.
    """
    single = batch_cfg.setdefault("single_sequence_config", {})
    existing = _normalize_roi_points(single.get("roi_points_px", None))
    summary_dir = out_root / "batch_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    roi_cfg = batch_cfg.get("batch_roi", {})
    select_once = bool(roi_cfg.get("batch_roi_select_once", True))
    reuse_existing = bool(roi_cfg.get("reuse_existing", True))
    roi_file_raw = roi_cfg.get("roi_points_file", str(summary_dir / "batch_roi_points.json"))
    roi_file = Path(roi_file_raw)
    if not roi_file.is_absolute():
        roi_file = summary_dir / roi_file

    if len(existing) >= 3:
        single["roi_points_px"] = existing
        _save_roi_points_json(roi_file, existing, {"source": "batch_config.single_sequence_config.roi_points_px"})
        print(f"批处理 ROI：使用 batch_config.yaml 中的固定 ROI，顶点数: {len(existing)}")
        return batch_cfg

    if reuse_existing:
        loaded = _load_roi_points_json(roi_file)
        if len(loaded) >= 3:
            single["roi_points_px"] = loaded
            print(f"批处理 ROI：复用已保存 ROI: {roi_file}")
            return batch_cfg

    if not select_once:
        print("批处理 ROI：未启用首次选择；未提供 roi_points_px 时，单序列将默认使用全图。")
        return batch_cfg

    if tasks is None or len(tasks) == 0:
        print("批处理 ROI：未发现任务，无法选择 ROI。")
        return batch_cfg

    ref_mode = str(roi_cfg.get("reference_task", "first"))
    ref_idx = 0
    if ref_mode.isdigit():
        ref_idx = max(0, min(int(ref_mode), len(tasks) - 1))
    task = tasks.iloc[ref_idx].to_dict()
    frames_dir = Path(task["frames_dir"])
    file_exts = batch_cfg.get("single_sequence_config", {}).get("file_extensions", [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])
    image_files = list_image_files(str(frames_dir), file_exts)
    if len(image_files) == 0:
        raise RuntimeError(f"用于 ROI 选择的参考序列没有图像: {frames_dir}")

    sample_count = int(roi_cfg.get("background_sample_count", batch_cfg.get("single_sequence_config", {}).get("background_sample_count", 80)))
    print("\nV3.1 首次批处理 ROI 选择")
    print(f"参考序列: pressure={task.get('pressure_folder')}, segment={task.get('segment_id')}")
    print(f"参考图像目录: {frames_dir}")
    print("正在生成参考伪曝光图...")
    background = estimate_background(image_files, sample_count)
    pseudo = make_pseudo_exposure(image_files, background)
    pseudo_path = summary_dir / "batch_roi_reference_pseudo_exposure.png"
    cv2.imwrite(str(pseudo_path), pseudo)

    points = select_polygon_interactive(
        pseudo,
        window_name="V3.1 Select ONE batch ROI for all segments",
        max_points=int(roi_cfg.get("roi_max_points", 4)),
    )
    if points is None or len(points) < 3:
        raise RuntimeError("批处理 ROI 选择被取消，无法继续。")
    points = _normalize_roi_points(points)
    single["roi_points_px"] = points
    _save_roi_points_json(roi_file, points, {
        "source": "interactive_first_sequence",
        "reference_pressure_folder": task.get("pressure_folder"),
        "reference_segment_id": task.get("segment_id"),
        "reference_frames_dir": str(frames_dir),
        "reference_pseudo_exposure": str(pseudo_path),
    })
    print(f"批处理 ROI 已保存: {roi_file}")
    print(f"后续所有任务将共用该 ROI: {points}")
    return batch_cfg


def run_one_task(task: dict, base_cfg: dict, batch_cfg: dict, project_dir: Path, segment_out: Path) -> dict:
    segment_out.mkdir(parents=True, exist_ok=True)
    cfg = make_sequence_config(base_cfg, batch_cfg, task, segment_out)
    cfg_path = segment_out / "config_used.yaml"
    save_yaml(cfg_path, cfg)

    main_py = project_dir / "main.py"
    start = time.time()
    log_path = segment_out / "run_stdout_stderr.txt"
    cmd = [sys.executable, str(main_py), "--config", str(cfg_path)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        cwd=str(project_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    elapsed = time.time() - start

    row = {
        **task,
        "output_dir": str(segment_out),
        "config_used": str(cfg_path),
        "start_time": None,
        "end_time": now_str(),
        "elapsed_s": elapsed,
        "return_code": proc.returncode,
        "processing_status": "success" if proc.returncode == 0 and success_done(segment_out) else "failed",
        "error_message": "" if proc.returncode == 0 else f"main.py returned {proc.returncode}; see run_stdout_stderr.txt",
    }
    seq = read_csv_if_exists(segment_out / SEQ_FILE)
    if len(seq) > 0:
        for c in seq.columns:
            if c not in row:
                val = seq.iloc[0][c]
                try:
                    if pd.isna(val):
                        val = None
                except Exception:
                    pass
                row[f"seq_{c}"] = val
    return row


def concat_and_save(dfs: list[pd.DataFrame], path: Path) -> pd.DataFrame:
    good = [d for d in dfs if d is not None and len(d) > 0]
    out = pd.concat(good, ignore_index=True) if good else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def _stats_for_series(s: pd.Series, prefix: str) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(len(x)),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}_median": float(x.median()),
        f"{prefix}_p25": float(x.quantile(0.25)),
        f"{prefix}_p75": float(x.quantile(0.75)),
        f"{prefix}_p90": float(x.quantile(0.90)),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_max": float(x.max()),
    }


def make_pressure_statistics(all_tracks: pd.DataFrame, all_sequences: pd.DataFrame, batch_cfg: dict) -> pd.DataFrame:
    if all_tracks is None or len(all_tracks) == 0:
        return pd.DataFrame()
    rows = []
    long_jump_threshold = batch_cfg.get("analysis", {}).get("long_jump_threshold_mm", None)
    for pressure, g in all_tracks.groupby("pressure_Pa", dropna=False):
        row = {"pressure_Pa": pressure, "n_tracks_valid": int(len(g))}
        if all_sequences is not None and len(all_sequences) > 0:
            sg = all_sequences[all_sequences["pressure_Pa"].astype(str) == str(pressure)]
            row["n_segments"] = int(len(sg))
        else:
            row["n_segments"] = int(g["segment_id"].nunique()) if "segment_id" in g.columns else np.nan
        if row.get("n_segments", 0):
            row["tracks_per_segment_mean"] = row["n_tracks_valid"] / max(row["n_segments"], 1)
        cols = {
            "jump_height_mm": "jump_height_mm",
            "horizontal_displacement_mm_abs": "horizontal_displacement_mm",
            "transfer_time_ms": "transfer_time_ms",
            "v0_mm_s": "V0_mm_s",
            "v0_star_z_mm_s": "V0_star_z_mm_s",
            "v0_star_total_mm_s_from_theta": "V0_star_total_mm_s",
            "a0_mm_s2": "a0_mm_s2",
            "theta0_deg_abs": "theta0_deg_abs",
        }
        for col, pref in cols.items():
            if col in g.columns:
                row.update(_stats_for_series(g[col], pref))
        if long_jump_threshold is not None and "horizontal_displacement_mm_abs" in g.columns:
            vals = pd.to_numeric(g["horizontal_displacement_mm_abs"], errors="coerce")
            row["long_jump_threshold_mm"] = float(long_jump_threshold)
            row["long_jump_ratio"] = float((vals > float(long_jump_threshold)).mean())
        if "seed_frame" in g.columns:
            row["unique_seed_count"] = int(g[["segment_id", "trajectory_id"]].drop_duplicates().shape[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    if "pressure_Pa" in out.columns:
        out = out.sort_values("pressure_Pa", na_position="last").reset_index(drop=True)
    return out


def make_basic_plots(pressure_stats: pd.DataFrame, out_dir: Path):
    if plt is None or pressure_stats is None or len(pressure_stats) == 0 or "pressure_Pa" not in pressure_stats.columns:
        return
    plots = [
        ("jump_height_mm_median", "pressure_vs_jump_height.png", "Jump height median (mm)"),
        ("horizontal_displacement_mm_median", "pressure_vs_horizontal_displacement.png", "Horizontal displacement median (mm)"),
        ("transfer_time_ms_median", "pressure_vs_transfer_time.png", "Transfer time median (ms)"),
        ("V0_mm_s_median", "pressure_vs_V0.png", "Apparent V0 median (mm/s)"),
        ("V0_star_z_mm_s_median", "pressure_vs_V0_star_z.png", "Energy-inferred vertical V0* median (mm/s)"),
        ("theta0_deg_abs_median", "pressure_vs_launch_angle.png", "Launch angle median (deg)"),
        ("a0_mm_s2_median", "pressure_vs_a0.png", "Initial acceleration median (mm/s²)"),
    ]
    x = pd.to_numeric(pressure_stats["pressure_Pa"], errors="coerce")
    for col, name, ylabel in plots:
        if col not in pressure_stats.columns:
            continue
        y = pd.to_numeric(pressure_stats[col], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() == 0:
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(x[ok], y[ok], marker="o")
        plt.xlabel("Pressure (Pa)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=200)
        plt.close()


def aggregate_results(tasks_df: pd.DataFrame, out_root: Path, batch_cfg: dict) -> dict:
    summary_dir = out_root / "batch_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    track_dfs, point_dfs, seq_dfs = [], [], []
    for _, task in tasks_df.iterrows():
        taskd = task.to_dict()
        seg_out = Path(taskd["output_dir"])
        tr = add_task_columns(read_csv_if_exists(seg_out / TRACK_FILE), taskd)
        pt = add_task_columns(read_csv_if_exists(seg_out / POINT_FILE), taskd)
        sq = add_task_columns(read_csv_if_exists(seg_out / SEQ_FILE), taskd)
        if len(tr):
            track_dfs.append(tr)
        if len(pt):
            point_dfs.append(pt)
        if len(sq):
            seq_dfs.append(sq)

    all_tracks = concat_and_save(track_dfs, summary_dir / "all_tracks_summary.csv")
    all_points = concat_and_save(point_dfs, summary_dir / "all_motion_points.csv")
    all_sequences = concat_and_save(seq_dfs, summary_dir / "all_sequence_summary.csv")

    pressure_stats = make_pressure_statistics(all_tracks, all_sequences, batch_cfg)
    pressure_stats.to_csv(summary_dir / "pressure_statistics.csv", index=False, encoding="utf-8-sig")

    # Pressure-specific track tables for easy manual checking.
    if len(all_tracks) > 0 and "pressure_Pa" in all_tracks.columns:
        pressure_dir = summary_dir / "by_pressure"
        pressure_dir.mkdir(exist_ok=True)
        for p, g in all_tracks.groupby("pressure_Pa", dropna=False):
            label = "unknown" if pd.isna(p) else f"{int(float(p))}Pa"
            g.to_csv(pressure_dir / f"pressure_{label}_track_summary.csv", index=False, encoding="utf-8-sig")

    if bool(batch_cfg.get("analysis", {}).get("save_basic_plots", True)):
        make_basic_plots(pressure_stats, summary_dir)

    return {
        "all_tracks": all_tracks,
        "all_points": all_points,
        "all_sequences": all_sequences,
        "pressure_stats": pressure_stats,
        "summary_dir": summary_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Particle Tracker V3 batch processor for pressure/segment statistics")
    parser.add_argument("--batch_config", default="batch_config.yaml", help="Path to batch_config.yaml")
    parser.add_argument("--discover_only", action="store_true", help="Only scan folders and write batch_tasks.csv")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    batch_cfg = load_yaml(args.batch_config)
    b = batch_cfg.get("batch", {})
    root = Path(b.get("root_dir", "."))
    out_root = Path(b.get("output_dir", root / "V3_batch_output"))
    out_root.mkdir(parents=True, exist_ok=True)
    summary_dir = out_root / "batch_summary"
    summary_dir.mkdir(exist_ok=True)

    base_config_path = Path(batch_cfg.get("single_sequence_config", {}).get("base_config", project_dir / "config.yaml"))
    if not base_config_path.is_absolute():
        base_config_path = project_dir / base_config_path
    base_cfg = load_yaml(base_config_path)

    print(f"扫描批处理根目录: {root}")
    tasks = discover_tasks(batch_cfg)
    if len(tasks) == 0:
        print("未发现任何 frames_enhanced 任务。请检查 root_dir、pressure_folder_pattern、segments_subdir、frames_subdir。")
        return
    tasks.to_csv(summary_dir / "batch_tasks.csv", index=False, encoding="utf-8-sig")
    print(f"发现任务数量: {len(tasks)}")
    if args.discover_only:
        print(f"已保存任务表: {summary_dir / 'batch_tasks.csv'}")
        return

    # V3.1: select a single batch ROI on the first available sequence, save it,
    # and inject the coordinates into every generated single-sequence config.
    batch_cfg = ensure_batch_roi(batch_cfg, tasks, project_dir, out_root)

    skip_if_done = bool(b.get("skip_if_done", True))
    overwrite = bool(b.get("overwrite_existing", False))
    continue_on_error = bool(b.get("continue_on_error", True))

    log_rows = []
    task_rows_with_out = []
    for idx, row in tasks.iterrows():
        task = row.to_dict()
        pressure_folder = str(task["pressure_folder"])
        seg_id = str(task["segment_id"])
        segment_out = out_root / pressure_folder / f"segment_{seg_id}"
        task["output_dir"] = str(segment_out)
        task_rows_with_out.append(task)
        print(f"\n[{idx+1}/{len(tasks)}] pressure={pressure_folder}, segment={seg_id}")
        if skip_if_done and not overwrite and success_done(segment_out):
            print("  已完成，跳过。")
            log_rows.append({**task, "processing_status": "skipped_done", "elapsed_s": 0, "error_message": ""})
            continue
        try:
            res = run_one_task(task, base_cfg, batch_cfg, project_dir, segment_out)
            print(f"  状态: {res['processing_status']}，耗时 {res['elapsed_s']:.1f} s")
            log_rows.append(res)
        except Exception as e:
            err = str(e)
            print(f"  失败: {err}")
            log_rows.append({**task, "processing_status": "failed", "elapsed_s": np.nan, "error_message": err})
            if not continue_on_error:
                break
        pd.DataFrame(log_rows).to_csv(summary_dir / "batch_processing_log.csv", index=False, encoding="utf-8-sig")

    tasks_out = pd.DataFrame(task_rows_with_out)
    tasks_out.to_csv(summary_dir / "batch_tasks_with_output.csv", index=False, encoding="utf-8-sig")
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(summary_dir / "batch_processing_log.csv", index=False, encoding="utf-8-sig")
    failed = log_df[log_df["processing_status"].astype(str).str.contains("failed", na=False)] if len(log_df) else pd.DataFrame()
    failed.to_csv(summary_dir / "batch_failed_tasks.csv", index=False, encoding="utf-8-sig")

    print("\n汇总所有 segment 结果...")
    aggregate_results(tasks_out, out_root, batch_cfg)
    print("完成 V3 批处理。核心输出：")
    print(f"  {summary_dir / 'all_tracks_summary.csv'}")
    print(f"  {summary_dir / 'all_motion_points.csv'}")
    print(f"  {summary_dir / 'all_sequence_summary.csv'}")
    print(f"  {summary_dir / 'pressure_statistics.csv'}")
    print(f"  {summary_dir / 'batch_processing_log.csv'}")


if __name__ == "__main__":
    main()
