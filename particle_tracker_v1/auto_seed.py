from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd

from .detection import Detection


@dataclass
class SeedRecord:
    seed_id: int
    seed_frame: int
    x: float
    z: float
    area: float = 0.0
    mean_intensity: float = 0.0
    circularity: float = 0.0
    confidence: float = 0.0
    source: str = "auto"


def _parse_frame_indices(frame_count: int, cfg: dict) -> List[int]:
    """Return seed-frame indices for auto seed extraction.

    Supported config:
      auto_seed_frame_indices: [10, 30, 50]
      auto_seed_frame_indices: "auto"
      auto_seed_frame_count: 5
      auto_seed_frame_start_ratio: 0.15
      auto_seed_frame_end_ratio: 0.85
    """
    raw = cfg.get("auto_seed_frame_indices", "auto")
    if isinstance(raw, list) and len(raw) > 0:
        frames = [int(v) for v in raw]
    else:
        n = max(2, int(cfg.get("auto_seed_frame_count", 5)))
        r0 = float(cfg.get("auto_seed_frame_start_ratio", 0.15))
        r1 = float(cfg.get("auto_seed_frame_end_ratio", 0.85))
        r0 = max(0.0, min(1.0, r0))
        r1 = max(0.0, min(1.0, r1))
        if r1 < r0:
            r0, r1 = r1, r0
        if frame_count <= 1:
            frames = [0]
        else:
            frames = np.linspace(r0 * (frame_count - 1), r1 * (frame_count - 1), n).round().astype(int).tolist()
    frames = sorted(set(max(0, min(frame_count - 1, int(f))) for f in frames))
    return frames


def _too_close_to_existing(det: Detection, selected: List[SeedRecord], min_distance_px: float, same_frame_only: bool = True) -> bool:
    for s in selected:
        if same_frame_only and int(s.seed_frame) != int(det.frame):
            continue
        dist = math.hypot(det.x - s.x, det.z - s.z)
        if dist < min_distance_px:
            return True
    return False


def generate_auto_seed_records(
    dets_by_frame: Dict[int, List[Detection]],
    frame_count: int,
    cfg: dict,
) -> tuple[List[SeedRecord], pd.DataFrame]:
    """Generate auto seed records from multiple time-separated frames.

    This intentionally samples multiple frames instead of one middle frame, because in
    particle-lift images many particles are still attached to the wall at early times
    and only become visible after launch.
    """
    seed_frames = _parse_frame_indices(frame_count, cfg)
    min_area = float(cfg.get("auto_seed_min_area_px", cfg.get("min_area_px", 5)))
    max_area = float(cfg.get("auto_seed_max_area_px", cfg.get("max_area_px", 350)))
    min_dark = float(cfg.get("auto_seed_min_mean_dark_intensity", cfg.get("min_mean_dark_intensity", 8)))
    min_conf = float(cfg.get("auto_seed_min_confidence", 0.02))
    min_circ = float(cfg.get("auto_seed_min_circularity", cfg.get("min_circularity", 0.2)))
    min_dist = float(cfg.get("auto_seed_min_distance_px", 8.0))
    max_per_frame = int(cfg.get("auto_seed_max_per_frame", 50))
    max_total = int(cfg.get("auto_seed_max_total", 200))

    selected: List[SeedRecord] = []
    audit_rows = []

    for frame in seed_frames:
        dets = list(dets_by_frame.get(int(frame), []))
        # Stronger particles first. This improves seed quality when max_per_frame is active.
        dets.sort(key=lambda d: (d.confidence, d.mean_intensity, d.area), reverse=True)
        kept_this_frame = 0
        for d in dets:
            reason = "ok"
            if d.area < min_area or d.area > max_area:
                reason = "area_out_of_range"
            elif d.mean_intensity < min_dark:
                reason = "too_weak_darkness"
            elif d.confidence < min_conf:
                reason = "low_confidence"
            elif d.circularity < min_circ:
                reason = "low_circularity"
            elif _too_close_to_existing(d, selected, min_dist, same_frame_only=True):
                reason = "too_close_same_frame_seed"
            elif kept_this_frame >= max_per_frame:
                reason = "max_per_frame_reached"
            elif len(selected) >= max_total:
                reason = "max_total_reached"

            accept = reason == "ok"
            audit_rows.append({
                "seed_frame": int(frame),
                "x_px": float(d.x),
                "z_px": float(d.z),
                "area_px": float(d.area),
                "mean_dark_intensity": float(d.mean_intensity),
                "circularity": float(d.circularity),
                "confidence": float(d.confidence),
                "accepted_as_seed": bool(accept),
                "reason": reason,
            })
            if not accept:
                continue
            selected.append(SeedRecord(
                seed_id=len(selected) + 1,
                seed_frame=int(frame),
                x=float(d.x),
                z=float(d.z),
                area=float(d.area),
                mean_intensity=float(d.mean_intensity),
                circularity=float(d.circularity),
                confidence=float(d.confidence),
                source="auto",
            ))
            kept_this_frame += 1
            if len(selected) >= max_total:
                break
        if len(selected) >= max_total:
            break

    return selected, pd.DataFrame(audit_rows)


def manual_seed_records(seed_frame: int, seed_points: list[tuple[float, float]]) -> List[SeedRecord]:
    return [
        SeedRecord(seed_id=i + 1, seed_frame=int(seed_frame), x=float(x), z=float(z), source="manual")
        for i, (x, z) in enumerate(seed_points)
    ]


def seed_records_to_dataframe(seeds: List[SeedRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "seed_id": int(s.seed_id),
            "seed_frame": int(s.seed_frame),
            "seed_time_s": None,
            "x_px": float(s.x),
            "z_px": float(s.z),
            "area_px": float(s.area),
            "mean_dark_intensity": float(s.mean_intensity),
            "circularity": float(s.circularity),
            "confidence": float(s.confidence),
            "source": s.source,
        }
        for s in seeds
    ])



def reviewed_seed_tuples_to_records(reviewed: List[tuple[int, float, float, str]], original_records: Optional[List[SeedRecord]] = None) -> List[SeedRecord]:
    """Convert reviewed interactive seed tuples back to SeedRecord objects.

    For original auto seeds, try to preserve area/intensity/circularity/confidence metadata
    by matching frame and nearby x,z. User-added seeds have zero metadata but remain valid;
    tracking will snap them to the nearest detection in that seed frame.
    """
    original_records = original_records or []
    out: List[SeedRecord] = []
    used_original = set()
    for i, (frame, x, z, source) in enumerate(reviewed, start=1):
        best_j = None
        best_dist = 1e9
        for j, s in enumerate(original_records):
            if j in used_original:
                continue
            if int(s.seed_frame) != int(frame):
                continue
            dist = math.hypot(float(s.x) - float(x), float(s.z) - float(z))
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist <= 2.5:
            old = original_records[best_j]
            used_original.add(best_j)
            out.append(SeedRecord(
                seed_id=i,
                seed_frame=int(frame),
                x=float(x),
                z=float(z),
                area=float(old.area),
                mean_intensity=float(old.mean_intensity),
                circularity=float(old.circularity),
                confidence=float(old.confidence),
                source=str(source),
            ))
        else:
            out.append(SeedRecord(
                seed_id=i,
                seed_frame=int(frame),
                x=float(x),
                z=float(z),
                area=0.0,
                mean_intensity=0.0,
                circularity=0.0,
                confidence=0.0,
                source=str(source),
            ))
    return out
