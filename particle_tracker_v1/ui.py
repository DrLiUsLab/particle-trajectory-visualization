from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

Point = Tuple[int, int]


def _draw_label_text(canvas: np.ndarray, text: Optional[str]) -> np.ndarray:
    if text:
        y0 = 24
        for line in text.split("\n"):
            cv2.putText(canvas, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
            y0 += 26
    return canvas


def _draw_points_and_polygon(
    image_bgr: np.ndarray,
    points: Sequence[Point],
    closed: bool = False,
    point_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (0, 255, 255),
    text: Optional[str] = None,
) -> np.ndarray:
    """Draw ROI vertices and polygon/polyline. Used only for ROI selection."""
    canvas = image_bgr.copy()
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=closed, color=line_color, thickness=2)
    for i, (x, y) in enumerate(points, start=1):
        cv2.circle(canvas, (int(x), int(y)), 5, point_color, -1)
        cv2.putText(canvas, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, point_color, 2, cv2.LINE_AA)
    return _draw_label_text(canvas, text)


def _draw_seed_points_only(
    image_bgr: np.ndarray,
    seeds: Sequence[Point],
    text: Optional[str] = None,
) -> np.ndarray:
    """Draw seed points only. Important: seed points are independent and must NOT be connected."""
    canvas = image_bgr.copy()
    for i, (x, y) in enumerate(seeds, start=1):
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.circle(canvas, (int(x), int(y)), 9, (255, 255, 255), 1)
        cv2.putText(canvas, str(i), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return _draw_label_text(canvas, text)


def polygon_to_mask(shape: Tuple[int, int], points: Sequence[Tuple[float, float]], padding_px: int = 0) -> np.ndarray:
    """Create a binary mask from polygon points. Returns uint8 mask, 255 inside.

    For strict tracking, keep padding_px=0. Any positive padding deliberately expands the ROI.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not points:
        mask[:, :] = 255
        return mask
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    if padding_px and padding_px > 0:
        k = int(padding_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def select_polygon_interactive(
    image_gray: np.ndarray,
    window_name: str = "Select polygon ROI",
    max_points: int = 4,
) -> Optional[List[Tuple[float, float]]]:
    """Interactive polygon selector.

    Controls:
      Left click: add a vertex
      Z / Backspace: undo last vertex
      C: clear all vertices
      Enter: confirm
      Esc: cancel
    """
    if image_gray.ndim == 2:
        base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        base = image_gray.copy()
    points: List[Point] = []

    def redraw() -> None:
        closed = len(points) >= 3
        txt = (
            "ROI: left-click add point | Z/Backspace undo | C clear | Enter confirm | Esc cancel\n"
            f"Need {max_points} points. Current: {len(points)}"
        )
        cv2.imshow(window_name, _draw_points_and_polygon(base, points, closed=closed, text=txt))

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < max_points:
                points.append((int(x), int(y)))
                redraw()
            else:
                print(f"ROI 已达到最大点数 {max_points}；如需修改请按 Z 撤回或 C 清空。")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    print("ROI 选择：左键添加顶点；Z 或 Backspace 撤回；C 清空；Enter 确认；Esc 取消。")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(points) >= 3:
                cv2.destroyWindow(window_name)
                return [(float(x), float(y)) for x, y in points]
            print("ROI 至少需要 3 个点，建议点击 4 个点后确认。")
        elif key in (ord('z'), ord('Z'), 8):
            if points:
                points.pop()
                redraw()
        elif key in (ord('c'), ord('C')):
            points.clear()
            redraw()
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None


def select_multi_seeds_interactive(
    image_gray: np.ndarray,
    roi_points: Optional[Sequence[Tuple[float, float]]] = None,
    roi_mask: Optional[np.ndarray] = None,
    window_name: str = "Select seed particles",
    max_seeds: int = 100,
) -> List[Tuple[float, float]]:
    """Select multiple seed points interactively.

    Controls:
      Left click: add a seed point
      Z / Backspace: undo last seed point
      C: clear all seeds
      Enter: confirm
      Esc: cancel
    """
    if image_gray.ndim == 2:
        base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        base = image_gray.copy()

    # Light ROI tint for visual reference only. The mask itself remains a hard constraint later.
    if roi_mask is not None:
        color = np.zeros_like(base)
        color[:, :, 1] = 255
        tint = cv2.addWeighted(base, 0.78, color, 0.22, 0)
        base = np.where(roi_mask[:, :, None] > 0, tint, base).astype(np.uint8)
    if roi_points is not None and len(roi_points) >= 3:
        pts = np.array(roi_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(base, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    seeds: List[Point] = []

    def redraw() -> None:
        txt = (
            "Seeds: left-click add independent points | Z/Backspace undo | C clear | Enter start | Esc cancel\n"
            f"Selected seeds: {len(seeds)} / {max_seeds}. No lines are drawn between seeds."
        )
        canvas = _draw_seed_points_only(base, seeds, text=txt)
        cv2.imshow(window_name, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if roi_mask is not None:
                h, w = roi_mask.shape[:2]
                if not (0 <= x < w and 0 <= y < h and roi_mask[y, x] > 0):
                    print("该种子点位于 ROI 外，已忽略。")
                    return
            if len(seeds) < max_seeds:
                seeds.append((int(x), int(y)))
                redraw()
            else:
                print(f"已达到最大手动种子数 max_seeds={max_seeds}。")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    print("种子选择：左键依次选择多个颗粒；Z 或 Backspace 撤回；C 清空；Enter 开始追踪；Esc 取消。")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if seeds:
                cv2.destroyWindow(window_name)
                return [(float(x), float(y)) for x, y in seeds]
            print("至少需要选择 1 个种子颗粒。")
        elif key in (ord('z'), ord('Z'), 8):
            if seeds:
                seeds.pop()
                redraw()
        elif key in (ord('c'), ord('C')):
            seeds.clear()
            redraw()
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户取消种子点选择。")



def review_auto_seeds_interactive(
    image_frames: dict[int, np.ndarray],
    seeds: Sequence[object],
    roi_mask: Optional[np.ndarray] = None,
    window_name: str = "Review auto seeds",
    max_extra_seeds: int = 500,
) -> List[Tuple[int, float, float, str]]:
    """Interactively review auto-selected seed points.

    Returns a list of tuples: (seed_frame, x, z, source).

    Controls:
      N / Right arrow: next seed frame
      P / Left arrow: previous seed frame
      Left click: add a seed on the current frame
      Right click: delete nearest seed on the current frame
      Z / Backspace: undo last add/delete action
      C: clear seeds on current frame
      Enter: accept reviewed seeds
      Esc: cancel and keep original auto seeds

    This function is intentionally simple and robust: it does not connect points,
    and it only allows adding/removing seeds for the selected seed frames.
    """
    if not seeds:
        return []

    frame_ids = sorted(set(int(getattr(s, 'seed_frame')) for s in seeds))
    if not frame_ids:
        return []

    # Internal mutable records: [frame, x, z, source]
    records: List[List[object]] = []
    for s in seeds:
        records.append([int(getattr(s, 'seed_frame')), float(getattr(s, 'x')), float(getattr(s, 'z')), str(getattr(s, 'source', 'auto'))])
    original = [r.copy() for r in records]
    history: List[List[List[object]]] = []
    current_idx = 0

    def _make_base(frame_id: int) -> np.ndarray:
        img = image_frames[frame_id]
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        if roi_mask is not None:
            color = np.zeros_like(base)
            color[:, :, 1] = 255
            tint = cv2.addWeighted(base, 0.82, color, 0.18, 0)
            base = np.where(roi_mask[:, :, None] > 0, tint, base).astype(np.uint8)
        return base

    def _seeds_current(frame_id: int):
        return [(i, r) for i, r in enumerate(records) if int(r[0]) == int(frame_id)]

    def redraw() -> None:
        frame_id = frame_ids[current_idx]
        canvas = _make_base(frame_id)
        curr = _seeds_current(frame_id)
        for j, (global_i, r) in enumerate(curr, start=1):
            x, y = int(round(float(r[1]))), int(round(float(r[2])))
            color = (0, 255, 0) if str(r[3]).startswith('auto') else (0, 0, 255)
            cv2.circle(canvas, (x, y), 5, color, -1)
            cv2.circle(canvas, (x, y), 9, (255, 255, 255), 1)
            # local number only; small to reduce occlusion
            cv2.putText(canvas, str(j), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        txt = (
            f"Auto seed review: frame {frame_id} ({current_idx+1}/{len(frame_ids)}), seeds in frame={len(curr)}, total={len(records)}\n"
            "N/Right next | P/Left previous | Left add | Right delete nearest | Z undo | C clear current | Enter accept | Esc keep original"
        )
        cv2.imshow(window_name, _draw_label_text(canvas, txt))

    def save_history():
        history.append([r.copy() for r in records])
        if len(history) > 50:
            history.pop(0)

    def on_mouse(event, x, y, flags, param):
        nonlocal records
        frame_id = frame_ids[current_idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            if roi_mask is not None:
                h, w = roi_mask.shape[:2]
                if not (0 <= x < w and 0 <= y < h and roi_mask[y, x] > 0):
                    print("新增种子点位于 ROI 外，已忽略。")
                    return
            if len(records) >= max_extra_seeds:
                print(f"种子总数已达到上限 {max_extra_seeds}。")
                return
            save_history()
            records.append([int(frame_id), float(x), float(y), 'manual_added_after_auto_review'])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            curr = _seeds_current(frame_id)
            if not curr:
                return
            dists = [(math.hypot(float(r[1])-x, float(r[2])-y), idx) for idx, r in curr]
            dist, idx = min(dists, key=lambda v: v[0])
            if dist <= 30:
                save_history()
                records.pop(idx)
                redraw()
            else:
                print("右键位置附近没有种子点；删除忽略。")

    import math
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    print("自动种子确认：N/P 切换帧；左键加种子；右键删除最近种子；Enter 接受；Esc 保留原自动种子。")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            cv2.destroyWindow(window_name)
            return [(int(r[0]), float(r[1]), float(r[2]), str(r[3])) for r in records]
        elif key == 27:
            cv2.destroyWindow(window_name)
            return [(int(r[0]), float(r[1]), float(r[2]), str(r[3])) for r in original]
        elif key in (ord('n'), ord('N'), 83):
            current_idx = (current_idx + 1) % len(frame_ids)
            redraw()
        elif key in (ord('p'), ord('P'), 81):
            current_idx = (current_idx - 1) % len(frame_ids)
            redraw()
        elif key in (ord('z'), ord('Z'), 8):
            if history:
                records = history.pop()
                redraw()
        elif key in (ord('c'), ord('C')):
            frame_id = frame_ids[current_idx]
            save_history()
            records = [r for r in records if int(r[0]) != int(frame_id)]
            redraw()
