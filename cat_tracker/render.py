"""Side panel: zoomed cat crop or empty state."""

from __future__ import annotations

import cv2
import numpy as np

from .cat_id import crop_cat_bgr, preview_enhance_bgr
from .detector import TRACK_CROP_PAD_FRAC, CatBox

COLOR_BG = (18, 22, 28)
COLOR_ACCENT = (120, 200, 255)
COLOR_BOX = (80, 220, 120)


def render_cat_panel(
    frame_bgr: np.ndarray,
    box: CatBox | None,
    panel_w: int,
    panel_h: int,
    pad_frac: float = TRACK_CROP_PAD_FRAC,
    identity: str | None = None,
    lowlight_boost: bool = True,
) -> np.ndarray:
    """Right column: enlarged crop of cat, or placeholder."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = COLOR_BG

    if box is None:
        cv2.putText(
            panel,
            "No cat",
            (24, panel_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            COLOR_ACCENT,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Point camera at a cat",
            (24, panel_h // 2 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (140, 150, 170),
            1,
            cv2.LINE_AA,
        )
        if identity:
            cv2.putText(
                panel,
                identity[:48],
                (12, panel_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 230, 255),
                2,
                cv2.LINE_AA,
            )
        return panel

    fh, fw = frame_bgr.shape[:2]
    bw = box.x2 - box.x1
    bh = box.y2 - box.y1
    if bw < 4 or bh < 4:
        return panel

    px = bw * pad_frac
    py = bh * pad_frac
    x1 = int(max(0, box.x1 - px))
    y1 = int(max(0, box.y1 - py))
    x2 = int(min(fw, box.x2 + px))
    y2 = int(min(fh, box.y2 + py))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return panel

    if lowlight_boost:
        crop = preview_enhance_bgr(crop)

    ch, cw = crop.shape[:2]
    scale = min(panel_w / cw, panel_h / ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ox = (panel_w - nw) // 2
    oy = (panel_h - nh) // 2
    panel[oy : oy + nh, ox : ox + nw] = resized
    cv2.rectangle(panel, (ox, oy), (ox + nw, oy + nh), COLOR_BOX, 2, cv2.LINE_AA)
    if identity:
        cv2.putText(
            panel,
            identity[:48],
            (12, panel_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 240, 255),
            2,
            cv2.LINE_AA,
        )
    return panel


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: list[CatBox],
    smooth: CatBox | None,
    identity: str | None = None,
) -> None:
    for b in boxes:
        p1 = (int(b.x1), int(b.y1))
        p2 = (int(b.x2), int(b.y2))
        cv2.rectangle(frame_bgr, p1, p2, (60, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame_bgr,
            f"cat {b.conf:.2f}",
            (int(b.x1), max(18, int(b.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 180, 255),
            1,
            cv2.LINE_AA,
        )
    if smooth is not None:
        p1 = (int(smooth.x1), int(smooth.y1))
        p2 = (int(smooth.x2), int(smooth.y2))
        cv2.rectangle(frame_bgr, p1, p2, COLOR_BOX, 2, cv2.LINE_AA)
        y0 = max(18, int(smooth.y1) - 8)
        if identity:
            cv2.putText(
                frame_bgr,
                identity[:40],
                (int(smooth.x1), y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                COLOR_BOX,
                2,
                cv2.LINE_AA,
            )
        elif not boxes:
            cv2.putText(
                frame_bgr,
                f"track {smooth.conf:.2f}",
                (int(smooth.x1), y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_BOX,
                1,
                cv2.LINE_AA,
            )


TRACK_COLORS = (
    (80, 220, 120),
    (80, 160, 255),
    (255, 180, 80),
    (220, 90, 200),
    (180, 255, 120),
)


def draw_tracked_cats(
    frame_bgr: np.ndarray,
    raw_boxes: list[CatBox],
    tracked: list[tuple[CatBox, str | None]],
) -> None:
    """Faint raw detections + bold smoothed boxes with per-track labels."""
    for b in raw_boxes:
        p1 = (int(b.x1), int(b.y1))
        p2 = (int(b.x2), int(b.y2))
        cv2.rectangle(frame_bgr, p1, p2, (55, 90, 120), 1, cv2.LINE_AA)
    for i, (b, lab) in enumerate(tracked):
        col = TRACK_COLORS[i % len(TRACK_COLORS)]
        p1 = (int(b.x1), int(b.y1))
        p2 = (int(b.x2), int(b.y2))
        cv2.rectangle(frame_bgr, p1, p2, col, 2, cv2.LINE_AA)
        y0 = max(18, int(b.y1) - 8)
        if lab is None:
            text = f"… #{i + 1}"
        elif not str(lab).strip():
            text = f"cat {i + 1}"
        else:
            text = str(lab)[:40]
        cv2.putText(
            frame_bgr,
            text[:40],
            (int(b.x1), y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            2,
            cv2.LINE_AA,
        )


def render_multi_cat_panel(
    frame_bgr: np.ndarray,
    tracked: list[tuple[CatBox, str | None]],
    panel_w: int,
    panel_h: int,
    pad_frac: float = TRACK_CROP_PAD_FRAC,
    lowlight_boost: bool = True,
) -> np.ndarray:
    """Side column: grid of zoomed cats (up to 4), with optional name overlay."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = COLOR_BG
    n = len(tracked)
    if n == 0:
        cv2.putText(
            panel,
            "No cat",
            (24, panel_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            COLOR_ACCENT,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Point camera at a cat",
            (24, panel_h // 2 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (140, 150, 170),
            1,
            cv2.LINE_AA,
        )
        return panel

    cols = 1 if n == 1 else 2
    rows = min(2, (n + cols - 1) // cols)
    cell_w = panel_w // cols
    cell_h = panel_h // rows

    for idx, (box, lab) in enumerate(tracked[:4]):
        r, c = idx // cols, idx % cols
        x0, y0 = c * cell_w, r * cell_h
        crop = crop_cat_bgr(frame_bgr, box, pad_frac=pad_frac)
        sub = np.full((cell_h, cell_w, 3), COLOR_BG, dtype=np.uint8)
        if crop is not None and crop.size > 0:
            if lowlight_boost:
                crop = preview_enhance_bgr(crop)
            ch, cw = crop.shape[:2]
            scale = min((cell_w - 8) / cw, (cell_h - 28) / ch)
            nw = max(1, min(int(cw * scale), cell_w - 8))
            nh = max(1, min(int(ch * scale), cell_h - 28))
            if not crop.flags["C_CONTIGUOUS"]:
                crop = np.ascontiguousarray(crop)
            rz = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
            ox = x0 + max(0, (cell_w - nw) // 2)
            oy = y0 + 4 + max(0, (cell_h - 28 - nh) // 2)
            ry, rx = oy - y0, ox - x0
            if ry + nh <= cell_h and rx + nw <= cell_w and ry >= 0 and rx >= 0:
                sub[ry : ry + nh, rx : rx + nw] = rz
            col = TRACK_COLORS[idx % len(TRACK_COLORS)]
            cv2.rectangle(sub, (ox - x0, oy - y0), (ox - x0 + nw, oy - y0 + nh), col, 2, cv2.LINE_AA)
        cap = (lab[:20] + "…") if lab and len(lab) > 20 else (lab or "…")
        cv2.putText(
            sub,
            cap,
            (6, cell_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 235, 255),
            1,
            cv2.LINE_AA,
        )
        panel[y0 : y0 + cell_h, x0 : x0 + cell_w] = sub
    return panel
