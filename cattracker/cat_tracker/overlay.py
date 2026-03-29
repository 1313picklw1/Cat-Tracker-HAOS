"""Recording overlays: local timestamp (top-right) + Cat Tracker mark (bottom-right)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Extra see-through on the logo (0..1); PNG alpha is multiplied by this over the video.
WATERMARK_ICON_OPACITY = 0.42

_logo_cache: tuple[Path, np.ndarray] | None = None


def _load_logo(path: Path) -> np.ndarray | None:
    global _logo_cache
    if not path.is_file():
        return None
    if _logo_cache is not None and _logo_cache[0] == path:
        return _logo_cache[1]
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        img[:, :, 3] = 255
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = 255
    # Keep BGRA for alpha compositing over video (do not flatten onto white).
    _logo_cache = (path, img)
    return img


def draw_text_black_white_outline(
    bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(
                bgr,
                text,
                (org[0] + dx, org[1] + dy),
                font,
                font_scale,
                (255, 255, 255),
                thickness + 2,
                cv2.LINE_AA,
            )
    cv2.putText(bgr, text, org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def apply_recording_overlays(bgr: np.ndarray, assets_dir: Path) -> None:
    """Mutates frame in place."""
    h, w = bgr.shape[:2]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fs = 0.62
    (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    draw_text_black_white_outline(bgr, ts, (max(8, w - tw - 14), 26), fs)

    logo_path = assets_dir / "watermark_icon.png"
    logo = _load_logo(logo_path)
    label = "Cat Tracker"
    font_small = 0.52
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_small, 1)
    margin = 10
    logoh = 32
    row_h = max(logoh, lh + 2)
    y1 = h - margin - row_h
    if logo is not None:
        ih, iw = logo.shape[:2]
        sc = logoh / max(ih, 1)
        nw = max(1, int(iw * sc))
        nh = max(1, int(ih * sc))
        logo_r = cv2.resize(logo, (nw, nh), interpolation=cv2.INTER_AREA)
        block_w = nw + 8 + lw
        x0 = w - margin - block_w
        y_logo = y1 + (row_h - nh) // 2
        if x0 >= 0 and y_logo >= 0 and x0 + nw <= w and y_logo + nh <= h:
            roi = bgr[y_logo : y_logo + nh, x0 : x0 + nw].astype(np.float32)
            fg = logo_r.astype(np.float32)
            b = fg[:, :, :3]
            a = (fg[:, :, 3:4] / 255.0) * float(WATERMARK_ICON_OPACITY)
            a = np.clip(a, 0.0, 1.0)
            blended = roi * (1.0 - a) + b * a
            bgr[y_logo : y_logo + nh, x0 : x0 + nw] = blended.astype(np.uint8)
        tx = x0 + nw + 8
        ty = y1 + (row_h + lh) // 2
    else:
        tx = w - margin - lw
        ty = y1 + (row_h + lh) // 2

    if 0 <= ty <= h and 0 <= tx < w:
        cv2.putText(
            bgr,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_small,
            (235, 240, 250),
            1,
            cv2.LINE_AA,
        )
