"""Scale recording frames to output size and add Cat Tracker overlays."""

from __future__ import annotations

from pathlib import Path

import cv2

from .overlay import apply_recording_overlays


def build_zoom_recording_frame(
    panel_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    assets_dir: Path,
) -> np.ndarray:
    """Zoom / multi-crop panel scaled to output resolution."""
    canvas = cv2.resize(panel_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    apply_recording_overlays(canvas, assets_dir)
    return canvas


def build_full_recording_frame(
    camera_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    assets_dir: Path,
) -> np.ndarray:
    """Full camera view (boxes already drawn) scaled to output resolution."""
    canvas = cv2.resize(camera_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    apply_recording_overlays(canvas, assets_dir)
    return canvas
