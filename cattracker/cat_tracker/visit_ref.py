"""Compare cat crops within a visit to detect a likely different individual."""

from __future__ import annotations

import cv2
import numpy as np

from .cat_id import boost_lowlight_bgr

_REF_SIDE = 72


def make_visit_ref_sig(bgr: np.ndarray) -> np.ndarray:
    """Small grayscale signature for comparison (cheap, robust enough for swap hints)."""
    if bgr.size == 0 or bgr.shape[0] < 4 or bgr.shape[1] < 4:
        return np.zeros((_REF_SIDE, _REF_SIDE), dtype=np.uint8)
    lit = boost_lowlight_bgr(bgr)
    g = cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (_REF_SIDE, _REF_SIDE), interpolation=cv2.INTER_AREA)


def visit_crop_dissimilarity(current_bgr: np.ndarray, ref_sig: np.ndarray) -> float:
    """
    Mean abs diff in [0, 1]. Same cat / pose often stays well below ~0.15;
    different cat or strong viewpoint change often exceeds ~0.18–0.22.
    """
    if ref_sig.size == 0 or current_bgr.size == 0:
        return 0.0
    cur = make_visit_ref_sig(current_bgr)
    a = cur.astype(np.float32) / 255.0
    b = ref_sig.astype(np.float32) / 255.0
    return float(np.mean(np.abs(a - b)))
