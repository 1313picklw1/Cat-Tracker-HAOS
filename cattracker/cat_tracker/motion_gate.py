"""Cheap frame differencing to detect movement (triggers Gemini)."""

from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, scale: float = 0.28, blur_ksize: int = 5, diff_threshold: int = 16) -> None:
        self.scale = scale
        self.blur_ksize = max(3, blur_ksize | 1)
        self.diff_threshold = diff_threshold
        self._prev: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> float:
        """
        Returns a motion score roughly in [0, 1]. Higher = more change vs previous frame.
        """
        h, w = frame_bgr.shape[:2]
        sw = max(32, int(w * self.scale))
        sh = max(24, int(h * self.scale))
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        if self._prev is None or self._prev.shape != gray.shape:
            self._prev = gray.copy()
            return 0.0

        d = cv2.absdiff(self._prev, gray)
        self._prev = gray.copy()
        _, mask = cv2.threshold(d, self.diff_threshold, 255, cv2.THRESH_BINARY)
        frac = float(np.mean(mask > 0))
        mean_norm = float(np.mean(d)) / 255.0
        # Webcams often look “static”; weight mean diff so AE noise / small moves still register
        score = min(1.0, 0.55 * frac + 4.5 * mean_norm)
        return score
