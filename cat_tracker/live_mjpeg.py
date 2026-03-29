"""Thread-safe latest-frame buffer for MJPEG browser preview."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class LiveMjpegHub:
    """Encode preview frames as JPEG for Flask multipart streaming."""

    def __init__(self, max_width: int = 800, jpeg_quality: int = 62, min_interval_s: float = 1.0 / 30.0) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._max_w = int(max_width)
        self._quality = int(jpeg_quality)
        self._min_interval_s = float(min_interval_s)
        self._last_push = 0.0

    def push_bgr(self, bgr: np.ndarray) -> None:
        if bgr is None or bgr.size == 0:
            return
        now = time.monotonic()
        if now - self._last_push < self._min_interval_s:
            return
        self._last_push = now
        h, w = bgr.shape[:2]
        frame = bgr
        if w > self._max_w:
            s = self._max_w / w
            frame = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            return
        blob = buf.tobytes()
        with self._lock:
            self._jpeg = blob
            self._seq += 1

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def get_frame(self) -> tuple[bytes | None, int]:
        """Latest JPEG and monotonic sequence (increments only when a new frame is encoded)."""
        with self._lock:
            return (self._jpeg, self._seq)
