"""Run Gemini identify() on a worker thread so the camera loop stays smooth."""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .gemini_cats import GeminiCatIdentifier


class GeminiIdentifyWorker:
    """One in-flight request per track_id; results collected via pop_results()."""

    def __init__(self, gemini_id: GeminiCatIdentifier) -> None:
        self._gid = gemini_id
        self._q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
        self._lock = threading.Lock()
        self._inflight: set[int] = set()
        self._results: list[tuple[int, str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gemini-identify")
        self._thread.start()

    def request(self, track_id: int, crop_bgr: np.ndarray) -> None:
        if crop_bgr.size == 0:
            return
        with self._lock:
            if track_id in self._inflight:
                return
            self._inflight.add(track_id)
        self._q.put((track_id, crop_bgr.copy()))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                tid, crop = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                name, _ = self._gid.identify(crop)
            except Exception:
                name = "unknown"
            with self._lock:
                self._inflight.discard(tid)
                self._results.append((tid, name))

    def pop_results(self) -> list[tuple[int, str]]:
        with self._lock:
            r = self._results[:]
            self._results.clear()
            return r

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
