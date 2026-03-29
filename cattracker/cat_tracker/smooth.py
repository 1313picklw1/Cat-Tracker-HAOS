"""EMA smoothing for bounding boxes (less jitter + short dropout tolerance)."""

from __future__ import annotations

from .detector import CatBox


class BoxSmoother:
    def __init__(self, alpha: float = 0.38, miss_before_clear: int = 5) -> None:
        self.alpha = alpha
        self.miss_before_clear = miss_before_clear
        self._x1 = self._y1 = self._x2 = self._y2 = 0.0
        self._conf = 0.0
        self._has = False
        self._miss = 0

    def reset(self) -> None:
        self._has = False
        self._miss = 0

    def update(self, box: CatBox | None) -> CatBox | None:
        if box is not None:
            self._miss = 0
            if not self._has:
                self._x1, self._y1 = box.x1, box.y1
                self._x2, self._y2 = box.x2, box.y2
                self._conf = box.conf
                self._has = True
                return CatBox(box.x1, box.y1, box.x2, box.y2, box.conf)
            a = self.alpha
            self._x1 = (1 - a) * self._x1 + a * box.x1
            self._y1 = (1 - a) * self._y1 + a * box.y1
            self._x2 = (1 - a) * self._x2 + a * box.x2
            self._y2 = (1 - a) * self._y2 + a * box.y2
            self._conf = box.conf
            return CatBox(self._x1, self._y1, self._x2, self._y2, self._conf)

        self._miss += 1
        if not self._has or self._miss > self.miss_before_clear:
            self._has = False
            return None
        return CatBox(self._x1, self._y1, self._x2, self._y2, self._conf)
