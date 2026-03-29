"""Per-camera live JSON for the sightings dashboard (multi-instance safe)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _safe_camera_id(raw: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", (raw or "main").strip())[:64]
    return s or "main"


class LiveStateWriter:
    """Writes sightings/live/<camera_id>.json periodically while the tracker runs."""

    def __init__(
        self,
        sightings_dir: Path,
        camera_id: str,
        label: str,
        viewer_url: str | None,
        every_n_frames: int = 6,
    ) -> None:
        self._dir = Path(sightings_dir) / "live"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cam_id = _safe_camera_id(camera_id)
        self._path = self._dir / f"{self._cam_id}.json"
        self._label = label.strip() or self._cam_id
        self._viewer_url = (viewer_url or "").strip() or None
        self._every = max(1, int(every_n_frames))

    def tick(self, frame_i: int, cat_in_frame: bool, cats_named: list[str]) -> None:
        if frame_i % self._every != 0:
            return
        payload = {
            "camera_id": self._cam_id,
            "label": self._label,
            "viewer_url": self._viewer_url,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cat_in_frame": bool(cat_in_frame),
            "cats_named": list(cats_named),
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def clear(self) -> None:
        """Remove live file on exit so dashboard does not show stale 'live'."""
        try:
            if self._path.is_file():
                self._path.unlink()
        except OSError:
            pass
