"""Persist last-seen thumbs + clip paths for the web dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .cat_id import boost_lowlight_bgr, recording_slug


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SightingLogger:
    """
    Writes under base_dir:
      thumbs/<slug>_<timestamp>.jpg
      index.json  — per-cat last_seen, last_thumb, last_video (paths relative to package root)
    """

    def __init__(self, base_dir: Path, pkg_root: Path, thumb_boost: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.pkg_root = Path(pkg_root).resolve()
        self.thumbs_dir = self.base_dir / "thumbs"
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "index.json"
        self._thumb_boost = bool(thumb_boost)
        self._cats: dict[str, dict] = {}
        self._recent_clips: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self._index_path.is_file():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._cats = data.get("cats") or {}
            raw = data.get("recent_clips")
            self._recent_clips = raw if isinstance(raw, list) else []
        except (OSError, json.JSONDecodeError):
            self._cats = {}
            self._recent_clips = []

    def _save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cats": self._cats,
            "recent_clips": self._recent_clips[:40],
            "updated": _utc_iso(),
        }
        self._index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_identified_cat(self, display_name: str, crop_bgr: np.ndarray) -> None:
        """Call when Gemini locks a known name; saves a thumbnail."""
        if display_name not in ("unknown", "none", "?", "") and crop_bgr.size > 0:
            slug = recording_slug(display_name)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fname = f"{slug}_{ts}.jpg"
            path = self.thumbs_dir / fname
            thumb = boost_lowlight_bgr(crop_bgr) if self._thumb_boost else crop_bgr
            cv2.imwrite(str(path), thumb)
            rel_thumb = str(path.relative_to(self.pkg_root))
            self._cats[display_name] = {
                "display": display_name,
                "slug": slug,
                "last_seen": _utc_iso(),
                "last_thumb": rel_thumb.replace("\\", "/"),
                "last_video": self._cats.get(display_name, {}).get("last_video"),
            }
            self._save()

    def register_recording(self, recording_path: Path, display_name: str | None) -> None:
        """
        Call when a clip is finalized. Always appends to recent_clips for the dashboard.
        If display_name is a real cat name, also updates that cat's last_video.
        """
        rec = Path(recording_path).resolve()
        if not rec.is_file():
            return
        try:
            if rec.stat().st_size < 64:
                return
        except OSError:
            return
        try:
            rel_vid = str(rec.relative_to(self.pkg_root)).replace("\\", "/")
        except ValueError:
            rel_vid = str(recording_path).replace("\\", "/")
        ts = _utc_iso()
        label = display_name.strip() if isinstance(display_name, str) else None
        self._recent_clips.insert(0, {"path": rel_vid, "saved_at": ts, "label": label})
        self._recent_clips = self._recent_clips[:40]
        ok = bool(label) and label not in ("unknown", "none", "?", "")
        if ok:
            entry = self._cats.setdefault(
                label,
                {"display": label, "slug": recording_slug(label)},
            )
            entry["last_video"] = rel_vid
            entry["last_seen"] = ts
        self._save()

    def attach_last_video(self, display_name: str, recording_path: Path) -> None:
        """Backward-compatible alias for register_recording with a named cat."""
        self.register_recording(recording_path, display_name)
