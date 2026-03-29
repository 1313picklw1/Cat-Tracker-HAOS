"""Multi-cat tracking: match YOLO boxes to persistent tracks (IoU + per-track smoothing)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import CatBox
from .smooth import BoxSmoother


def iou_xyxy(a: CatBox, b: CatBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    ba = max(1.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _merge_duplicate_tracks(active: list[CatTrack], iou_thresh: float = 0.6) -> list[CatTrack]:
    """
    Two smoothed boxes on the same real cat (YOLO double-hit or jitter) become two
    CatTracks with separate Gemini state — one may read "Peaches" and the other "???".
    Merge overlapping tracks: keep the older id, copy a name onto it if missing.
    """
    if len(active) <= 1:
        return active
    ordered = sorted(active, key=lambda t: t.id)
    remove: set[int] = set()
    for i, a in enumerate(ordered):
        if a.id in remove or a.last_box is None:
            continue
        for b in ordered[i + 1 :]:
            if b.id in remove or b.last_box is None:
                continue
            if iou_xyxy(a.last_box, b.last_box) < iou_thresh:
                continue
            if b.gemini_label and b.gemini_label not in ("unknown", "none"):
                if not a.gemini_label or a.gemini_label in ("unknown", "none"):
                    a.gemini_label = b.gemini_label
            remove.add(b.id)
    return [t for t in ordered if t.id not in remove]


@dataclass
class CatTrack:
    id: int
    smoother: BoxSmoother
    last_box: CatBox | None = None
    gemini_label: str | None = None


@dataclass
class MultiCatTracker:
    """Greedy IoU matching each frame; one BoxSmoother per track."""

    alpha: float = 0.38
    miss_before_clear: int = 14
    iou_match_min: float = 0.12
    _next_id: int = 1
    _tracks: dict[int, CatTrack] = field(default_factory=dict)

    def update(self, detections: list[CatBox]) -> list[CatTrack]:
        dets = list(detections)
        track_ids = list(self._tracks.keys())

        pairs: list[tuple[float, int, int]] = []
        for tid in track_ids:
            tr = self._tracks[tid]
            pred = tr.last_box
            if pred is None:
                continue
            for j, d in enumerate(dets):
                pairs.append((iou_xyxy(pred, d), tid, j))
        pairs.sort(key=lambda x: -x[0])

        assigned_t: set[int] = set()
        assigned_j: set[int] = set()
        det_for_tid: dict[int, CatBox | None] = {tid: None for tid in track_ids}

        for score, tid, j in pairs:
            if score < self.iou_match_min:
                break
            if tid in assigned_t or j in assigned_j:
                continue
            assigned_t.add(tid)
            assigned_j.add(j)
            det_for_tid[tid] = dets[j]

        new_tracks: dict[int, CatTrack] = {}
        active: list[CatTrack] = []

        for tid in track_ids:
            tr = self._tracks[tid]
            matched = det_for_tid.get(tid)
            out = tr.smoother.update(matched)
            if out is None:
                continue
            tr.last_box = out
            new_tracks[tid] = tr
            active.append(tr)

        for j, d in enumerate(dets):
            if j in assigned_j:
                continue
            tid = self._next_id
            self._next_id += 1
            sm = BoxSmoother(alpha=self.alpha, miss_before_clear=self.miss_before_clear)
            out = sm.update(d)
            if out is None:
                continue
            tr = CatTrack(id=tid, smoother=sm, last_box=out, gemini_label=None)
            new_tracks[tid] = tr
            active.append(tr)

        active.sort(key=lambda t: t.id)
        active = _merge_duplicate_tracks(active, iou_thresh=0.6)
        self._tracks = {t.id: t for t in active}
        return active

    def any_cat(self) -> bool:
        return bool(self._tracks)
