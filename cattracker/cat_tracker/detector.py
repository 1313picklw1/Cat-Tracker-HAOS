"""YOLOv8 — COCO class `cat` only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Shared by crop, side-panel zoom, and Gemini so the panel matches what we send to the API.
TRACK_CROP_PAD_FRAC = 0.22


@dataclass
class CatBox:
    """xyxy in pixel coords, confidence 0..1."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


def _iou_xyxy(a: CatBox, b: CatBox) -> float:
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


def _nms_cat_boxes(boxes: list[CatBox], iou_thresh: float = 0.5) -> list[CatBox]:
    """
    Drop duplicate/overlapping detections (YOLO often emits two boxes on one cat).
    Keeps highest-confidence box when IoU is high; real separate cats rarely exceed ~0.35 IoU.
    """
    if len(boxes) <= 1:
        return boxes
    boxes = sorted(boxes, key=lambda b: b.conf, reverse=True)
    kept: list[CatBox] = []
    for b in boxes:
        if all(_iou_xyxy(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


class CatDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str | None = None,
        imgsz: int = 416,
        conf: float = 0.45,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._conf = float(max(0.05, min(0.99, conf)))
        cat_id = None
        for k, v in self._model.names.items():
            if v == "cat":
                cat_id = int(k)
                break
        self._cat_id = 15 if cat_id is None else cat_id

        if device is None:
            try:
                import torch

                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"
        self._device = device
        # FP16 on MPS is uneven across torch/ultralytics builds; CUDA is the reliable win.
        self._half = isinstance(device, str) and device.lower().startswith("cuda")

    def detect(self, frame_bgr: np.ndarray) -> list[CatBox]:
        r = self._model.predict(
            frame_bgr,
            classes=[self._cat_id],
            conf=self._conf,
            verbose=False,
            imgsz=self._imgsz,
            device=self._device,
            half=self._half,
            max_det=5,
        )[0]
        out: list[CatBox] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            out.append(CatBox(float(x1), float(y1), float(x2), float(y2), float(conf[i])))
        out.sort(key=lambda b: b.conf, reverse=True)
        return _nms_cat_boxes(out, iou_thresh=0.5)
