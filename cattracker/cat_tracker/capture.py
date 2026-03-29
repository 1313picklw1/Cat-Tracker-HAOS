"""Webcam and IP / network stream capture (OpenCV)."""

from __future__ import annotations

import platform
import time

import cv2


def _looks_like_stream_url(source: str) -> bool:
    s = source.strip().lower()
    return "://" in s or s.startswith("rtsp:")


def _open_stream_url(url: str) -> cv2.VideoCapture | None:
    """RTSP, HTTP(S) MJPEG/H264, etc. Uses FFmpeg backend when available."""
    u = url.strip()
    cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(u)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    for attr_name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        a = getattr(cv2, attr_name, None)
        if a is not None:
            try:
                cap.set(a, 15000)
            except Exception:
                pass
            break
    # Warm up — IP cameras often need several reads
    for _ in range(80):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap
        time.sleep(0.05)
    return cap


def open_camera(
    source: str | int = "0",
    *,
    patient_first_index: bool = True,
) -> cv2.VideoCapture | None:
    """
    Local camera: integer index (0, 1, …). On macOS, index 0 with default
    patient mode still tries 0 → 1 → 2 like before.

    IP / network: pass a URL, e.g. rtsp://user:pass@192.168.1.50:554/stream
    or http://192.168.1.10/mjpg/video.mjpg
    """
    if isinstance(source, str) and _looks_like_stream_url(source):
        return _open_stream_url(source)

    try:
        if isinstance(source, str):
            s = source.strip()
            idx = int(s) if s.isdigit() else 0
        else:
            idx = int(source)
    except (ValueError, TypeError):
        idx = 0

    def try_open(index: int, api: int | None, *, patient: bool) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None
        attempts = 35 if patient else 8
        delay = 0.12 if patient else 0.02
        for _ in range(attempts):
            ok, _ = cap.read()
            if ok:
                return cap
            time.sleep(delay)
        cap.release()
        return None

    if platform.system() == "Darwin":
        avf = getattr(cv2, "CAP_AVFOUNDATION", None)
        if idx == 0 and patient_first_index:
            if avf is not None:
                for try_idx in range(3):
                    cap = try_open(try_idx, avf, patient=(try_idx == 0))
                    if cap is not None:
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        return cap
            for try_idx in range(3):
                cap = try_open(try_idx, None, patient=(try_idx == 0))
                if cap is not None:
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return cap
            return None
        if avf is not None:
            cap = try_open(idx, avf, patient=True)
            if cap is not None:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
        cap = try_open(idx, None, patient=True)
        if cap is not None:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    cap = try_open(idx, None, patient=True)
    if cap is not None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap
