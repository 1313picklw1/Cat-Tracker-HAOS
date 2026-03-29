"""Start/stop paired video segments (zoom + full view) when cats are in view."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .cat_id import recording_slug


def _try_open_writer(
    out_dir: Path,
    filename_stem: str,
    fps: float,
    w: int,
    h: int,
) -> tuple[cv2.VideoWriter, Path] | None:
    """
    Prefer H.264-style fourcc first for Safari/Chrome playback.
    OpenCV codec support varies by OS.
    """
    candidates: list[tuple[str, str]] = [
        (".mp4", "avc1"),
        (".mp4", "H264"),
        (".mp4", "mp4v"),
        (".avi", "MJPG"),
        (".avi", "XVID"),
    ]
    for ext, cc in candidates:
        path = out_dir / f"{filename_stem}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, path
        writer.release()
    return None


class CatDualRecorder:
    """Writes `{slug}_{ts}_zoom.*` and `{slug}_{ts}_full.*`; optional mic + ffmpeg AAC mux."""

    def __init__(
        self,
        out_dir: Path,
        fps: float,
        frame_size: tuple[int, int],
        *,
        record_audio: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(1.0, float(fps))
        self.frame_w, self.frame_h = frame_size
        self._record_audio = bool(record_audio)
        self._writer_z: cv2.VideoWriter | None = None
        self._writer_f: cv2.VideoWriter | None = None
        self._path_z: Path | None = None
        self._path_f: Path | None = None
        self._slug = "cat"
        self._ts = ""
        self._open_fail_logged = False
        self._mic: object | None = None
        self._frame_count = 0
        self._segment_t0 = 0.0
        self._writer_fps_used = self.fps

    @property
    def active(self) -> bool:
        return self._writer_z is not None

    def start(
        self,
        name_slug: str | None = None,
        *,
        segment_fps: float | None = None,
    ) -> tuple[Path | None, Path | None]:
        self.stop()
        self._ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._slug = recording_slug((name_slug or "cat").strip() or "cat")
        fps_write = float(segment_fps) if segment_fps is not None else self.fps
        fps_write = max(4.0, min(90.0, fps_write))
        self._writer_fps_used = fps_write
        self._frame_count = 0
        self._segment_t0 = time.monotonic()
        w, h = self.frame_w, self.frame_h
        oz = _try_open_writer(self.out_dir, f"{self._slug}_{self._ts}_zoom", fps_write, w, h)
        of = _try_open_writer(self.out_dir, f"{self._slug}_{self._ts}_full", fps_write, w, h)
        if oz is None or of is None:
            if oz:
                oz[0].release()
            if of:
                of[0].release()
            if not self._open_fail_logged:
                print(
                    "Could not open dual VideoWriters (tried avc1/H264/mp4v MP4, MJPG AVI). "
                    "Install ffmpeg-backed OpenCV for best browser playback.",
                    flush=True,
                )
                self._open_fail_logged = True
            return None, None
        self._open_fail_logged = False
        self._writer_z, self._path_z = oz
        self._writer_f, self._path_f = of

        self._mic = None
        if self._record_audio:
            from .audio_recorder import MicSegmentRecorder

            mic = MicSegmentRecorder()
            if mic.start():
                self._mic = mic
                print("Recording audio (microphone) for this clip …", flush=True)

        print(
            f"Recording → {self._path_z.name} + {self._path_f.name} @ {fps_write:.1f} fps (metadata matches live loop)",
            flush=True,
        )
        return self._path_z, self._path_f

    def _resize_write(self, writer: cv2.VideoWriter, frame: np.ndarray) -> None:
        if frame.shape[1] != self.frame_w or frame.shape[0] != self.frame_h:
            frame = cv2.resize(frame, (self.frame_w, self.frame_h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

    def write(self, zoom_bgr: np.ndarray, full_bgr: np.ndarray) -> None:
        if self._writer_z is None or self._writer_f is None:
            return
        self._resize_write(self._writer_z, zoom_bgr)
        self._resize_write(self._writer_f, full_bgr)
        self._frame_count += 1

    def stop(self, rename_slug: str | None = None) -> tuple[Path | None, Path | None]:
        pz, pf = self._path_z, self._path_f
        if self._writer_z is not None:
            self._writer_z.release()
            self._writer_z = None
        if self._writer_f is not None:
            self._writer_f.release()
            self._writer_f = None
        self._path_z = self._path_f = None

        wav_path: Path | None = None
        if self._mic is not None:
            mic = self._mic
            self._mic = None
            try:
                wav_path = mic.stop()
            except Exception as e:
                print(f"[audio] mic stop: {e}", flush=True)

        if pz is None and pf is None:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)
            return None, None

        new_s = recording_slug(rename_slug.strip()) if rename_slug and rename_slug.strip() else None
        if new_s and new_s != self._slug and pz is not None and pf is not None:
            nz = pz.with_name(f"{new_s}_{self._ts}_zoom{pz.suffix}")
            nf = pf.with_name(f"{new_s}_{self._ts}_full{pf.suffix}")
            try:
                if nz != pz:
                    pz.rename(nz)
                    pz = nz
                if nf != pf:
                    pf.rename(nf)
                    pf = nf
            except OSError as e:
                print(f"Could not rename recording to {new_s}: {e}", flush=True)

        dur = max(1e-6, time.monotonic() - self._segment_t0)
        eff_fps = self._frame_count / dur
        if (
            pz is not None
            and pf is not None
            and self._frame_count >= 4
            and eff_fps > 1.0
            and abs(eff_fps - self._writer_fps_used) / self._writer_fps_used > 0.12
        ):
            from .ffmpeg_mux import retime_video_to_measured_fps

            print(
                f"[record] correcting playback speed: wrote @ {self._writer_fps_used:.1f} fps "
                f"but captured {self._frame_count} frames in {dur:.1f}s → {eff_fps:.1f} fps",
                flush=True,
            )
            if pz is not None and pz.is_file():
                nz = retime_video_to_measured_fps(pz, self._frame_count, dur)
                if nz is not None:
                    pz = nz
            if pf is not None and pf.is_file():
                nf = retime_video_to_measured_fps(pf, self._frame_count, dur)
                if nf is not None:
                    pf = nf

        if wav_path is not None and wav_path.is_file():
            try:
                if wav_path.stat().st_size < 2000:
                    print(
                        "[audio] microphone capture is nearly empty — on macOS allow Terminal or Python "
                        "in System Settings → Privacy & Security → Microphone.",
                        flush=True,
                    )
            except OSError:
                pass
            from .ffmpeg_mux import ffmpeg_available, mux_video_with_wav

            if ffmpeg_available():
                if pz is not None:
                    m = mux_video_with_wav(pz, wav_path)
                    if m is not None:
                        pz = m
                if pf is not None:
                    m = mux_video_with_wav(pf, wav_path)
                    if m is not None:
                        pf = m
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

        for p in (pz, pf):
            if p is None:
                continue
            try:
                b = p.stat().st_size
                print(f"Saved {p} ({b} bytes)", flush=True)
                if b < 256:
                    print("Warning: clip is very small — codec may not have written frames.", flush=True)
            except OSError:
                print(f"Saved {p}", flush=True)
        return pz, pf
