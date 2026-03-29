"""Capture microphone to WAV during a clip (optional sounddevice dependency)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np


class MicSegmentRecorder:
    """Non-blocking mic capture; stop() writes a temporary WAV file."""

    def __init__(self, sample_rate: int = 44100, channels: int = 1) -> None:
        self.sample_rate = max(8000, int(sample_rate))
        self.channels = max(1, min(2, int(channels)))
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._sd = None
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> bool:
        try:
            import sounddevice as sd
        except ImportError:
            print(
                "[audio] sounddevice not installed — pip install sounddevice (PortAudio). "
                "Clips will be video-only.",
                flush=True,
            )
            return False
        self._sd = sd
        self._frames = []
        self._running = True

        def cb(indata, frames, t, status) -> None:  # noqa: ARG001
            if status:
                print(f"[audio] mic stream: {status}", flush=True)
            if self._running:
                with self._lock:
                    self._frames.append(indata.copy())

        device = None
        raw_dev = os.environ.get("CATTRACKER_MIC_DEVICE", "").strip()
        if raw_dev.isdigit():
            device = int(raw_dev)

        try:
            dev_i = device if device is not None else sd.default.device[0]
            info = sd.query_devices(dev_i, "input")
            print(
                f"[audio] microphone: {info.get('name', '?')} (device index {dev_i}; "
                f"set CATTRACKER_MIC_DEVICE=N to pick another)",
                flush=True,
            )
        except Exception:
            pass

        try:
            self._stream = sd.InputStream(
                device=device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
                callback=cb,
                blocksize=2048,
                latency="high",
            )
            self._stream.start()
        except Exception as e:
            print(f"[audio] could not open microphone: {e}", flush=True)
            self._stream = None
            self._running = False
            return False
        return True

    def stop(self) -> Path | None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._lock:
            chunks = self._frames
            self._frames = []

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        import tempfile
        import wave

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="cattracker_mic_")
        import os

        os.close(fd)
        path_p = Path(path)
        try:
            with wave.open(str(path_p), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio.astype(np.int16).tobytes())
        except OSError:
            path_p.unlink(missing_ok=True)
            return None
        return path_p
