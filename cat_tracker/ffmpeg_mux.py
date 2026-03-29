"""Mux microphone WAV into a video file using ffmpeg (must be on PATH)."""

from __future__ import annotations

import secrets
import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def retime_video_to_measured_fps(
    video: Path,
    frame_count: int,
    duration_sec: float,
) -> Path | None:
    """
    Re-encode video to a CFR stream at frame_count/duration so players don't play too fast.
    Used when OpenCV's declared fps drifted from real capture rate.
    """
    video = Path(video)
    if not video.is_file() or not ffmpeg_available():
        return None
    if frame_count < 2 or duration_sec < 0.05:
        return None
    out_fps = max(4.0, min(60.0, frame_count / duration_sec))
    tmp = video.parent / f"{video.stem}.__retime_{secrets.token_hex(4)}__.mp4"
    final_mp4 = video.with_suffix(".mp4")
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"fps={out_fps},setpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ],
            capture_output=True,
            timeout=600,
            text=True,
        )
        if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size < 64:
            if r.stderr:
                print(f"[record] ffmpeg retime: {r.stderr[-400:]}", flush=True)
            tmp.unlink(missing_ok=True)
            return None
        if video.exists() and video.resolve() != tmp.resolve():
            video.unlink()
        tmp.rename(final_mp4)
        return final_mp4
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[record] retime failed: {e}", flush=True)
        tmp.unlink(missing_ok=True)
        return None


def mux_video_with_wav(video: Path, wav: Path) -> Path | None:
    """
    Replace video with an MP4 that contains the same video stream (re-encode if needed)
    + AAC audio from wav. Returns final path, or None on failure (original video kept).
    """
    video = Path(video)
    wav = Path(wav)
    if not video.is_file() or not wav.is_file():
        return None
    if not ffmpeg_available():
        print("[audio] ffmpeg not on PATH — install ffmpeg to mux mic into clips.", flush=True)
        return None

    final_mp4 = video.with_suffix(".mp4")
    tmp = video.parent / f"{video.stem}.__mux_{secrets.token_hex(4)}__.mp4"

    def _run(cmd: list[str]) -> bool:
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                timeout=600,
                text=True,
            )
            ok = r.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 64
            if not ok and r.stderr:
                print(f"[audio] ffmpeg: {r.stderr[-500:]}", flush=True)
            return ok
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as e:
            print(f"[audio] ffmpeg run: {e}", flush=True)
            return False

    # Try stream copy first (works for H.264 in MP4)
    ok = _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(tmp),
        ]
    )
    if not ok:
        tmp.unlink(missing_ok=True)
        ok = _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-i",
                str(wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                str(tmp),
            ]
        )

    if not ok:
        tmp.unlink(missing_ok=True)
        print(
            f"[audio] ffmpeg mux failed for {video.name} (install ffmpeg; on Mac allow mic for Python in "
            f"System Settings → Privacy & Security → Microphone)",
            flush=True,
        )
        return None

    try:
        if video.exists() and video.resolve() != tmp.resolve():
            video.unlink()
        tmp.rename(final_mp4)
    except OSError as e:
        print(f"[audio] could not replace video with muxed file: {e}", flush=True)
        tmp.unlink(missing_ok=True)
        return None
    return final_mp4
