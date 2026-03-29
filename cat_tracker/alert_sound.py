"""Optional alert when cats appear (cross-platform, no extra Python deps)."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path


def play_alert_sound(path: Path) -> None:
    """Play a short sound file; failures are silent (tracker keeps running)."""
    p = Path(path)
    if not p.is_file():
        return
    fp = str(p.resolve())
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(
                ["afplay", fp],
                check=False,
                timeout=60,
                capture_output=True,
            )
        elif system == "Windows":
            import winsound

            winsound.PlaySound(fp, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            for cmd in (
                ["paplay", fp],
                ["aplay", "-q", fp],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", fp],
            ):
                try:
                    subprocess.run(
                        cmd,
                        check=False,
                        timeout=60,
                        capture_output=True,
                    )
                    return
                except FileNotFoundError:
                    continue
    except (subprocess.TimeoutExpired, OSError, Exception):
        print(f"[sound] could not play {fp}", file=sys.stderr, flush=True)


class CatPresentSoundAlert:
    """Play sound on 0→≥1 cat tracks edge, with cooldown after cats leave."""

    def __init__(self, path: Path | None, cooldown_s: float = 5.0) -> None:
        self.path = Path(path) if path is not None else None
        self.cooldown_s = max(0.5, float(cooldown_s))
        self._had_cats = False
        self._last_play = -1e9

    def tick(self, cats_now: bool) -> None:
        if self.path is None or not self.path.is_file():
            self._had_cats = cats_now
            return
        now = time.monotonic()
        rising = cats_now and not self._had_cats
        self._had_cats = cats_now
        if not rising:
            return
        if now - self._last_play < self.cooldown_s:
            return
        self._last_play = now
        play_alert_sound(self.path)
