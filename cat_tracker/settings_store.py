"""Persist tracker options for the web Settings panel (applied on next app start)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_FILENAME = "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "camera_mode": "hardware",
    "hardware_camera_index": 0,
    "ip_camera_url": "",
    "mirror_camera": True,
    "record_audio": True,
    "yolo_conf": 0.45,
}


def settings_path(pkg_root: Path) -> Path:
    return Path(pkg_root).resolve() / SETTINGS_FILENAME


def load_raw(pkg_root: Path) -> dict[str, Any]:
    p = settings_path(pkg_root)
    data = dict(DEFAULT_SETTINGS)
    if not p.is_file():
        return data
    try:
        disk = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(disk, dict):
            for k, v in disk.items():
                if k in DEFAULT_SETTINGS:
                    data[k] = v
    except (OSError, json.JSONDecodeError):
        pass
    return data


def save(pkg_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate, merge onto defaults, write JSON. Returns what was saved."""
    data = dict(DEFAULT_SETTINGS)
    mode = str(payload.get("camera_mode", "hardware")).strip().lower()
    data["camera_mode"] = "ip" if mode == "ip" else "hardware"
    try:
        data["hardware_camera_index"] = max(0, min(20, int(payload.get("hardware_camera_index", 0))))
    except (TypeError, ValueError):
        data["hardware_camera_index"] = 0
    url = str(payload.get("ip_camera_url", "") or "").strip()
    data["ip_camera_url"] = url[:2048]
    data["mirror_camera"] = bool(payload.get("mirror_camera", True))
    data["record_audio"] = bool(payload.get("record_audio", True))
    try:
        data["yolo_conf"] = float(payload.get("yolo_conf", 0.45))
        data["yolo_conf"] = max(0.05, min(0.95, data["yolo_conf"]))
    except (TypeError, ValueError):
        data["yolo_conf"] = 0.45

    pkg_root = Path(pkg_root).resolve()
    pkg_root.mkdir(parents=True, exist_ok=True)
    settings_path(pkg_root).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def argparse_defaults(pkg_root: Path) -> dict[str, Any]:
    """Map settings file → argparse dest names for parser.set_defaults()."""
    s = load_raw(pkg_root)
    out: dict[str, Any] = {}
    if s.get("camera_mode") == "ip" and (s.get("ip_camera_url") or "").strip():
        out["camera"] = s["ip_camera_url"].strip()
    else:
        out["camera"] = str(int(s.get("hardware_camera_index", 0)))
    out["no_mirror"] = not bool(s.get("mirror_camera", True))
    out["no_record_audio"] = not bool(s.get("record_audio", True))
    out["yolo_conf"] = float(s.get("yolo_conf", 0.45))
    return out


def to_api_dict(pkg_root: Path) -> dict[str, Any]:
    return load_raw(pkg_root)
