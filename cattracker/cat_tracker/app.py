"""CatTracker main window: detect cats, draw boxes, crop panel, auto-record clips."""

from __future__ import annotations

import argparse
import collections
import logging
import os
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .alert_sound import CatPresentSoundAlert
from .capture import open_camera
from .cat_id import CatIdentifier, LabelSmoother, crop_cat_bgr, make_device, recording_slug
from .detector import TRACK_CROP_PAD_FRAC, CatBox, CatDetector
from .gemini_async import GeminiIdentifyWorker
from .gemini_cats import (
    GeminiCatIdentifier,
    gemini_api_key_hint,
    load_gemini_prompt_file,
    resolve_gemini_api_key,
)
from .motion_gate import MotionGate
from .record_compose import build_full_recording_frame, build_zoom_recording_frame
from .recorder import CatDualRecorder
from .render import draw_tracked_cats, render_multi_cat_panel
from .live_mjpeg import LiveMjpegHub
from .live_state import LiveStateWriter
from .sightings_log import SightingLogger
from .track_manager import CatTrack, MultiCatTracker
from .visit_ref import make_visit_ref_sig, visit_crop_dissimilarity

WINDOW_TITLE = "CatTracker - Q or Esc to quit"


def _run_sightings_web_server(
    pkg_root: Path, host: str, port: int, stream_hub: LiveMjpegHub | None
) -> None:
    """Flask dev server (background thread)."""
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    from .web_dashboard import create_app

    app = create_app(pkg_root, stream_hub=stream_hub)
    try:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    except OSError as e:
        print(f"Sightings dashboard could not bind {host}:{port} — {e}", file=sys.stderr, flush=True)


def _start_sightings_web_background(
    pkg_root: Path, host: str, port: int, stream_hub: LiveMjpegHub | None
) -> None:
    t = threading.Thread(
        target=_run_sightings_web_server,
        args=(pkg_root, host, port, stream_hub),
        daemon=True,
        name="cattracker-sightings-web",
    )
    t.start()


def _should_trigger_gemini_swap_recheck(
    *,
    args: argparse.Namespace,
    frame_i: int,
    gemini_id: GeminiCatIdentifier,
    per_visit: bool,
    gemini_visit_open: bool,
    gemini_session_label: str | None,
    gemini_pending_api: bool,
    cat_now: bool,
    crop: np.ndarray | None,
    gemini_visit_ref_sig: np.ndarray | None,
    cooldown_ok: bool,
    gem_rl_ok: bool,
) -> bool:
    """True if the current crop looks like a different cat than the locked visit ref."""
    if not per_visit or float(args.gemini_swap_threshold) <= 0:
        return False
    if not (
        gemini_visit_open
        and gemini_session_label
        and gemini_session_label not in ("unknown", "none")
        and any(
            gemini_session_label.strip().lower() == n.strip().lower() for n in gemini_id.names
        )
        and not gemini_pending_api
        and cat_now
        and crop is not None
        and gemini_visit_ref_sig is not None
    ):
        return False
    if frame_i % 5 != 0:
        return False
    diff = visit_crop_dissimilarity(crop, gemini_visit_ref_sig)
    if diff < float(args.gemini_swap_threshold):
        return False
    if not (cooldown_ok and gem_rl_ok):
        return False
    if args.verbose:
        print(f"[Gemini] likely different cat (crop Δ={diff:.3f}) — re-identifying …", flush=True)
    return True


def _clip_label_for_sighting(
    gemini_id: GeminiCatIdentifier | None,
    args: argparse.Namespace,
    gemini_session_label: str | None,
    id_smoother: LabelSmoother,
) -> str | None:
    """Display name to associate with the recording that just ended."""
    if gemini_id is not None and gemini_id.ready:
        if not args.gemini_refresh:
            lab = gemini_session_label
        else:
            m = id_smoother.mode()
            lab = m if m != "?" else None
        if lab and lab not in ("unknown", "none", "?"):
            return lab
        return None
    m = id_smoother.mode()
    return m if m != "?" else None


def _box_area(b: CatBox) -> float:
    return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)


def _tracks_by_area(tracks: list[CatTrack]) -> list[CatTrack]:
    return sorted(tracks, key=lambda t: _box_area(t.last_box) if t.last_box else 0.0, reverse=True)


def _compose_multi_clip_label(
    gemini_id: GeminiCatIdentifier | None,
    args: argparse.Namespace,
    tracks: list[CatTrack],
    gemini_session_label: str | None,
    id_smoother: LabelSmoother,
    ref_by_track: dict[int, LabelSmoother],
) -> str | None:
    if gemini_id is not None and gemini_id.ready and not args.gemini_refresh:
        labs = sorted(
            {
                t.gemini_label
                for t in tracks
                if t.gemini_label and t.gemini_label not in ("unknown", "none", "?")
            }
        )
        if not labs:
            return None
        return " & ".join(labs) if len(labs) > 1 else labs[0]
    if gemini_id is not None and gemini_id.ready:
        return _clip_label_for_sighting(gemini_id, args, gemini_session_label, id_smoother)
    if ref_by_track:
        names: list[str] = []
        for t in tracks:
            sm = ref_by_track.get(t.id)
            if sm is None:
                continue
            m = sm.mode()
            if m != "?" and m != "unknown":
                names.append(m)
        uniq = sorted(set(names))
        if not uniq:
            return None
        return " & ".join(uniq) if len(uniq) > 1 else uniq[0]
    return _clip_label_for_sighting(gemini_id, args, gemini_session_label, id_smoother)


def _recording_rename_slug(
    gemini_id: GeminiCatIdentifier | None,
    args: argparse.Namespace,
    tracks: list[CatTrack],
    gemini_session_label: str | None,
    id_smoother: LabelSmoother,
    ref_by_track: dict[int, LabelSmoother],
) -> str | None:
    lab = _compose_multi_clip_label(
        gemini_id, args, tracks, gemini_session_label, id_smoother, ref_by_track
    )
    return recording_slug(lab) if lab else None


def _draw_rec_indicator(bgr) -> None:
    cv2.circle(bgr, (28, 28), 10, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        bgr,
        "REC",
        (44, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 40, 255),
        2,
        cv2.LINE_AA,
    )


def run() -> None:
    _pkg_root = Path(__file__).resolve().parent.parent
    from .settings_store import argparse_defaults

    parser = argparse.ArgumentParser(description="Webcam cat detection (YOLOv8)")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics weights path or name")
    parser.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO letterbox size (smaller = faster; try 320)")
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.45,
        metavar="P",
        help="Min YOLO confidence for class cat (0.25 = default Ultralytics; higher = fewer false positives)",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Do not flip camera horizontally")
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Webcam index (0, 1, …) or stream URL: rtsp://…, http://…/video.mjpg, https://…",
    )
    parser.add_argument(
        "--alert-sound",
        type=Path,
        default=None,
        help="Play this file when cats first appear (0→≥1 track). macOS: WAV/MP3/M4A; Linux: WAV (paplay/aplay)",
    )
    parser.add_argument(
        "--alert-sound-cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between sounds when cats leave and come back",
    )
    parser.add_argument("--smooth", type=float, default=0.38, help="Box EMA 0..1 (higher = snappier)")
    parser.add_argument(
        "--miss",
        type=int,
        default=14,
        help="YOLO empty frames before dropping smoothed box (higher = tolerates look-away longer)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable automatic recording when a cat is visible",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Folder for MP4 clips (default: cattracker/recordings)",
    )
    parser.add_argument("--record-fps", type=float, default=30.0, help="FPS metadata for saved video (default 30)")
    parser.add_argument(
        "--record-use-camera-fps",
        action="store_true",
        help="Set recording FPS from the camera driver instead of --record-fps",
    )
    parser.add_argument(
        "--no-record-use-loop-fps",
        action="store_true",
        help="Do not infer recording FPS from the main loop; use --record-fps (or camera FPS) only",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        metavar="N",
        help="Target main-loop rate (sleep if faster). Use 0 to run as fast as possible.",
    )
    parser.add_argument("--record-w", type=int, default=1280, help="Recording frame width")
    parser.add_argument("--record-h", type=int, default=720, help="Recording frame height")
    parser.add_argument(
        "--record-end-miss",
        type=int,
        default=24,
        help="Consecutive frames with no tracked cat before a clip ends (higher = fewer tiny split MP4s)",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Reference photos: subfolder per cat name (default: cattracker/ref)",
    )
    parser.add_argument(
        "--no-id",
        action="store_true",
        help="Disable cat naming (no Gemini, no ref embeddings)",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Google Gemini vision + text descriptions instead of ref/ photos",
    )
    parser.add_argument(
        "--gemini-prompt",
        type=Path,
        default=None,
        help="Prompt file (default: cattracker/gemini_cats.txt). First line can be NAMES: A, B, C",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model id (see https://ai.google.dev/gemini-api/docs/models ; e.g. gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--gemini-every",
        type=int,
        default=20,
        help="With --gemini-no-motion: call Gemini every N frames on YOLO crop",
    )
    parser.add_argument(
        "--gemini-no-motion",
        action="store_true",
        help="Call Gemini on a fixed frame interval instead of when motion is detected",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.012,
        help="Motion score (0–1) to trigger Gemini; lower = more sensitive (try 0.008)",
    )
    parser.add_argument(
        "--gemini-cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between Gemini calls (motion mode)",
    )
    parser.add_argument(
        "--gemini-stale",
        type=float,
        default=14.0,
        help="If YOLO still sees a cat, refresh Gemini after this many seconds without a call",
    )
    parser.add_argument(
        "--gemini-max-side",
        type=int,
        default=None,
        metavar="PX",
        help="Max image side in pixels sent to Gemini (default ~768–1024); lower = smaller upload, often faster",
    )
    parser.add_argument(
        "--no-lowlight-boost",
        action="store_true",
        help="Disable CLAHE + shadow lift on crops for Gemini, zoom panel, and sighting thumbs",
    )
    parser.add_argument(
        "--gemini-refresh",
        action="store_true",
        help="Re-call Gemini on motion/stale (old behavior). Default: one API call when a cat appears, until they leave",
    )
    parser.add_argument(
        "--gemini-visit-end-miss",
        type=int,
        default=22,
        help="Consecutive no-detection frames before a Gemini 'visit' ends (avoids re-calling API after a 1-frame dropout)",
    )
    parser.add_argument(
        "--gemini-swap-threshold",
        type=float,
        default=0.0,
        metavar="D",
        help="Per-visit: if crop differs this much from the locked-ID crop (0–1), call Gemini again to fix wrong cat. 0=off (default)",
    )
    parser.add_argument(
        "--no-sightings-log",
        action="store_true",
        help="Do not write sightings/thumbs/index.json for the web dashboard",
    )
    parser.add_argument(
        "--sightings-dir",
        type=Path,
        default=None,
        help="Sightings folder (default: cattracker/sightings)",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Do not start the sightings dashboard (same as run_web.py) in the background",
    )
    parser.add_argument(
        "--web-host",
        default="0.0.0.0",
        help="Sightings dashboard bind (0.0.0.0 = all interfaces, reachable on LAN)",
    )
    parser.add_argument("--web-port", type=int, default=5050, help="Sightings dashboard port")
    parser.add_argument(
        "--web-live-fps",
        type=float,
        default=12.0,
        metavar="N",
        help="Max MJPEG frames per second for the dashboard (lower = less CPU and LAN traffic; default 12)",
    )
    parser.add_argument(
        "--web-live-max-width",
        type=int,
        default=720,
        metavar="PX",
        help="Scale live web preview to this width before JPEG encode (default 720)",
    )
    parser.add_argument(
        "--web-live-jpeg-quality",
        type=int,
        default=58,
        metavar="Q",
        help="JPEG quality 1–100 for live web preview (default 58; lower = smaller/faster)",
    )
    parser.add_argument(
        "--no-live-state",
        action="store_true",
        help="Do not write sightings/live/<camera_id>.json (dashboard 'live on camera' feature)",
    )
    parser.add_argument(
        "--camera-id",
        default="main",
        help="Unique id for this tracker when running multiple cameras (live dashboard)",
    )
    parser.add_argument(
        "--camera-label",
        default="Main camera",
        help="Human-readable camera name in the dashboard",
    )
    parser.add_argument(
        "--camera-viewer-url",
        default=None,
        help="Optional URL opened when you click 'Watch live' (e.g. another viewer or stream page)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No OpenCV window (servers / Home Assistant add-on); use web UI or MQTT. Stop with SIGTERM.",
    )
    parser.add_argument(
        "--mqtt-host",
        default=None,
        metavar="HOST",
        help="MQTT broker for Home Assistant (or set CATTRACKER_MQTT_HOST); enables discovery + state topics",
    )
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT port (default 1883)")
    parser.add_argument("--mqtt-user", default=None, help="MQTT username (or CATTRACKER_MQTT_USER)")
    parser.add_argument("--mqtt-password", default=None, help="MQTT password (or CATTRACKER_MQTT_PASSWORD)")
    parser.add_argument(
        "--mqtt-prefix",
        default="cattracker",
        help="MQTT topic prefix; full base is {prefix}/{camera_id}/…",
    )
    parser.add_argument(
        "--mqtt-discovery-prefix",
        default="homeassistant",
        help="Home Assistant MQTT discovery prefix (default homeassistant)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print motion score and each Gemini trigger to the terminal",
    )
    parser.add_argument(
        "--sim",
        type=float,
        default=0.42,
        help="(ref mode) Min cosine similarity to accept a named cat",
    )
    parser.add_argument(
        "--no-fur-enhance",
        action="store_true",
        help="(ref mode) Disable CLAHE on crops",
    )
    parser.add_argument(
        "--id-every",
        type=int,
        default=2,
        help="(ref mode) Run embedding every N frames",
    )
    parser.add_argument(
        "--id-device",
        default=None,
        help="(ref mode) torch device for ResNet",
    )
    parser.add_argument(
        "--no-record-audio",
        action="store_true",
        help="Do not record microphone or mux AAC audio into clips (ffmpeg required on PATH to mux)",
    )
    parser.set_defaults(**argparse_defaults(_pkg_root))
    args = parser.parse_args()
    if os.environ.get("CATTRACKER_HEADLESS", "").strip().lower() in ("1", "true", "yes"):
        args.headless = True

    mqtt_broker = (args.mqtt_host or "").strip() or None
    if not mqtt_broker:
        mqtt_broker = os.environ.get("CATTRACKER_MQTT_HOST", "").strip() or None
    if os.environ.get("CATTRACKER_MQTT_PORT", "").strip().isdigit():
        args.mqtt_port = int(os.environ["CATTRACKER_MQTT_PORT"])
    mqtt_user = (args.mqtt_user or os.environ.get("CATTRACKER_MQTT_USER") or "").strip() or None
    mqtt_password = (args.mqtt_password or os.environ.get("CATTRACKER_MQTT_PASSWORD") or "").strip() or None

    if (_pkg_root / "settings.json").is_file():
        print(
            f"Using {_pkg_root / 'settings.json'} as defaults (override with CLI flags).",
            flush=True,
        )

    cam_src = args.camera
    is_stream = "://" in cam_src or cam_src.strip().lower().startswith("rtsp:")
    if is_stream:
        print(f"Opening network camera: {cam_src[:80]}{'…' if len(cam_src) > 80 else ''}", flush=True)
    cap = open_camera(cam_src)
    if cap is None:
        print("Could not open camera (check --camera index or stream URL).")
        sys.exit(1)

    if not is_stream:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print("Loading YOLO (first run downloads yolov8n.pt) …")
    det = CatDetector(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=float(args.yolo_conf),
    )
    multi_tracker = MultiCatTracker(alpha=args.smooth, miss_before_clear=max(1, args.miss))

    pkg_root = Path(__file__).resolve().parent.parent
    assets_dir = Path(__file__).resolve().parent / "assets"
    ref_dir = args.ref_dir if args.ref_dir is not None else pkg_root / "ref"
    sightings_path = args.sightings_dir if args.sightings_dir is not None else pkg_root / "sightings"

    sighting_logger: SightingLogger | None = None
    if not args.no_sightings_log:
        sighting_logger = SightingLogger(
            sightings_path, pkg_root, thumb_boost=not args.no_lowlight_boost
        )

    live_writer: LiveStateWriter | None = None
    if not args.no_live_state:
        live_writer = LiveStateWriter(
            sightings_path,
            args.camera_id,
            args.camera_label,
            args.camera_viewer_url,
        )

    record_end_miss = max(1, int(args.record_end_miss))
    gemini_visit_end_miss = max(1, int(args.gemini_visit_end_miss))

    ref_id: CatIdentifier | None = None
    gemini_id: GeminiCatIdentifier | None = None
    motion_gate: MotionGate | None = None
    id_smoother = LabelSmoother(maxlen=11)
    last_sim = 0.0

    if args.no_id:
        print("Cat naming OFF (--no-id)")
    elif args.gemini:
        key, key_env = resolve_gemini_api_key()
        if not key:
            print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment for --gemini.")
            sys.exit(1)
        print(
            f"Gemini: key from {key_env} ({gemini_api_key_hint(key)}). "
            "Quota is per Google Cloud project — new keys created in the same project do not reset limits.",
            flush=True,
        )
        prompt_path = args.gemini_prompt if args.gemini_prompt is not None else pkg_root / "gemini_cats.txt"
        if not prompt_path.is_file():
            print(f"Gemini prompt file not found: {prompt_path}")
            sys.exit(1)
        names, body = load_gemini_prompt_file(prompt_path)
        if not names or not body.strip():
            print(f"No cat names or descriptions in {prompt_path} (need NAMES: line or 'Name is a ...' text).")
            sys.exit(1)
        try:
            gemini_id = GeminiCatIdentifier(
                api_key=key,
                names=names,
                description_block=body.strip(),
                model=args.gemini_model,
                image_max_side=args.gemini_max_side,
                lowlight_boost=not args.no_lowlight_boost,
            )
            if args.gemini_no_motion:
                print(
                    f"Gemini ID ON — {', '.join(names)} (model={args.gemini_model}, "
                    f"every {args.gemini_every} frames, YOLO crop)"
                    + (
                        "; refresh mode"
                        if args.gemini_refresh
                        else f"; once per visit (visit ends after {gemini_visit_end_miss} no-det frames)"
                    )
                )
            else:
                motion_gate = MotionGate()
                if args.gemini_refresh:
                    print(
                        f"Gemini ID ON — {', '.join(names)} (model={args.gemini_model}, "
                        f"motion ≥{args.motion_threshold}, cooldown {args.gemini_cooldown}s, "
                        f"stale {args.gemini_stale}s)"
                    )
                else:
                    print(
                        f"Gemini ID ON — {', '.join(names)} (model={args.gemini_model}, "
                        f"one call per visit; visit ends after {gemini_visit_end_miss} no-detection frames "
                        f"(not one blip) — use --gemini-refresh for old behavior)"
                    )
        except Exception as e:
            print(f"Failed to init Gemini: {e}")
            sys.exit(1)
    else:
        print("Loading identity model (ResNet50) & scanning ref/ …")
        tdev = make_device(args.id_device)
        ref_id = CatIdentifier(
            ref_dir,
            tdev,
            det,
            sim_threshold=float(args.sim),
            fur_enhance=not args.no_fur_enhance,
        )
        if ref_id.ready:
            fe = "CLAHE fur boost ON" if not args.no_fur_enhance else "fur boost OFF"
            print(f"Ref ID ON — {len(ref_id.names)} cat(s): {', '.join(ref_id.names)} ({fe})")
        else:
            print(f"Ref ID OFF — add subfolders with photos under {ref_dir}, or use --gemini")
            ref_id = None

    gem_worker: GeminiIdentifyWorker | None = None
    if gemini_id is not None and gemini_id.ready and not args.gemini_refresh:
        gem_worker = GeminiIdentifyWorker(gemini_id)
        print("Gemini runs in a background thread (one identify per cat track; UI stays smooth).", flush=True)

    record_dir = args.record_dir if args.record_dir is not None else pkg_root / "recordings"
    rec_size = (args.record_w, args.record_h)
    fps = float(args.record_fps)
    if args.record_use_camera_fps:
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        if cap_fps and cap_fps > 1:
            fps = float(cap_fps)

    target_frame_period = 0.0
    if float(args.target_fps) > 0:
        target_frame_period = 1.0 / float(args.target_fps)

    recorder: CatDualRecorder | None = None
    if not args.no_record:
        recorder = CatDualRecorder(
            record_dir,
            fps,
            rec_size,
            record_audio=not args.no_record_audio,
        )
        _au = " + microphone → AAC (ffmpeg)" if not args.no_record_audio else ""
        print(
            f"Auto-record ON → {record_dir} ({rec_size[0]}x{rec_size[1]} @ {fps:.1f} fps, "
            f"pairs *_zoom + *_full{_au}, end after {record_end_miss} no-cat frames)"
        )
    else:
        print("Auto-record OFF (--no-record)")

    if live_writer is not None:
        print(
            f"Live camera state → {sightings_path / 'live'}  "
            f"(id={args.camera_id!r}, label={args.camera_label!r})",
            flush=True,
        )

    if sighting_logger is not None:
        print(f"Sighting log → {sightings_path}", flush=True)

    stream_hub: LiveMjpegHub | None = None
    if not args.no_web:
        hub_interval = (1.0 / float(args.target_fps)) if float(args.target_fps) > 0 else 0.0
        web_cap = max(1.0, float(args.web_live_fps))
        web_interval = 1.0 / web_cap
        stream_interval = max(hub_interval, web_interval) if hub_interval > 0 else web_interval
        stream_hub = LiveMjpegHub(
            max_width=max(320, int(args.web_live_max_width)),
            jpeg_quality=max(30, min(95, int(args.web_live_jpeg_quality))),
            min_interval_s=stream_interval,
        )
        _start_sightings_web_background(pkg_root, args.web_host, int(args.web_port), stream_hub)
        lan_hint = (
            f"  On this machine: http://127.0.0.1:{int(args.web_port)}/"
            if str(args.web_host) in ("0.0.0.0", "::")
            else ""
        )
        print(
            f"Web dashboard + live preview → http://{args.web_host}:{int(args.web_port)}/"
            f"{lan_hint} (background; use --no-web to skip)",
            flush=True,
        )

    ha_mqtt = None
    if mqtt_broker:
        try:
            from .ha_mqtt import HaMqttPublisher

            ha_mqtt = HaMqttPublisher(
                broker=mqtt_broker,
                port=int(args.mqtt_port),
                user=mqtt_user,
                password=mqtt_password,
                camera_id=args.camera_id,
                camera_label=args.camera_label,
                topic_prefix=args.mqtt_prefix,
                discovery_prefix=args.mqtt_discovery_prefix,
            )
            if not ha_mqtt.start():
                ha_mqtt = None
        except Exception as e:
            print(f"[ha-mqtt] disabled: {e}", flush=True)
            ha_mqtt = None

    if not args.headless:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    sound_alert = CatPresentSoundAlert(
        args.alert_sound,
        cooldown_s=float(args.alert_sound_cooldown),
    )
    if args.alert_sound is not None:
        sp = Path(args.alert_sound)
        if sp.is_file():
            print(f"Alert sound on cat present → {sp}", flush=True)
        else:
            print(f"Alert sound file not found (ignored): {sp}", flush=True)

    frame_i = 0
    ref_every = max(1, int(args.id_every))
    gem_every = max(1, int(args.gemini_every))
    last_gemini_t = -1e9
    motion_score = 0.0
    gemini_kickoff_done = False
    gemini_pending_api = False
    gemini_session_label: str | None = None
    gemini_visit_open = False
    gem_visit_absent_streak = 0
    gemini_visit_ref_sig: np.ndarray | None = None
    rec_no_cat_streak = 0
    loop_fps_samples: collections.deque[float] = collections.deque(maxlen=80)
    ref_smooth_by_track: dict[int, LabelSmoother] = {}
    last_sim_ref: dict[int, float] = {}
    gem_logged_ids: set[int] = set()
    last_nonempty_tracks: list[CatTrack] = []

    def _ref_smooth(tid: int) -> LabelSmoother:
        sm = ref_smooth_by_track.get(tid)
        if sm is None:
            sm = LabelSmoother(maxlen=11)
            ref_smooth_by_track[tid] = sm
        return sm

    def _segment_record_fps() -> float | None:
        if args.no_record_use_loop_fps:
            return None
        if len(loop_fps_samples) < 8:
            return None
        med_dt = statistics.median(loop_fps_samples)
        if med_dt <= 1e-6:
            return None
        return max(4.0, min(60.0, 1.0 / med_dt))

    shutdown = threading.Event()

    def _request_shutdown(*_a: object) -> None:
        shutdown.set()

    if args.headless:
        signal.signal(signal.SIGTERM, _request_shutdown)
        try:
            signal.signal(signal.SIGINT, _request_shutdown)
        except ValueError:
            pass
        print("Headless mode — no GUI window; stop with SIGTERM or Ctrl+C.", flush=True)

    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            boxes = det.detect(frame)
            tracks = multi_tracker.update(boxes)
            ta = _tracks_by_area(tracks)
            primary = ta[0] if ta else None
            smooth_box = primary.last_box if primary else None
            cat_now = bool(tracks)
            tid_map = {t.id: t for t in tracks}
            sound_alert.tick(cat_now)

            if tracks:
                last_nonempty_tracks = list(tracks)
                rec_no_cat_streak = 0
            elif recorder is not None and recorder.active:
                rec_no_cat_streak += 1
            else:
                rec_no_cat_streak = 0

            if motion_gate is not None:
                motion_score = motion_gate.update(frame)

            if (
                recorder is not None
                and recorder.active
                and not tracks
                and rec_no_cat_streak >= record_end_miss
            ):
                rename = _recording_rename_slug(
                    gemini_id,
                    args,
                    last_nonempty_tracks,
                    gemini_session_label,
                    id_smoother,
                    ref_smooth_by_track,
                )
                pz, pf = recorder.stop(rename_slug=rename)
                if sighting_logger is not None:
                    lab = _compose_multi_clip_label(
                        gemini_id,
                        args,
                        last_nonempty_tracks,
                        gemini_session_label,
                        id_smoother,
                        ref_smooth_by_track,
                    )
                    if pz is not None:
                        sighting_logger.register_recording(pz, lab)
                    if pf is not None:
                        sighting_logger.register_recording(pf, lab)
                rec_no_cat_streak = 0

            identity_hud: str | None = None

            if gem_worker is not None:
                for tid, name in gem_worker.pop_results():
                    tr = tid_map.get(tid)
                    if tr is None:
                        continue
                    tr.gemini_label = "unknown" if name in ("none", "unknown") else name
                    if args.verbose:
                        print(f"[Gemini] track {tid} → {tr.gemini_label}", flush=True)
                    if (
                        name not in ("unknown", "none", "?")
                        and tid not in gem_logged_ids
                        and sighting_logger is not None
                        and tr.last_box is not None
                    ):
                        cr = crop_cat_bgr(frame, tr.last_box, pad_frac=TRACK_CROP_PAD_FRAC)
                        if cr is not None:
                            sighting_logger.log_identified_cat(name, cr)
                            gem_logged_ids.add(tid)
                for tid in list(gem_logged_ids):
                    if tid not in tid_map:
                        gem_logged_ids.discard(tid)

                for t in tracks:
                    if t.gemini_label is None and t.last_box is not None:
                        cr = crop_cat_bgr(frame, t.last_box, pad_frac=TRACK_CROP_PAD_FRAC)
                        if cr is not None:
                            gem_worker.request(t.id, cr)

                parts: list[str] = []
                for t in ta:
                    gl = t.gemini_label
                    if gl and gl not in ("unknown", "none"):
                        parts.append(gl)
                    elif gl == "unknown":
                        parts.append("?")
                if parts:
                    identity_hud = ", ".join(parts) + "  (gemini)"
                elif tracks:
                    identity_hud = "…  (gemini)"

            elif gemini_id is not None and gemini_id.ready:
                per_visit = not args.gemini_refresh
                crop = (
                    crop_cat_bgr(frame, smooth_box, pad_frac=TRACK_CROP_PAD_FRAC)
                    if smooth_box is not None
                    else None
                )

                if per_visit:
                    if smooth_box is not None:
                        gem_visit_absent_streak = 0
                        if not gemini_visit_open:
                            gemini_visit_open = True
                            gemini_pending_api = True
                            gemini_session_label = None
                            gemini_visit_ref_sig = None
                            id_smoother.clear()
                    else:
                        if gemini_visit_open:
                            gem_visit_absent_streak += 1
                            if gem_visit_absent_streak >= gemini_visit_end_miss:
                                gemini_visit_open = False
                                gemini_pending_api = False
                                gemini_session_label = None
                                gemini_visit_ref_sig = None
                                id_smoother.clear()
                                gem_visit_absent_streak = 0
                        else:
                            gem_visit_absent_streak = 0

                if args.gemini_no_motion:
                    if per_visit:
                        now = time.monotonic()
                        gem_rl_ok = now >= gemini_id.rate_limit_until
                        cooldown_ok = (now - last_gemini_t >= float(args.gemini_cooldown)) and gem_rl_ok
                        if _should_trigger_gemini_swap_recheck(
                            args=args,
                            frame_i=frame_i,
                            gemini_id=gemini_id,
                            per_visit=per_visit,
                            gemini_visit_open=gemini_visit_open,
                            gemini_session_label=gemini_session_label,
                            gemini_pending_api=gemini_pending_api,
                            cat_now=cat_now,
                            crop=crop,
                            gemini_visit_ref_sig=gemini_visit_ref_sig,
                            cooldown_ok=cooldown_ok,
                            gem_rl_ok=gem_rl_ok,
                        ):
                            gemini_pending_api = True
                        if (
                            gemini_pending_api
                            and crop is not None
                            and cooldown_ok
                            and gem_rl_ok
                            and frame_i % gem_every == 0
                        ):
                            if args.verbose:
                                print("[Gemini] per-visit (interval mode) …", flush=True)
                            name, last_sim = gemini_id.identify(crop)
                            last_gemini_t = now
                            if gemini_id.last_call_rate_limited:
                                pass
                            else:
                                gemini_pending_api = False
                                if name == "none":
                                    gemini_session_label = None
                                    gemini_visit_ref_sig = None
                                    id_smoother.clear()
                                elif name == "unknown":
                                    gemini_session_label = "unknown"
                                    gemini_visit_ref_sig = None
                                    id_smoother.push("unknown")
                                else:
                                    gemini_session_label = name
                                    gemini_visit_ref_sig = make_visit_ref_sig(crop)
                                    id_smoother.push(name)
                                    if sighting_logger is not None:
                                        sighting_logger.log_identified_cat(name, crop)
                    else:
                        if smooth_box is None:
                            id_smoother.clear()
                        elif frame_i % gem_every == 0:
                            if crop is not None:
                                name, last_sim = gemini_id.identify(crop)
                                if name == "unknown":
                                    id_smoother.push("unknown")
                                else:
                                    id_smoother.push(name)

                else:
                    now = time.monotonic()
                    gem_rl_ok = now >= gemini_id.rate_limit_until
                    cooldown_ok = (now - last_gemini_t >= float(args.gemini_cooldown)) and gem_rl_ok

                    if per_visit:
                        if _should_trigger_gemini_swap_recheck(
                            args=args,
                            frame_i=frame_i,
                            gemini_id=gemini_id,
                            per_visit=per_visit,
                            gemini_visit_open=gemini_visit_open,
                            gemini_session_label=gemini_session_label,
                            gemini_pending_api=gemini_pending_api,
                            cat_now=cat_now,
                            crop=crop,
                            gemini_visit_ref_sig=gemini_visit_ref_sig,
                            cooldown_ok=cooldown_ok,
                            gem_rl_ok=gem_rl_ok,
                        ):
                            gemini_pending_api = True
                        if (
                            gemini_pending_api
                            and cat_now
                            and crop is not None
                            and cooldown_ok
                            and gem_rl_ok
                        ):
                            if args.verbose:
                                print("[Gemini] per-visit identify_scene …", flush=True)
                            name, last_sim = gemini_id.identify_scene(frame, crop)
                            last_gemini_t = now
                            if gemini_id.last_call_rate_limited:
                                pass
                            else:
                                gemini_pending_api = False
                                if name == "none":
                                    gemini_session_label = None
                                    gemini_visit_ref_sig = None
                                    id_smoother.clear()
                                elif name == "unknown":
                                    gemini_session_label = "unknown"
                                    gemini_visit_ref_sig = None
                                    id_smoother.push("unknown")
                                else:
                                    gemini_session_label = name
                                    gemini_visit_ref_sig = make_visit_ref_sig(crop)
                                    id_smoother.push(name)
                                    if sighting_logger is not None:
                                        sighting_logger.log_identified_cat(name, crop)
                    else:
                        motion_on = motion_score >= float(args.motion_threshold)
                        stale = cat_now and (now - last_gemini_t >= float(args.gemini_stale))
                        kickoff = not gemini_kickoff_done and frame_i >= 1
                        if cooldown_ok and (motion_on or stale or kickoff):
                            if args.verbose:
                                print(
                                    f"[Gemini] trigger motion={motion_score:.4f} "
                                    f"(≥{args.motion_threshold}) motion_on={motion_on} "
                                    f"stale={stale} kickoff={kickoff}",
                                    flush=True,
                                )
                            name, last_sim = gemini_id.identify_scene(frame, crop)
                            if kickoff and not gemini_id.last_call_rate_limited:
                                gemini_kickoff_done = True
                            last_gemini_t = now
                            if args.verbose:
                                print(f"[Gemini] → {name}", flush=True)
                            if name == "none":
                                id_smoother.clear()
                            elif name != "unknown":
                                id_smoother.push(name)
                                if sighting_logger is not None and crop is not None:
                                    sighting_logger.log_identified_cat(name, crop)

                if args.gemini_refresh:
                    m = id_smoother.mode()
                    if m != "?":
                        identity_hud = f"{m}  (gemini)"
                else:
                    if gemini_session_label:
                        identity_hud = f"{gemini_session_label}  (gemini)"
                    elif gemini_visit_open and gemini_pending_api:
                        identity_hud = "…  (gemini)"

            elif ref_id is not None and ref_id.ready:
                alive = {t.id for t in tracks}
                for k in list(ref_smooth_by_track.keys()):
                    if k not in alive:
                        del ref_smooth_by_track[k]
                        last_sim_ref.pop(k, None)
                for t in tracks:
                    if t.last_box is None:
                        continue
                    if frame_i % ref_every == 0:
                        cr = crop_cat_bgr(frame, t.last_box, pad_frac=TRACK_CROP_PAD_FRAC)
                        if cr is not None:
                            name, sim = ref_id.match(cr)
                            _ref_smooth(t.id).push(name)
                            last_sim_ref[t.id] = sim
                hud_parts: list[str] = []
                for t in ta:
                    sm = ref_smooth_by_track.get(t.id)
                    if sm is None:
                        continue
                    m = sm.mode()
                    if m != "?":
                        hud_parts.append(f"{m} ({last_sim_ref.get(t.id, 0.0):.2f})")
                if hud_parts:
                    identity_hud = ", ".join(hud_parts)

            named_live: list[str] = []
            if gem_worker is not None:
                for t in tracks:
                    gl = t.gemini_label
                    if gl and gl not in ("unknown", "none"):
                        named_live.append(gl)
                named_live = sorted(set(named_live))
            elif gemini_id is not None and gemini_id.ready and smooth_box is not None:
                if not args.gemini_refresh:
                    gl = gemini_session_label
                    if gl and gl not in ("unknown", "none"):
                        named_live = [gl]
                else:
                    m = id_smoother.mode()
                    if m != "?":
                        named_live = [m]
            elif ref_id is not None and ref_id.ready:
                for t in tracks:
                    sm = ref_smooth_by_track.get(t.id)
                    if sm is None:
                        continue
                    m = sm.mode()
                    if m != "?" and m != "unknown":
                        named_live.append(m)
                named_live = sorted(set(named_live))

            if live_writer is not None:
                live_writer.tick(frame_i, bool(tracks), named_live)
            if ha_mqtt is not None:
                ha_mqtt.publish(bool(tracks), named_live)

            tracked_pairs: list[tuple[CatBox, str | None]] = []
            for t in ta:
                if t.last_box is None:
                    continue
                lbl: str | None = None
                if gem_worker is not None:
                    if t.gemini_label is None:
                        lbl = None
                    elif t.gemini_label == "unknown":
                        lbl = "?"
                    else:
                        lbl = t.gemini_label
                elif gemini_id is not None and gemini_id.ready:
                    if primary is not None and t.id == primary.id:
                        gl = gemini_session_label
                        if gl in ("unknown",):
                            lbl = "?"
                        elif gl in (None, "none"):
                            lbl = None
                        else:
                            lbl = gl
                elif ref_id is not None and ref_id.ready:
                    sm = ref_smooth_by_track.get(t.id)
                    m = sm.mode() if sm else "?"
                    lbl = None if m in ("?", "unknown") else m
                tracked_pairs.append((t.last_box, lbl))

            panel = render_multi_cat_panel(
                frame,
                tracked_pairs,
                w,
                h,
                pad_frac=TRACK_CROP_PAD_FRAC,
                lowlight_boost=not args.no_lowlight_boost,
            )
            draw_tracked_cats(frame, boxes, tracked_pairs)
            preview = cv2.hconcat([frame, panel])

            if recorder is not None:
                if tracks:
                    if not recorder.active:
                        recorder.start("cat", segment_fps=_segment_record_fps())
                if recorder.active:
                    frame_rec = frame.copy()
                    draw_tracked_cats(frame_rec, boxes, tracked_pairs)
                    zoom_bgr = build_zoom_recording_frame(
                        panel, rec_size[0], rec_size[1], assets_dir
                    )
                    full_bgr = build_full_recording_frame(
                        frame_rec, rec_size[0], rec_size[1], assets_dir
                    )
                    recorder.write(zoom_bgr, full_bgr)

            if recorder is not None and recorder.active:
                _draw_rec_indicator(preview)

            if stream_hub is not None:
                stream_hub.push_bgr(preview)

            if not args.headless:
                cv2.imshow(WINDOW_TITLE, preview)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            else:
                if shutdown.is_set():
                    break
                time.sleep(0.002)
            frame_i += 1
            if target_frame_period > 0:
                slack = target_frame_period - (time.perf_counter() - t_loop)
                if slack > 0:
                    time.sleep(slack)
            loop_fps_samples.append(time.perf_counter() - t_loop)
    finally:
        if recorder is not None and recorder.active:
            rename = _recording_rename_slug(
                gemini_id,
                args,
                last_nonempty_tracks,
                gemini_session_label,
                id_smoother,
                ref_smooth_by_track,
            )
            pz, pf = recorder.stop(rename_slug=rename)
            if sighting_logger is not None:
                lab = _compose_multi_clip_label(
                    gemini_id,
                    args,
                    last_nonempty_tracks,
                    gemini_session_label,
                    id_smoother,
                    ref_smooth_by_track,
                )
                if pz is not None:
                    sighting_logger.register_recording(pz, lab)
                if pf is not None:
                    sighting_logger.register_recording(pf, lab)
        if gem_worker is not None:
            gem_worker.close()
        if live_writer is not None:
            live_writer.clear()
        if ha_mqtt is not None:
            ha_mqtt.stop()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
