"""Tiny Flask UI to browse last sightings per cat (thumbs + clips + live cameras)."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file

from .settings_store import save, to_api_dict

from .live_mjpeg import LiveMjpegHub

_LIVE_STALE_SEC = 14.0

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CatTracker sightings</title>
  <style>
    :root { --bg:#0f1218; --card:#1a202c; --text:#e7ecf3; --accent:#7dd3fc; --live:#4ade80; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
           margin: 0; padding: 1.5rem; line-height: 1.45; }
    h1 { font-weight: 600; font-size: 1.25rem; margin: 0 0 1rem; }
    h2.section { font-size: 1rem; font-weight: 600; margin: 0 0 .5rem; color: #c4d4f5; }
    .live-panel { margin-bottom: 1.25rem; }
    .live-panel img { max-width: 100%; width: min(920px, 100%); border-radius: 10px; border: 1px solid #374151;
      display: block; background: #111; min-height: 120px; }
    .recent-row { display: flex; gap: .75rem; overflow-x: auto; padding-bottom: .35rem; margin-bottom: .25rem; }
    .recent-cell { flex: 0 0 auto; max-width: 220px; }
    .recent-cell video { max-height: 140px; border-radius: 8px; background: #000; width: 100%; }
    .recent-cap { font-size: .72rem; opacity: .8; margin-top: .25rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; }
    .card { background: var(--card); border-radius: 10px; overflow: hidden; cursor: pointer;
            border: 1px solid #2d3748; transition: border-color .15s; position: relative; }
    .card:hover { border-color: var(--accent); }
    .card img { width: 100%; aspect-ratio: 4/3; object-fit: cover; display: block; background:#111; }
    .card .meta { padding: .6rem .75rem; font-size: .85rem; }
    .card .name { font-weight: 600; color: var(--accent); }
    .card .time { opacity: .75; font-size: .78rem; margin-top: .2rem; }
    .live-pill { position: absolute; top: 8px; right: 8px; background: var(--live); color: #052e16;
      font-size: .68rem; font-weight: 700; padding: .2rem .45rem; border-radius: 6px; }
    .empty { opacity: .6; padding: 2rem; text-align: center; }
    dialog { border: none; border-radius: 12px; padding: 0; max-width: min(92vw, 720px);
             background: var(--card); color: var(--text); }
    dialog::backdrop { background: rgba(0,0,0,.65); }
    .dlg-inner { padding: 1rem 1.25rem 1.25rem; }
    .dlg-inner h2 { margin: 0 0 .5rem; font-size: 1.1rem; }
    .dlg-inner video { width: 100%; border-radius: 8px; margin-top: .75rem; background:#000; }
    .dlg-inner .no-vid { opacity: .6; font-size: .9rem; margin-top: .75rem; }
    .live-block { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #374151; }
    .live-block h3 { font-size: .95rem; margin: 0 0 .5rem; color: #a5b4fc; }
    .cam-row { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .35rem 0; font-size: .88rem; }
    .cam-row a, .cam-row button { color: var(--accent); text-decoration: none; background: #374151;
      border: none; border-radius: 6px; padding: .35rem .65rem; cursor: pointer; font: inherit; color: #fff; }
    .cam-row a:hover, .cam-row button:hover { background: #4b5563; }
    .muted { opacity: .65; font-size: .82rem; }
    button.close { float: right; background: #374151; color: #fff; border: none;
                   border-radius: 6px; padding: .35rem .65rem; cursor: pointer; }
    .toolbar { display: flex; gap: .5rem; align-items: center; margin: -.25rem 0 1rem; flex-wrap: wrap; }
    .toolbar button { background: #374151; color: #fff; border: none; border-radius: 8px;
      padding: .45rem .85rem; cursor: pointer; font: inherit; }
    .toolbar button:hover { background: #4b5563; }
    #settings-dlg { max-width: min(94vw, 480px); }
    .settings-form label { display: block; font-size: .82rem; margin: .75rem 0 .25rem; color: #a8b8d8; }
    .settings-form input[type=text], .settings-form input[type=number], .settings-form select {
      width: 100%; padding: .45rem .55rem; border-radius: 8px; border: 1px solid #374151;
      background: #111827; color: var(--text); font: inherit; box-sizing: border-box; }
    .settings-form .row { margin: .5rem 0; }
    .settings-form .hint { font-size: .76rem; opacity: .75; margin-top: .35rem; }
    .settings-actions { display: flex; gap: .5rem; margin-top: 1.1rem; flex-wrap: wrap; }
    .settings-actions button { background: #2563eb; color: #fff; border: none; border-radius: 8px;
      padding: .5rem 1rem; cursor: pointer; font: inherit; }
    .settings-actions button.secondary { background: #374151; }
  </style>
</head>
<body>
  <h1>Cat sightings</h1>
  <div class="toolbar">
    <button type="button" id="btn-settings">Camera & recording settings</button>
  </div>
  <div id="live-panel" class="live-panel" style="display:none">
    <h2 class="section">Live preview</h2>
    <img id="live-mjpeg" src="" alt="Live camera" />
    <p class="muted" style="margin:.4rem 0 0;font-size:.82rem">Same view as the OpenCV window (boxes + side panel). Only while <code>run.py</code> is running on this machine.</p>
  </div>
  <div id="recent-wrap" style="display:none;margin-bottom:1.25rem">
    <h2 class="section">Recent clips</h2>
    <div id="recent-clips" class="recent-row"></div>
  </div>
  <div id="root" class="grid"></div>
  <dialog id="dlg">
    <div class="dlg-inner">
      <button class="close" type="button" onclick="document.getElementById('dlg').close()">Close</button>
      <h2 id="dlg-title"></h2>
      <p id="dlg-seen"></p>
      <div id="dlg-live" class="live-block" style="display:none">
        <h3>Live cameras</h3>
        <div id="dlg-live-rows"></div>
        <div id="dlg-live-embed" style="display:none;margin-top:.75rem">
          <p class="muted" style="margin:0 0 .35rem">This tracker (embedded)</p>
          <img id="dlg-live-mjpeg" src="" alt="" style="width:100%;border-radius:8px;border:1px solid #374151;background:#111"/>
        </div>
      </div>
      <img id="dlg-img" alt="" style="width:100%;border-radius:8px;display:none"/>
      <video id="dlg-vid" controls playsinline style="display:none"></video>
      <p id="dlg-novid" class="no-vid" style="display:none">No clip saved for this visit yet.</p>
      <p class="muted" style="margin-top:.75rem">Clips are .mp4 or .avi depending on what OpenCV can write on your system.</p>
    </div>
  </dialog>
  <dialog id="settings-dlg">
    <div class="dlg-inner">
      <button class="close" type="button" id="settings-close">Close</button>
      <h2>Tracker settings</h2>
      <p class="muted" style="margin:0 0 .75rem;font-size:.85rem">Saved to <code>settings.json</code> in the CatTracker folder. <strong>Restart <code>run.py</code></strong> after saving so the tracker picks up changes.</p>
      <form class="settings-form" id="settings-form" onsubmit="return false">
        <label>Camera source</label>
        <select id="set-camera-mode" name="camera_mode">
          <option value="hardware">Hardware camera (USB / built-in)</option>
          <option value="ip">IP / network stream</option>
        </select>
        <label>Hardware camera index</label>
        <input type="number" id="set-hw-index" min="0" max="20" step="1" value="0"/>
        <div class="hint">Usually <code>0</code> for the default webcam. On macOS the app may still probe 0→2 when index is 0.</div>
        <label>Stream URL (RTSP / HTTP MJPEG)</label>
        <input type="text" id="set-ip-url" placeholder="rtsp://user:pass@192.168.1.50:554/stream"/>
        <div class="hint">Used when “IP / network stream” is selected.</div>
        <div class="row"><label><input type="checkbox" id="set-mirror"/> Mirror camera horizontally (typical for selfie USB cams)</label></div>
        <div class="row"><label><input type="checkbox" id="set-record-audio" checked/> Record microphone into clips (needs <code>ffmpeg</code> + <code>sounddevice</code>)</label></div>
        <label>YOLO cat confidence minimum</label>
        <input type="number" id="set-yolo-conf" min="0.05" max="0.95" step="0.05" value="0.45"/>
        <div class="hint">Higher = fewer false “cat” boxes on random objects.</div>
        <div id="settings-msg" class="hint" style="min-height:1.2rem;color:#7dd3fc"></div>
        <div class="settings-actions">
          <button type="button" id="settings-save">Save settings</button>
          <button type="button" class="secondary" id="settings-reload">Reload from disk</button>
        </div>
      </form>
    </div>
  </dialog>
  <script>
  function normName(s) { return (s || '').trim().toLowerCase(); }

  async function loadCats() {
    const r = await fetch('/api/cats');
    return r.json();
  }

  function renderRecent(data) {
    const wrap = document.getElementById('recent-wrap');
    const box = document.getElementById('recent-clips');
    if (!wrap || !box) return;
    const clips = data.recent_clips || [];
    if (!clips.length) {
      wrap.style.display = 'none';
      box.innerHTML = '';
      return;
    }
    wrap.style.display = 'block';

    const desired = clips.filter(function (cl) { return cl.video_url; });
    const used = new Set();
    const existing = new Map();
    box.querySelectorAll('.recent-cell').forEach(function (cell) {
      const k = cell.getAttribute('data-clip-key');
      if (k) existing.set(k, cell);
    });

    function clipCaption(cl) {
      const p = (cl.path || '');
      const kind = p.indexOf('_full.') >= 0 ? 'full view' : (p.indexOf('_zoom.') >= 0 ? 'zoom' : 'clip');
      return (cl.label || 'Unnamed') + ' · ' + kind + ' · ' + (cl.saved_at || '');
    }

    for (let i = 0; i < desired.length; i++) {
      const cl = desired[i];
      const key = cl.path || cl.video_url;
      if (!key) continue;
      used.add(key);
      let cell = existing.get(key);
      if (!cell) {
        cell = document.createElement('div');
        cell.className = 'recent-cell';
        cell.setAttribute('data-clip-key', key);
        const v = document.createElement('video');
        v.controls = true;
        v.playsInline = true;
        v.preload = 'metadata';
        v.src = cl.video_url;
        const cap = document.createElement('div');
        cap.className = 'recent-cap';
        cell.appendChild(v);
        cell.appendChild(cap);
        existing.set(key, cell);
      }
      const v = cell.querySelector('video');
      const cap = cell.querySelector('.recent-cap');
      if (v && cl.video_url) {
        try {
          const want = new URL(cl.video_url, location.origin).href;
          if (v.src !== want) v.src = cl.video_url;
        } catch (e) {
          if (v.getAttribute('src') !== cl.video_url) v.src = cl.video_url;
        }
      }
      if (cap) cap.textContent = clipCaption(cl);
      box.appendChild(cell);
    }

    box.querySelectorAll('.recent-cell').forEach(function (cell) {
      const k = cell.getAttribute('data-clip-key');
      if (k && !used.has(k)) cell.remove();
    });
  }

  function renderGrid(data) {
    const root = document.getElementById('root');
    root.innerHTML = '';
    const cats = data.cats || [];
    if (!cats.length) {
      root.innerHTML = '<div class="empty">No sightings logged yet. Run CatTracker with Gemini and visit a cat.</div>';
      return;
    }
    for (const c of cats) {
      const el = document.createElement('div');
      el.className = 'card';
      const liveCams = c.live_cameras || [];
      const onLive = liveCams.length > 0;
      const vis = c.thumb_url
        ? '<img loading="lazy" alt="" src="' + c.thumb_url + '"/>'
        : '<div class="ph" style="aspect-ratio:4/3;background:#222;display:flex;align-items:center;justify-content:center;font-size:2rem">🐱</div>';
      el.innerHTML = (onLive ? '<span class="live-pill">LIVE</span>' : '') + vis
        + '<div class="meta"><div class="name"></div><div class="time"></div></div>';
      el.querySelector('.name').textContent = c.display || c.slug;
      let t = c.last_seen ? ('Last seen: ' + c.last_seen) : '';
      if (onLive) t += (t ? ' · ' : '') + 'On ' + liveCams.map(x => x.label).join(', ');
      el.querySelector('.time').textContent = t;
      el.addEventListener('click', () => openDlg(c));
      root.appendChild(el);
    }
  }

  let refreshTimer;
  async function refresh() {
    try {
      const data = await loadCats();
      const panel = document.getElementById('live-panel');
      const img = document.getElementById('live-mjpeg');
      if (panel && img) {
        if (data.live_stream) {
          panel.style.display = 'block';
          const base = '/live.mjpg';
          if (!img.src || img.src.indexOf('live.mjpg') < 0) {
            img.src = base + '?' + Date.now();
          }
        } else {
          panel.style.display = 'none';
          img.removeAttribute('src');
        }
      }
      renderRecent(data);
      renderGrid(data);
    } catch (e) { console.error(e); }
  }

  function openDlg(c) {
    const dlg = document.getElementById('dlg');
    document.getElementById('dlg-title').textContent = c.display || c.slug;
    document.getElementById('dlg-seen').textContent = c.last_seen ? ('Last seen (UTC): ' + c.last_seen) : '';
    const liveBlock = document.getElementById('dlg-live');
    const liveRows = document.getElementById('dlg-live-rows');
    const cams = c.live_cameras || [];
    if (cams.length) {
      liveBlock.style.display = 'block';
      liveRows.innerHTML = '';
      for (const cam of cams) {
        const row = document.createElement('div');
        row.className = 'cam-row';
        const label = document.createElement('span');
        label.textContent = cam.label + ' (' + cam.camera_id + ')';
        row.appendChild(label);
        if (cam.viewer_url) {
          const a = document.createElement('a');
          a.href = cam.viewer_url;
          a.target = '_blank';
          a.rel = 'noopener noreferrer';
          a.textContent = 'Watch live';
          row.appendChild(a);
        } else {
          const hint = document.createElement('span');
          hint.className = 'muted';
          hint.textContent = '— optional external viewer: --camera-viewer-url';
          row.appendChild(hint);
        }
        liveRows.appendChild(row);
      }
      const emb = document.getElementById('dlg-live-embed');
      const dimg = document.getElementById('dlg-live-mjpeg');
      if (emb && dimg) {
        emb.style.display = 'block';
        dimg.src = '/live.mjpg?' + Date.now();
      }
    } else {
      liveBlock.style.display = 'none';
      const emb = document.getElementById('dlg-live-embed');
      if (emb) emb.style.display = 'none';
    }
    const img = document.getElementById('dlg-img');
    const vid = document.getElementById('dlg-vid');
    const novid = document.getElementById('dlg-novid');
    if (c.thumb_url) { img.src = c.thumb_url; img.style.display = 'block'; } else { img.style.display = 'none'; }
    if (c.video_url) {
      vid.src = c.video_url; vid.style.display = 'block'; novid.style.display = 'none';
    } else {
      vid.removeAttribute('src'); vid.style.display = 'none'; novid.style.display = 'block';
    }
    dlg.showModal();
  }

  async function loadSettingsForm() {
    const r = await fetch('/api/settings');
    const s = await r.json();
    document.getElementById('set-camera-mode').value = s.camera_mode === 'ip' ? 'ip' : 'hardware';
    document.getElementById('set-hw-index').value = s.hardware_camera_index != null ? s.hardware_camera_index : 0;
    document.getElementById('set-ip-url').value = s.ip_camera_url || '';
    document.getElementById('set-mirror').checked = !!s.mirror_camera;
    document.getElementById('set-record-audio').checked = s.record_audio !== false;
    document.getElementById('set-yolo-conf').value = s.yolo_conf != null ? s.yolo_conf : 0.45;
    document.getElementById('settings-msg').textContent = '';
  }

  async function saveSettingsForm() {
    const msg = document.getElementById('settings-msg');
    msg.textContent = 'Saving…';
    const body = {
      camera_mode: document.getElementById('set-camera-mode').value,
      hardware_camera_index: parseInt(document.getElementById('set-hw-index').value, 10) || 0,
      ip_camera_url: document.getElementById('set-ip-url').value.trim(),
      mirror_camera: document.getElementById('set-mirror').checked,
      record_audio: document.getElementById('set-record-audio').checked,
      yolo_conf: parseFloat(document.getElementById('set-yolo-conf').value) || 0.45
    };
    try {
      const r = await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await r.json();
      if (j.ok) msg.textContent = 'Saved. Restart run.py to apply.';
      else msg.textContent = 'Save failed.';
    } catch (e) {
      msg.textContent = 'Save failed: ' + e;
    }
  }

  document.getElementById('btn-settings').addEventListener('click', async () => {
    await loadSettingsForm();
    document.getElementById('settings-dlg').showModal();
  });
  document.getElementById('settings-close').addEventListener('click', () => document.getElementById('settings-dlg').close());
  document.getElementById('settings-save').addEventListener('click', saveSettingsForm);
  document.getElementById('settings-reload').addEventListener('click', loadSettingsForm);

  refresh();
  refreshTimer = setInterval(refresh, 4000);
  </script>
</body>
</html>
"""


def _parse_iso_age_sec(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds()
    except (ValueError, TypeError):
        return None


def _load_fresh_live_cameras(pkg_root: Path) -> list[dict]:
    live_dir = pkg_root / "sightings" / "live"
    if not live_dir.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(live_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        age = _parse_iso_age_sec(data.get("updated"))
        if age is None or age > _LIVE_STALE_SEC:
            continue
        out.append(data)
    return out


def _live_match_for_cat(display: str, live_cams: list[dict]) -> list[dict]:
    d = (display or "").strip().lower()
    if not d:
        return []
    matched: list[dict] = []
    for cam in live_cams:
        if not cam.get("cat_in_frame"):
            continue
        names = cam.get("cats_named") or []
        for n in names:
            if isinstance(n, str) and n.strip().lower() == d:
                matched.append(
                    {
                        "camera_id": cam.get("camera_id", ""),
                        "label": cam.get("label", cam.get("camera_id", "Camera")),
                        "viewer_url": cam.get("viewer_url"),
                    }
                )
                break
    return matched


def create_app(pkg_root: Path, stream_hub: LiveMjpegHub | None = None) -> Flask:
    pkg_root = Path(pkg_root).resolve()
    sightings_dir = pkg_root / "sightings"
    index_path = sightings_dir / "index.json"

    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return _DASHBOARD_HTML

    @app.route("/live.mjpg")
    def live_mjpeg():
        hub = stream_hub
        if hub is None:
            abort(503)

        boundary = b"--frame\r\n"

        def gen():
            # Send only when the hub publishes a *new* frame. Previously we re-yielded the same
            # JPEG in a tight loop, flooding the socket and causing huge browser-side lag.
            last_seq = -1
            while True:
                jpg, seq = hub.get_frame()
                if jpg and seq > last_seq:
                    last_seq = seq
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                elif jpg is None:
                    time.sleep(0.05)
                else:
                    time.sleep(0.02)

        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )

    @app.route("/api/live")
    def api_live():
        return {"cameras": _load_fresh_live_cameras(pkg_root)}

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings():
        if request.method == "GET":
            return jsonify(to_api_dict(pkg_root))
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "expected JSON object"}), 400
        saved = save(pkg_root, payload)
        return jsonify({"ok": True, "settings": saved})

    def _recent_clips_payload(data: dict) -> list[dict]:
        raw = data.get("recent_clips")
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            rel = item.get("path")
            if not rel or not isinstance(rel, str):
                continue
            out.append(
                {
                    "path": rel,
                    "saved_at": item.get("saved_at"),
                    "label": item.get("label"),
                    "video_url": f"/media/{rel}",
                }
            )
        return out

    @app.route("/api/cats")
    def api_cats():
        live_cams = _load_fresh_live_cameras(pkg_root)
        empty = {"cats": [], "recent_clips": [], "live_stream": stream_hub is not None}
        if not index_path.is_file():
            return empty
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        raw = data.get("cats") or {}
        out = []
        for _k, v in sorted(raw.items(), key=lambda x: (x[1].get("last_seen") or ""), reverse=True):
            thumb = v.get("last_thumb")
            video = v.get("last_video")
            display = v.get("display", _k)
            out.append(
                {
                    "display": display,
                    "slug": v.get("slug", _k),
                    "last_seen": v.get("last_seen"),
                    "thumb_url": f"/media/{thumb}" if thumb else None,
                    "video_url": f"/media/{video}" if video else None,
                    "live_cameras": _live_match_for_cat(display, live_cams),
                }
            )
        return {
            "cats": out,
            "recent_clips": _recent_clips_payload(data),
            "live_stream": stream_hub is not None,
        }

    @app.route("/media/<path:rel>")
    def media(rel: str):
        rel = rel.lstrip("/\\")
        if ".." in rel or rel.startswith(("/", "\\")):
            abort(404)
        full = (pkg_root / rel).resolve()
        try:
            full.relative_to(pkg_root)
        except ValueError:
            abort(404)
        if not full.is_file():
            abort(404)
        mt = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
        return send_file(full, mimetype=mt, conditional=True, max_age=0)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="CatTracker sightings web UI")
    here = Path(__file__).resolve().parent.parent
    parser.add_argument("--root", type=Path, default=here, help="cattracker package root (sightings + recordings)")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (0.0.0.0 = LAN; use 127.0.0.1 for local only)",
    )
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    app = create_app(args.root)
    print(f"Open http://{args.host}:{args.port}/  (data root {args.root.resolve()})")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
