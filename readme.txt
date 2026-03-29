CatTracker
==========

Webcam app that detects **cats** only (YOLOv8 + COCO). Left: live view with boxes.
Right: zoomed crop of the highest-confidence cat, or a “No cat” panel.

Who is who — pick one
----------------------

**A) Gemini (text descriptions, no photos)**  
  export GEMINI_API_KEY="..."   # from Google AI Studio  
  python run.py --gemini  

  **Quota is per Google Cloud project**, not per API key string. If you “make a new key”
  in the **same** AI Studio / GCP project, you still share the same limits as the old key.
  For a separate pool you need a **different project** (or enable billing / another model).

  Edit **`gemini_cats.txt`**: optional first line `NAMES: Snoop, Pancake, ...` then
  sentences like *Snoop is a full black cat. Pancake is a stripy tabby...*

  **Default (motion mode):** **One Gemini call per cat visit** — when YOLO starts
  seeing a cat, we ask Gemini once and keep that name until the visit **really**
  ends. A visit ends only after **`--gemini-visit-end-miss`** consecutive frames
  with **no** smoothed box (default **22**), so a **brief dropout** (edge of frame,
  motion blur, looking away) does **not** reset the visit or call Gemini again.
  **`--miss`** (default **14**) is how many frames YOLO can miss before the box
  disappears — raise it (e.g. **20–25**) if the cat often turns away from the
  camera. Use **`--gemini-refresh`** for the old re-query-on-motion behavior.
  If **another cat** appears during the same visit, **`--gemini-swap-threshold`**
  (default **0.21**) detects a much **different-looking crop** vs the one used for
  the current name and **calls Gemini again** to correct it. Use **0** to turn off.

  **Cooldown** (`--gemini-cooldown`, default 2s) still applies between calls if a
  visit retries (e.g. rate limit). YOLO crop is sent when available; otherwise the
  **full frame**. In **dark** scenes, crops are **brightened** (CLAHE + shadow lift)
  for the **zoom panel**, **Gemini**, **sighting thumbnails**, and **swap detection**
  so dark-coated cats stay visible; use **`--no-lowlight-boost`** if it looks washed
  out in a bright room.

  **Sightings + small website:** With sightings logging on (default), thumbnails and
  clip paths go under **`sightings/`** (`index.json` + `thumbs/`). Every finished
  recording is also listed under **Recent clips** on the dashboard, even if the cat
  was not named yet. The **`python run.py`** process starts the dashboard whenever
  **`--no-web`** is not set, at **http://127.0.0.1:5050/** (**`--web-port`** /
  **`--web-host`**). The home page embeds a **live MJPEG preview** of the same OpenCV
  view (camera + boxes + panel) at **`/live.mjpg`** while the tracker is running.
  **`--no-sightings-log`** disables `sightings/` writes (no thumbs/clips in the UI).
  Standalone UI only (no live stream): **`python run_web.py`**.

  **Recordings:** If clips fail or are empty, OpenCV may not support MP4 on your OS;
  the recorder tries **mp4v / avc1** then **MJPG `.avi`**. Watch the terminal for
  `Saved … (N bytes)`.

  **Live on camera:** Each run can write **`sightings/live/<camera-id>.json`** (default
  on). Use **`--camera-id`** / **`--camera-label`** per webcam; **`--camera-viewer-url`**
  is the link opened when you click **Watch live** in the dashboard (your own stream
  page, etc.). **`--no-live-state`** disables live files.

  **Legacy interval mode:** `python run.py --gemini --gemini-no-motion` uses
  `--gemini-every` on the YOLO crop only (no “is there a cat?” on the whole frame).
  With **`--gemini-refresh`**, it polls every N frames; otherwise **once per visit**
  on the first qualifying frame.

  Model: `--gemini-model` (default `gemini-2.5-flash`). Google drops old ids (e.g. `gemini-1.5-flash`
  often returns 404); see https://ai.google.dev/gemini-api/docs/models — try `gemini-2.5-flash-lite` for a lighter option.

  If you see **429 / quota exceeded** (sometimes `limit: 0` on the free tier for a
  model), wait for the reset, switch model, or enable billing on the API project.
  The app backs off using the server’s retry hint so it won’t hammer the API.

**B) Reference photos (local, no API)**  
  `ref/<CatName>/` with JPG/PNG — see `ref/readme.txt`. Default when you **omit**
  `--gemini`.

**Neither:** `python run.py --no-id`

**Auto-record:** While a cat is tracked, each sighting saves an MP4 under
`cattracker/recordings/`. Filename starts with the **matched cat slug** (or
`cat` if unknown). The saved frame layout: **full camera (with boxes) in the
top-left**, larger **cat zoom panel** filling the rest (1280x720 by default),
plus an **identity line** at the bottom when known. A red **REC** dot appears on
the live window while recording. Clips **end only after** **`--record-end-miss`**
consecutive frames with no tracked cat (default **24**, ~0.8s at 30fps) so brief
YOLO dropouts do not split one visit into many tiny files. Raise it (e.g. `48`)
if you still see chatter, or **`--miss`** (box smoother) to hold the box longer.

  cd cattracker
  pip install -r requirements.txt
  python run.py --gemini

Home Assistant
--------------
- **Supervisor add-on:** the `cattracker/` folder at the **repository root** is a Home Assistant
  add-on (see `cattracker/README.md`). Add this repo under **Settings → Add-ons → Repositories**,
  install **Cat Tracker**, set **mqtt_host** (e.g. `core-mosquitto`) for MQTT discovery entities.
- **CLI / Docker elsewhere:** use **`--headless`** (no OpenCV window) and either **`--mqtt-host`**
  or **`CATTRACKER_MQTT_HOST`** so the **MQTT** integration gets a **cat present** binary sensor
  and **cats identified** text sensor (`paho-mqtt` in `requirements.txt`).

First launch downloads **yolov8n.pt** (small, fast). On Apple Silicon, inference
uses **MPS** when available.

Options (partial)
-----------------
  --gemini                  Use Gemini instead of ref/ photos
  --gemini-prompt PATH      Prompt file (default: ./gemini_cats.txt)
  --gemini-model NAME       e.g. gemini-2.5-flash, gemini-2.5-flash-lite
  --gemini-no-motion        Poll every --gemini-every frames (no motion gate)
  --gemini-every N          With --gemini-no-motion only
  --motion-threshold F      Motion score to trigger Gemini (default 0.012)
  --gemini-cooldown SEC     Min seconds between Gemini calls (default 2)
  --gemini-stale SEC        Refresh while YOLO sees cat with no motion (default 14)
  --gemini-max-side PX      Cap image size sent to Gemini (smaller often a bit faster)
  --no-lowlight-boost       Disable CLAHE/shadow lift (default ON for dark rooms)
  --gemini-refresh          Re-query Gemini on motion/stale (not once per visit)
  --gemini-visit-end-miss N Frames with no box before visit ends (default 22)
  --gemini-swap-threshold D  Re-ID if crop differs from locked ref (0=off, default 0.21)
  --miss N                  YOLO misses before dropping box (default 14)
  --no-sightings-log        Don’t write sightings/ for the web UI
  --sightings-dir PATH      Where to store thumbs + index.json
  --no-web                  Don’t start dashboard in background (logging can still run)
  --web-host ADDR           Dashboard bind (default 127.0.0.1)
  --web-port N              Dashboard port (default 5050)
  --camera-id ID            Live-state file id (multi-camera)
  --camera-label TEXT       Name shown in dashboard
  --camera-viewer-url URL   “Watch live” opens this in a new tab
  --headless                No GUI window (SIGTERM to stop; for servers / HA add-on)
  --mqtt-host HOST          Home Assistant MQTT broker; optional CATTRACKER_MQTT_* env vars
  --mqtt-port / --mqtt-user / --mqtt-password
  --no-live-state           Don’t write sightings/live/*.json
  --verbose                 Log motion score and each Gemini call

**Why Gemini feels slow:** Most delay is **Google’s network + model** (often a few seconds per
call), not local Python. You can try `--gemini-max-side 512` or `640` to shrink uploads slightly.
There is still a **cooldown** between calls (`--gemini-cooldown`, default 2s).

**Speed / 30 FPS:** The main loop targets **`--target-fps 30`** (default): it **sleeps** when
YOLO finishes faster than that so the app stays near 30 FPS instead of pinning CPU. Use
**`--target-fps 0`** to run uncapped. The camera is set to **30 FPS** and **buffer size 1**
(less lag). YOLO defaults to **`--imgsz 416`** for speed (try **`640`** if boxes feel loose).
Saved clips default to **`--record-fps 30`**; use **`--record-use-camera-fps`** if you want the
driver-reported FPS in the file metadata instead.

**Zoom / box accuracy:** The **green** box is **smoothed** and trails the **cyan** raw YOLO
box — that can look “off” vs the cat. Raise `--smooth` toward **0.5–0.6** for snappier tracking
(more jitter), or lower for steadier boxes. For better localization use **`--imgsz 640`** and/or
a larger model, e.g. **`--model yolov8s.pt`** (heavier). The side-panel zoom now uses the **same
padding** as the crop sent to Gemini.

If Gemini seems never to run: use `--verbose` (watch `motion=` vs threshold), or
lower `--motion-threshold`. The app also does one **kickoff** call on frame 1 so
the API is hit even with a static scene. You must run with **`--gemini`** and
`GEMINI_API_KEY` set.
  --no-mirror / --imgsz / --device / --model  (YOLO)
  --target-fps N            Main loop cap (default 30; 0 = unlimited)
  --record-use-camera-fps   Use camera driver FPS for clips instead of --record-fps
  --no-record / --record-dir / --record-fps / --record-w / --record-h / --record-end-miss
  --ref-dir / --sim / --no-fur-enhance / --id-every  (ref mode only)
  --no-id                   No naming at all

Q or Esc — quit

Note: YOLO + Gemini cost money per API call; adjust `--gemini-every` to balance
latency and spend. This is not cryptographic identity — wrong guesses can happen.
