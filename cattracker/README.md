# Cat Tracker — Home Assistant add-on

Supervisor add-on: USB webcam + YOLO + optional Gemini (or `ref/` photos on Share) + web UI + MQTT entities for Home Assistant.

This GitHub repo is **HA-only** (the `cattracker/` add-on folder at the repo root). Source for the full desktop app lives in your local project copy, not in this remote.

## Install

1. **Settings → Add-ons → Add-on store → ⋮ → Repositories** and add the **repository URL** (root must contain this `cattracker` folder and `repository.json`).
2. Refresh the store, install **Cat Tracker**, open **Configuration**, then **Start**.

### Options (UI)

| Option | Purpose |
|--------|--------|
| **Camera source** | **USB** = local camera (`camera_usb_index`, usually `0`). **IP** = network stream (`camera_url`: `rtsp://…`, `http://…` MJPEG, etc.). |
| **Mirror camera** | Flip image for USB selfie cams; turn **off** for most IP cameras if the picture looks mirrored. |
| **Identification** | **Gemini** (needs **Gemini API key** + optional **Gemini model**), **ref** (photos on Share), or **none**. |
| **MQTT broker host** | e.g. `core-mosquitto` — leave empty to disable MQTT entities. |
| **Record microphone** | Default **off** in Docker (no PortAudio); clips are video-only unless you know your setup supports audio. |

Edit **`gemini_cats.txt`** in the image by rebuilding after changing the repo, or map a file over `/app/gemini_cats.txt` with advanced Docker overrides.

**Ref mode:** add photos under **Share** → `cattracker/ref/<CatName>/` (one folder per cat name).

3. Enable **MQTT** in Home Assistant if you set **mqtt_host**.

## Build failed (“unknown error”)

The first install **builds a large image** (PyTorch + YOLO). Check:

1. **Disk space** — need several GB free on the HA host (Supervisor blocks git/build when almost full).
2. **Logs** — **Settings → System → Logs**, or SSH: `ha supervisor logs`, or **Add-on** → **Log** after a failed build.
3. **RAM** — Raspberry Pi builds can OOM; try a **PC / NUC** HA install or increase swap (advanced).

This add-on uses **opencv-python-headless** and **no `sounddevice`** by default so the Docker build does not compile PortAudio.

## MQTT

With **mqtt_host** set, CatTracker publishes Home Assistant **MQTT discovery** configs:

- Binary sensor (occupancy): cat in frame **ON**/**OFF**
- Sensor: comma-separated **identified** cat names (empty when unknown)

Device name uses **camera_label** / **camera_id**.

## Local run (same MQTT)

```bash
export CATTRACKER_MQTT_HOST=<broker>
export CATTRACKER_HEADLESS=1
python run.py --gemini --headless --mqtt-host <broker> --web-host 0.0.0.0
```

## Updating the add-on copy from a full CatTracker tree

If you keep the **full app** elsewhere, refresh `cattracker/cat_tracker/`, `run.py`, `requirements.txt`, and `gemini_cats.txt` before committing:

```bash
rsync -a --delete --exclude '__pycache__' /path/to/full/cat_tracker/ ./cattracker/cat_tracker/
cp /path/to/full/run.py /path/to/full/requirements.txt /path/to/full/gemini_cats.txt ./cattracker/
```

Then commit and push this repo.

## Notes

- First run downloads YOLO weights (`yolov8n.pt`).
- Image is large (PyTorch). Prefer **amd64**; **aarch64** may be slow.
- Update `repository.json` at the **repository root** with your real Git URL.
