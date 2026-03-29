# Cat Tracker — Home Assistant add-on

Supervisor add-on: USB webcam + YOLO + optional Gemini (or `ref/` photos on Share) + web UI + MQTT entities for Home Assistant.

This GitHub repo is **HA-only** (the `cattracker/` add-on folder at the repo root). Source for the full desktop app lives in your local project copy, not in this remote.

## Stuck on “Installing”?

The **first install builds a Docker image** and downloads **PyTorch + dependencies (often ~1–2 GB)**. On a Raspberry Pi or slow internet this can take **20–60+ minutes** and the UI barely moves — that is normal.

- **Confirm it is working:** SSH / **Terminal** add-on → `ha supervisor logs` (or `docker logs hassio_supervisor`) and look for lines like **`CatTracker [4/6] PyTorch`** — if that line appeared, it is downloading; wait longer.
- **Do not** cancel and retry every minute (you restart the big download).
- After it finishes once, **updates are faster** (layers cached).

If **nothing** new appears in logs for **well over an hour**, check **disk space**, **RAM** (Pi OOM), and **network** (Pi-hole / firewall blocking `pypi.org` / `download.pytorch.org`).

### Nothing downloads at all (install never starts)

The first step must **pull the base image** from **Docker Hub** (`python:3.11-slim-bookworm`). If your HA host or VM **blocks `registry-1.docker.io`** or **`ghcr.io`**, the UI can sit forever with no visible progress.

- On the HA host (SSH): `docker pull python:3.11-slim-bookworm` — if this hangs or errors, fix **DNS / firewall / proxy** first.
- This add-on uses **Docker Hub** for the base (not GitHub Container Registry) so it works on networks that only allow Docker Hub.

## Install

1. **Settings → Add-ons → Add-on store → ⋮ → Repositories** and add the **repository URL** (root must contain this `cattracker` folder and `repository.json`).
2. Refresh the store, install **Cat Tracker** (expect a long first build), open **Configuration**, then **Start**.

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

Supervisor often hides the real error behind that message. You need the **build** log:

1. **SSH / Terminal add-on** (or host shell):  
   `ha supervisor logs`  
   or  
   `docker logs hassio_supervisor 2>&1 | tail -200`  
   Scroll for `cattracker`, `pip`, `torch`, `no space`, or `OOM`.

2. **Disk space** — PyTorch + Ultralytics needs **several GB free** during `pip install`. If you previously saw **GitRepo.clone blocked … not enough free space**, fix storage first.

3. **RAM (Raspberry Pi)** — building this image can exceed **2–4 GB RAM** and get OOM-killed. Prefer **Home Assistant on x86_64 / NUC**, or build the image on a PC and load it (advanced), or run CatTracker on another machine with MQTT only.

4. After changing the Dockerfile, use **⋯ → Rebuild** on the add-on so Supervisor doesn’t reuse a bad cached layer.

This add-on uses **opencv-python-headless** and **no PortAudio** (no `sounddevice` in the image).

### `apt-get: not found` / `/bin/ash`

That means the build used an **Alpine** base. This repo’s **`build.yaml`** pins **`python:3.11-slim-bookworm`** (Debian on **Docker Hub**) so **`apt-get`** exists. If you still see Alpine, your Supervisor may be ignoring `build.yaml` — open an issue with your HA OS / Supervisor version. **Rebuild** after updating.

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
