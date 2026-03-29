# Cat Tracker — Home Assistant add-on

Supervisor add-on: USB webcam + YOLO + optional Gemini (or `ref/` photos on Share) + web UI + MQTT entities for Home Assistant.

This GitHub repo is **HA-only** (the `cattracker/` add-on folder at the repo root). Source for the full desktop app lives in your local project copy, not in this remote.

## Install

1. **Settings → Add-ons → Add-on store → ⋮ → Repositories** and add the **repository URL** (root must contain this `cattracker` folder and `repository.json`).
2. Refresh the store, install **Cat Tracker**, configure options, **Start**.
3. Enable **MQTT** (e.g. Mosquitto add-on). Set **mqtt_host** to your broker (often `core-mosquitto` for the official Mosquitto add-on on the same host).
4. **Gemini**: set **identification** to `gemini` and paste **gemini_api_key**. Edit prompts by mapping a file over `/app/gemini_cats.txt` or rebuild after changing the repo copy.
5. **Ref mode**: set **identification** to `ref` and add photos under **Share** → `cattracker/ref/<CatName>/` (one folder per cat name, JPG/PNG inside).

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
