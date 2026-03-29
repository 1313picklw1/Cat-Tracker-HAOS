#!/bin/sh
set -e
CFG="/data/options.json"

if [ ! -f "$CFG" ]; then
  echo "CatTracker: missing $CFG" >&2
  exit 1
fi

CAM=$(jq -r '.camera // "0"' "$CFG")
IDENT=$(jq -r '.identification // "gemini"' "$CFG")
KEY=$(jq -r '.gemini_api_key // ""' "$CFG")
MQTT_HOST=$(jq -r '.mqtt_host // ""' "$CFG")
MQTT_PORT=$(jq -r '.mqtt_port // 1883' "$CFG")
MQTT_USER=$(jq -r '.mqtt_user // ""' "$CFG")
MQTT_PASS=$(jq -r '.mqtt_password // ""' "$CFG")
WEB_PORT=$(jq -r '.web_port // 5050' "$CFG")
REC_SUB=$(jq -r '.recordings_subdir // "cattracker/recordings"' "$CFG")
SIG_SUB=$(jq -r '.sightings_subdir // "cattracker/sightings"' "$CFG")
REF_SUB=$(jq -r '.ref_subdir // "cattracker/ref"' "$CFG")
CAM_ID=$(jq -r '.camera_id // "main"' "$CFG")
CAM_LABEL=$(jq -r '.camera_label // "Cat Tracker"' "$CFG")

RECORD_DIR="/share/${REC_SUB}"
SIGHTINGS_DIR="/share/${SIG_SUB}"
REF_DIR="/share/${REF_SUB}"

mkdir -p "$RECORD_DIR" "$SIGHTINGS_DIR" "$REF_DIR"

export GEMINI_API_KEY="$KEY"
export CATTRACKER_HEADLESS=1

if [ -n "$MQTT_HOST" ]; then
  export CATTRACKER_MQTT_HOST="$MQTT_HOST"
  export CATTRACKER_MQTT_PORT="$MQTT_PORT"
  [ -n "$MQTT_USER" ] && export CATTRACKER_MQTT_USER="$MQTT_USER"
  [ -n "$MQTT_PASS" ] && export CATTRACKER_MQTT_PASSWORD="$MQTT_PASS"
fi

set -- python /app/run.py \
  --headless \
  --web-host 0.0.0.0 \
  --web-port "$WEB_PORT" \
  --camera "$CAM" \
  --camera-id "$CAM_ID" \
  --camera-label "$CAM_LABEL" \
  --record-dir "$RECORD_DIR" \
  --sightings-dir "$SIGHTINGS_DIR"

case "$IDENT" in
  gemini) set -- "$@" --gemini ;;
  ref) set -- "$@" --ref-dir "$REF_DIR" ;;
  none) set -- "$@" --no-id ;;
  *) set -- "$@" --gemini ;;
esac

exec "$@"
