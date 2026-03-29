#!/bin/sh
# Refresh ha_addon/cattracker from repo root (run after editing cat_tracker/).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/cattracker"
mkdir -p "$DEST/cat_tracker"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "$ROOT/cat_tracker/" "$DEST/cat_tracker/"
cp "$ROOT/run.py" "$ROOT/requirements.txt" "$ROOT/gemini_cats.txt" "$DEST/"
echo "Synced → $DEST"
