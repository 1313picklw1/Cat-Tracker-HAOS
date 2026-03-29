#!/usr/bin/env python3
"""
CatTracker — cat-only YOLO webcam demo.

  cd cattracker
  pip install -r requirements.txt
  python run.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cat_tracker.app import run

if __name__ == "__main__":
    run()
