#!/usr/bin/env python3
"""
Sightings dashboard only (no camera). The main app starts this automatically
when sightings logging is on; use this if you only want the browser UI.
  python run_web.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cat_tracker.web_dashboard import main

if __name__ == "__main__":
    main()
