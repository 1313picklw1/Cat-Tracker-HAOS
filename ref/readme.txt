Reference photos — who is which cat
=====================================

Put one **subfolder per cat**, using the folder name as their label (e.g. `Snoop`, `Oscar`).

**Use many photos per cat** — every `.jpg` / `.png` in that folder is loaded.
The app compares the live view to **each** reference and picks the **strongest**
match for that cat, then picks which cat wins overall. More angles, lighting,
and distances make recognition steadier than a single snapshot.

  ref/
    Snoop/
      01.jpg
      02.jpg
      03.png
    Oscar/
      a.jpeg
      b.jpg

Supported extensions: .jpg .jpeg .png .webp .bmp

Tips
----
  • Folder name = label, e.g. ref/Snoop/ for cat “Snoop” (case-sensitive on some OS).
  • Several photos per cat work better than one; each file is matched separately.
  • Add shots that look like your webcam: similar distance, lighting, angle.
  • The app mirrors the live camera; ref matching blends left/right flips so
    snapshots from a phone still align, but extra Snoop photos from the same
    webcam help a lot.
  • **Black / very dark fur:** webcams often lose coat detail (flat black blob).
    The app applies **CLAHE** on crops by default to bring texture back; add
    bright, diffuse light and ref photos taken with the **same webcam**. If still
    “unknown”, try:  python run.py --sim 0.35
  • Clear face/body shots help. If YOLO finds a cat in a ref image, that crop is used;
    otherwise the whole file is used.
  • Run from `cattracker/` so the default `--ref-dir ref` resolves here.
