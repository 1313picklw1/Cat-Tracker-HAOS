"""Match live cat crops to reference photos (one folder per cat name under ref/)."""

from __future__ import annotations

import re
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

from .detector import TRACK_CROP_PAD_FRAC, CatBox, CatDetector

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def enhance_dark_fur_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE on luminance — webcams often crush black fur into a flat patch; this
    pulls out coat texture before the neural net sees the crop.
    """
    if bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    tile = 4 if min(h, w) < 112 else 8
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(tile, tile))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l2 = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l2, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def preview_enhance_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Live UI / recordings: avoid heavy CLAHE in normal light (flat, gray-ish look).
    Only apply strong low-light boost when the scene is actually dark.
    """
    if bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, _, _ = cv2.split(lab)
    mean_l = float(np.mean(l_ch)) / 255.0
    if mean_l >= 0.40:
        return bgr
    return boost_lowlight_bgr(bgr)


def boost_lowlight_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Stronger lift for dim webcams: CLAHE + gentle shadow gamma so dark / black
    cats (and coats) stay visible for you and for Gemini.
    """
    if bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    m = min(h, w)
    tile = max(4, min(16, max(m // 28, 4)))
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(tile, tile))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    mean_l = float(np.mean(l_ch)) / 255.0
    if mean_l > 0.58:
        return enhance_dark_fur_bgr(bgr)
    l2 = clahe.apply(l_ch)
    gamma = 0.72 if mean_l < 0.35 else 0.82
    lift = 10.0 if mean_l < 0.35 else 4.0
    l3 = l2.astype(np.float32) / 255.0
    l3 = np.power(l3, gamma) * 255.0 + lift
    l3 = np.clip(l3, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l3, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def crop_cat_bgr(frame_bgr: np.ndarray, box: CatBox, pad_frac: float = TRACK_CROP_PAD_FRAC) -> np.ndarray | None:
    fh, fw = frame_bgr.shape[:2]
    bw = box.x2 - box.x1
    bh = box.y2 - box.y1
    if bw < 2 or bh < 2:
        return None
    px, py = bw * pad_frac, bh * pad_frac
    x1 = int(max(0, box.x1 - px))
    y1 = int(max(0, box.y1 - py))
    x2 = int(min(fw, box.x2 + px))
    y2 = int(min(fh, box.y2 + py))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    if not crop.flags["C_CONTIGUOUS"]:
        crop = np.ascontiguousarray(crop)
    return crop


def recording_slug(label: str) -> str:
    """Safe filename fragment from a display label."""
    s = re.sub(r"[^\w\-\s]+", "", label, flags=re.UNICODE)
    s = s.strip().replace(" ", "_")
    return s or "cat"


class LabelSmoother:
    """Majority vote over recent IDs to reduce flicker."""

    def __init__(self, maxlen: int = 11) -> None:
        self._buf: deque[str] = deque(maxlen=maxlen)

    def push(self, label: str) -> None:
        self._buf.append(label)

    def clear(self) -> None:
        self._buf.clear()

    def mode(self) -> str:
        if not self._buf:
            return "?"
        return Counter(self._buf).most_common(1)[0][0]


class CatIdentifier:
    """
    Load **every** image under ref/<Name>/ — each file becomes its own embedding.
    At match time, for each cat we take the **best** cosine similarity across
    all of that cat’s reference photos (so many angles/poses help).

    Embeddings blend crop + horizontally flipped crop so ref photos (usually not
    mirrored) still match the live feed (often mirrored).
    """

    def __init__(
        self,
        ref_root: Path,
        device: torch.device,
        detector: CatDetector,
        sim_threshold: float = 0.42,
        fur_enhance: bool = True,
    ) -> None:
        self.ref_root = Path(ref_root)
        self.device = device
        self.detector = detector
        self.sim_threshold = sim_threshold
        self._fur_enhance = fur_enhance
        self.names: list[str] = []
        # name -> (K, D) float tensor, each row L2-normalized (K = number of ref images)
        self._galleries: dict[str, torch.Tensor] = {}

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        self._net = torch.nn.Sequential(*list(backbone.children())[:-1])
        self._net.to(device)
        self._net.eval()
        self._tfm = weights.transforms()

        self._load_refs()

    @property
    def ready(self) -> bool:
        return len(self._galleries) > 0

    def _embed_pil(self, pil: Image.Image) -> torch.Tensor:
        t = self._tfm(pil.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            e = self._net(t).squeeze().flatten()
        e = e / (e.norm() + 1e-8)
        return e

    def embed_bgr(self, bgr: np.ndarray) -> torch.Tensor | None:
        if bgr.size == 0 or bgr.shape[0] < 16 or bgr.shape[1] < 16:
            return None
        if self._fur_enhance:
            bgr = enhance_dark_fur_bgr(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self._embed_pil(pil)

    def _symmetric_embed(self, bgr: np.ndarray) -> torch.Tensor | None:
        """L2-normalized (embed(crop) + embed(flip(crop))) — matches mirrored webcam."""
        e0 = self.embed_bgr(bgr)
        if e0 is None:
            return None
        e1 = self.embed_bgr(cv2.flip(bgr, 1))
        if e1 is None:
            return e0
        s = e0 + e1
        return s / (s.norm() + 1e-8)

    def _ref_crop(self, path: Path, bgr: np.ndarray) -> np.ndarray:
        boxes = self.detector.detect(bgr)
        if boxes:
            c = crop_cat_bgr(bgr, boxes[0], pad_frac=0.18)
            if c is not None:
                return c
        return bgr

    def _load_refs(self) -> None:
        if not self.ref_root.is_dir():
            return

        vecs: list[torch.Tensor] = []
        labels: list[str] = []

        for sub in sorted(self.ref_root.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            cat_name = sub.name
            got = 0
            for p in sorted(sub.iterdir()):
                if p.suffix.lower() not in _IMG_EXT:
                    continue
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                crop = self._ref_crop(p, bgr)
                e = self._symmetric_embed(crop)
                if e is None:
                    continue
                vecs.append(e.cpu())
                labels.append(cat_name)
                got += 1
            if got:
                print(f"  ref/{cat_name}/  → {got} image(s)")

        if not vecs:
            return

        by_name: dict[str, list[torch.Tensor]] = {}
        for lab, v in zip(labels, vecs):
            by_name.setdefault(lab, []).append(v)

        self.names = sorted(by_name.keys())
        self._galleries = {}
        for lab in self.names:
            stacked = torch.stack(by_name[lab], dim=0).to(self.device)
            stacked = stacked / (stacked.norm(dim=1, keepdim=True) + 1e-8)
            self._galleries[lab] = stacked

    def match(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if not self.ready:
            return "?", 0.0
        e = self._symmetric_embed(crop_bgr)
        if e is None:
            return "?", 0.0
        best_name = ""
        best_score = -1.0
        for name in self.names:
            g = self._galleries[name]
            sims = g @ e
            mx = float(sims.max().item())
            if mx > best_score:
                best_score = mx
                best_name = name
        if best_score < self.sim_threshold:
            return "unknown", best_score
        return best_name, best_score


def make_device(preference: str | None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
