"""Identify cats with Google Gemini vision from a text description (no reference photos)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .cat_id import boost_lowlight_bgr


def extract_names_from_descriptions(body: str) -> list[str]:
    """Parse 'Name is a ...' sentences (e.g. Snoop is a full black cat.)."""
    pat = re.compile(
        r"(?:^|[.!?]\s+)\s*([A-Za-z][A-Za-z0-9'-]{0,39})\s+is\s+a\b",
        re.MULTILINE | re.IGNORECASE,
    )
    seen: list[str] = []
    for m in pat.finditer(body):
        n = m.group(1)
        if n not in seen:
            seen.append(n)
    return seen


def extract_names_from_label_lines(body: str) -> list[str]:
    """
    Parse lines like 'Peaches (calico): ...' or 'Snoop: ...' at the start of a line.
    Skips preamble lines such as 'system instructions:'.
    """
    seen: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("system ") or low.startswith("names:"):
            continue
        m = re.match(
            r"^([A-Za-z][A-Za-z0-9'-]{0,39})\s*(?:\([^)]*\))?\s*:\s*",
            line,
        )
        if not m:
            continue
        n = m.group(1)
        if n.lower() in ("http", "https", "names"):
            continue
        if not any(x.lower() == n.lower() for x in seen):
            seen.append(n)
    return seen


def load_gemini_prompt_file(path: Path) -> tuple[list[str], str]:
    """
    Optional first line: NAMES: Snoop, Pancake, Peaches, Socks
    Rest: free-form descriptions (also used to infer names if NAMES omitted).
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], ""
    lines = text.splitlines()
    names: list[str] = []
    body = text
    first = lines[0].strip()
    if first.upper().startswith("NAMES:"):
        part = first.split(":", 1)[1]
        names = [n.strip() for n in part.split(",") if n.strip()]
        body = "\n".join(lines[1:]).strip()
    if not names:
        names = extract_names_from_descriptions(body)
    if not names:
        names = extract_names_from_label_lines(body)
    return names, body


def _resize_for_api(bgr: np.ndarray, max_side: int = 768) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    s = max_side / max(h, w)
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _parse_retry_delay_seconds(exc: BaseException) -> float | None:
    """Best-effort parse of server hint (Gemini / gRPC style) from exception text."""
    s = str(exc)
    m = re.search(r"retry in ([\d.]+)\s*s", s, re.IGNORECASE)
    if m:
        return max(1.0, float(m.group(1)))
    m = re.search(r"seconds:\s*(\d+)", s)
    if m:
        return max(1.0, float(m.group(1)))
    return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("ResourceExhausted", "TooManyRequests"):
        return True
    s = str(exc).lower()
    return "429" in s or ("quota" in s and "exceed" in s)


def _is_model_not_found_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "404" in s and ("not found" in s or "is not found" in s) and "model" in s


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if "```" in t:
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _negated_name_mention(text: str, match_start: int) -> bool:
    """True if 'not' / 'never' / contraction immediately precedes this name (word-boundary match)."""
    lo = text.lower()
    w = lo[max(0, match_start - 40) : match_start]
    return bool(
        re.search(
            r"\b(not|never|no)\s+$",
            w,
        )
        or re.search(
            r"\b(isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't)\s+$",
            w,
        )
    )


def _loose_json_cat_value(text: str) -> str | None:
    """Best-effort extract cat/name field when strict JSON parse fails."""
    m = re.search(
        r'["\']cat["\']\s*:\s*["\']([^"\'\\]+)["\']',
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r'["\']name["\']\s*:\s*["\']([^"\'\\]+)["\']',
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def _best_name_from_prose(text: str, allowed: list[str]) -> str | None:
    """
    Find allowed names in free text; ignore clear negations ('not Snoop').
    If several positive mentions, prefer the last (model often corrects earlier guess).
    """
    low = text.lower()
    hits: list[tuple[int, str]] = []
    for a in allowed:
        for m in re.finditer(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", low):
            if _negated_name_mention(text, m.start()):
                continue
            hits.append((m.start(), a))
    if not hits:
        return None
    hits.sort(key=lambda x: x[0])
    return hits[-1][1]


def _parse_crop_response(text: str, allowed: list[str]) -> str:
    """Crop path: no separate 'none' — map to unknown."""
    canon = {a.lower(): a for a in allowed}
    try:
        data = json.loads(_strip_json_fence(text))
        if isinstance(data, dict):
            cat = data.get("cat", data.get("name", "unknown"))
        else:
            cat = "unknown"
        c = str(cat).strip()
        if c.lower() in ("unknown", "none", "", "no"):
            return "unknown"
        if c.lower() in canon:
            return canon[c.lower()]
    except (json.JSONDecodeError, TypeError):
        pass
    loose = _loose_json_cat_value(text)
    if loose:
        canon = {a.lower(): a for a in allowed}
        k = loose.lower()
        if k in ("unknown", "none", "no", ""):
            return "unknown"
        if k in canon:
            return canon[k]
    hit = _best_name_from_prose(text, allowed)
    if hit is not None:
        return hit
    return "unknown"


def _parse_scene_response(text: str, allowed: list[str]) -> str:
    """Full-frame / motion path: explicit no-cat -> 'none'."""
    canon = {a.lower(): a for a in allowed}
    try:
        data = json.loads(_strip_json_fence(text))
        if isinstance(data, dict):
            cat = data.get("cat", data.get("name", "unknown"))
        else:
            cat = "unknown"
        c = str(cat).strip()
        if c.lower() in ("none", "no", "false", "null"):
            return "none"
        if c.lower() in ("unknown", "", "unclear"):
            return "unknown"
        if c.lower() in canon:
            return canon[c.lower()]
    except (json.JSONDecodeError, TypeError):
        pass
    loose = _loose_json_cat_value(text)
    if loose:
        c = loose.strip()
        cl = c.lower()
        canon = {a.lower(): a for a in allowed}
        if cl in ("none", "no", "false", "null"):
            return "none"
        if cl in ("unknown", "", "unclear"):
            return "unknown"
        if cl in canon:
            return canon[cl]
    low = text.lower()
    if re.search(r"\bno cat\b|\bno visible cat\b|\bno cats\b", low):
        return "none"
    hit = _best_name_from_prose(text, allowed)
    if hit is not None:
        return hit
    return "unknown"


class GeminiCatIdentifier:
    def __init__(
        self,
        api_key: str,
        names: list[str],
        description_block: str,
        model: str = "gemini-2.5-flash",
        image_max_side: int | None = None,
        lowlight_boost: bool = True,
    ) -> None:
        import google.generativeai as genai

        if not names:
            raise ValueError("No cat names — add NAMES: line or 'Name is a ...' sentences in prompt file")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(model)
        self.names = list(names)
        self._lowlight_boost = bool(lowlight_boost)
        if image_max_side is not None:
            m = max(256, min(int(image_max_side), 2048))
            self._max_identify = m
            self._max_scene_crop = m
            self._max_scene_full = m
        else:
            self._max_identify = 768
            self._max_scene_crop = 896
            self._max_scene_full = 1024
        self._rate_limit_until = 0.0
        self.last_call_rate_limited = False
        allowed = ", ".join(f'"{n}"' for n in names)
        discrimination = (
            "\n\nCoat / pattern rules (avoid mixing up tabby vs solid black):\n"
            "- If you see any TABBY pattern (stripes, mackerel lines, or a brown/tan/gray base with darker markings), "
            "that cat is NOT the solid black cat — pick the tabby from your list or unknown.\n"
            "- Solid black means the visible fur is uniformly dark with NO stripe pattern on the body. "
            "Dim lighting or shadows are not enough to call a cat solid black if stripes or a warm brown coat are plausible.\n"
            "- If you are torn between a dark tabby and solid black, answer unknown instead of guessing.\n"
        )
        self._crop_instruction = (
            "You are given ONE image: a webcam crop that should show a single cat "
            "(maybe partial, blurry, or dark).\n\n"
            f"You must pick exactly one cat name from this list: {allowed}\n"
            'or use "unknown" if the image has no clear cat, multiple cats, or none of the listed cats fit.\n\n'
            "Use these descriptions to match appearance, coat pattern, and colors:\n\n"
            f"{description_block}"
            f"{discrimination}\n\n"
            'Reply with ONLY a JSON object and no other text, for example: {"cat":"Snoop"}\n'
            "Use the exact spelling of the name from the list."
        )
        self._scene_instruction = (
            "You are given ONE image from a home webcam (full view or a crop).\n\n"
            "Step 1: Is there clearly at least one cat visible in the image?\n"
            "Step 2: If YES, decide which cat from the list best matches (coat, pattern, colors). "
            "If several cats appear, pick the one that is largest or most central.\n"
            "If NO cat is visible, you must answer with cat set to none.\n\n"
            f"Allowed cat names (exact spelling): {allowed}\n\n"
            "Reference descriptions:\n\n"
            f"{description_block}"
            f"{discrimination}\n\n"
            'Reply with ONLY valid JSON, no markdown:\n'
            '- If no cat is visible: {"cat":"none"}\n'
            '- If a cat is visible but you cannot match the list: {"cat":"unknown"}\n'
            '- Otherwise: {"cat":"Name"} with Name from the allowed list.'
        )

    def _prep_bgr_for_vision(self, bgr: np.ndarray) -> np.ndarray:
        if not self._lowlight_boost or bgr.size == 0:
            return bgr
        return boost_lowlight_bgr(bgr)

    @property
    def ready(self) -> bool:
        return len(self.names) > 0

    @property
    def rate_limit_until(self) -> float:
        """Monotonic time before another API call is allowed (429 / quota backoff)."""
        return self._rate_limit_until

    def _generate(self, instruction: str, pil: Image.Image) -> str:
        now = time.monotonic()
        try:
            try:
                resp = self._model.generate_content(
                    [instruction, pil],
                    request_options={"timeout": 60},
                )
            except TypeError:
                resp = self._model.generate_content([instruction, pil])
            return (resp.text or "").strip()
        except Exception as e:
            if _is_rate_limit_error(e):
                parsed = _parse_retry_delay_seconds(e)
                delay = parsed if parsed is not None else 60.0
                self._rate_limit_until = max(self._rate_limit_until, now + delay)
                self.last_call_rate_limited = True
                print(
                    f"Gemini rate limited (quota/429) — backing off ~{delay:.0f}s. "
                    f"See https://ai.google.dev/gemini-api/docs/rate-limits ; "
                    f"try --gemini-model gemini-2.5-flash-lite or enable billing.",
                    flush=True,
                )
                return ""
            if _is_model_not_found_error(e):
                self.last_call_rate_limited = True
                self._rate_limit_until = max(self._rate_limit_until, now + 15.0)
                print(
                    "Gemini model id not available on this API (404). "
                    "Older names like gemini-1.5-flash are often removed; use a current id, e.g.\n"
                    "  --gemini-model gemini-2.5-flash\n"
                    "  --gemini-model gemini-2.5-flash-lite\n"
                    "List: https://ai.google.dev/gemini-api/docs/models",
                    flush=True,
                )
                return ""
            print(f"Gemini request failed: {e}")
            return ""

    def identify(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        self.last_call_rate_limited = False
        if time.monotonic() < self._rate_limit_until:
            self.last_call_rate_limited = True
            return "unknown", 0.0
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
            return "unknown", 0.0
        small = _resize_for_api(self._prep_bgr_for_vision(crop_bgr), max_side=self._max_identify)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        text = self._generate(self._crop_instruction, pil)
        name = _parse_crop_response(text, self.names)
        conf = 0.88 if name != "unknown" else 0.0
        return name, conf

    def identify_scene(self, frame_bgr: np.ndarray, crop_bgr: np.ndarray | None) -> tuple[str, float]:
        """
        Motion-triggered: ask if any cat is present and which name.
        Prefers YOLO crop when available; otherwise full frame.
        """
        self.last_call_rate_limited = False
        if time.monotonic() < self._rate_limit_until:
            self.last_call_rate_limited = True
            return "unknown", 0.0
        hint = ""
        if (
            crop_bgr is not None
            and crop_bgr.size > 0
            and crop_bgr.shape[0] >= 48
            and crop_bgr.shape[1] >= 48
        ):
            img = _resize_for_api(self._prep_bgr_for_vision(crop_bgr), max_side=self._max_scene_crop)
            hint = "\n(Context: image is a zoomed crop where a cat-sized object was detected.)"
        else:
            img = _resize_for_api(self._prep_bgr_for_vision(frame_bgr), max_side=self._max_scene_full)
            hint = "\n(Context: full webcam frame — look anywhere for a cat.)"

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        text = self._generate(self._scene_instruction + hint, pil)
        name = _parse_scene_response(text, self.names)
        if name == "none":
            return "none", 0.0
        if name == "unknown":
            return "unknown", 0.0
        return name, 0.9


def resolve_gemini_api_key() -> tuple[str | None, str | None]:
    """Return (key, env_var_name). GEMINI_API_KEY wins over GOOGLE_API_KEY."""
    g = os.environ.get("GEMINI_API_KEY")
    if g and g.strip():
        return g.strip(), "GEMINI_API_KEY"
    o = os.environ.get("GOOGLE_API_KEY")
    if o and o.strip():
        return o.strip(), "GOOGLE_API_KEY"
    return None, None


def default_api_key() -> str | None:
    k, _ = resolve_gemini_api_key()
    return k


def gemini_api_key_hint(key: str) -> str:
    """Non-secret fingerprint so you can confirm which key the process loaded."""
    k = key.strip()
    if len(k) < 8:
        return "key length unexpected"
    return f"ends …{k[-4:]}"
