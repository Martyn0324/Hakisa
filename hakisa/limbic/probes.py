"""HUD probes: read lives / bombs / score / game-over from the screen *without* trained models.

Hakisa 1.0 trained three ResNet "LimbicSystem" regressors on hand-labelled crops (the armpit-and-
tendril-destroying part of the project).  These probes replace them with classic, calibrate-once
image processing — no labels, no pre-trained vision:

* ``pixel_count``  count pixels of an icon colour in a region (lives/bombs drawn as icons)
* ``digits``       read a number by template-matching 10 digit glyphs cropped once from a screenshot
* ``template``     is a reference image (e.g. "GAME OVER") visible in a region? -> 0/1
* ``locate``       where is a reference sprite (e.g. the player's ship)? -> (x, y) or None
* ``change``       mean absolute change of a region since the last read

Regions are ``[x, y, w, h]`` in pixels of the captured image.
"""

from __future__ import annotations

import importlib
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _crop(screen: np.ndarray, region) -> np.ndarray:
    x, y, w, h = (int(v) for v in region)
    return screen[y : y + h, x : x + w]


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


class Probe:
    kind = "base"

    def __init__(self, name: str, region, calibrated: bool = True):
        self.name = name
        self.region = tuple(int(v) for v in region)
        self.calibrated = calibrated

    def read(self, screen: np.ndarray) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class PixelCountProbe(Probe):
    kind = "pixel_count"

    def __init__(self, name, region, color, tol: int = 30, unit_pixels: float = 1.0, max_value: int | None = None, calibrated=True):
        super().__init__(name, region, calibrated)
        self.color = np.asarray(color, dtype=np.int16)[:3]
        self.tol = int(tol)
        self.unit = float(unit_pixels)
        self.max_value = max_value

    def count(self, screen: np.ndarray) -> int:
        patch = _crop(screen, self.region)[..., :3].astype(np.int16)
        return int((np.abs(patch - self.color) <= self.tol).all(axis=-1).sum())

    def read(self, screen: np.ndarray) -> int:
        value = int(round(self.count(screen) / max(self.unit, 1e-6)))
        return min(value, self.max_value) if self.max_value is not None else value


def _segments(mask_cols: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, on in enumerate(mask_cols):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask_cols)))
    return segs


class DigitTemplateProbe(Probe):
    """Reads an integer: binarise -> split glyphs on empty columns -> nearest digit template."""

    kind = "digits"

    def __init__(
        self,
        name,
        region,
        templates: dict[str, np.ndarray],
        ink_threshold: float = 128.0,
        min_score: float = 0.7,
        invert: bool = False,
        calibrated=True,
    ):
        super().__init__(name, region, calibrated)
        if not templates:
            raise ValueError("digit probe needs templates for 0-9")
        self.ink_threshold = ink_threshold
        self.min_score = min_score
        self.invert = invert
        self.templates = {d: self._normalise(t) for d, t in templates.items()}

    def _ink(self, img: np.ndarray) -> np.ndarray:
        g = _gray(img) if img.ndim == 3 else img.astype(np.float32)
        return (g < self.ink_threshold) if self.invert else (g >= self.ink_threshold)

    def _normalise(self, glyph: np.ndarray) -> np.ndarray:
        ink = glyph if glyph.dtype == bool else self._ink(glyph)
        rows = np.nonzero(ink.any(1))[0]
        cols = np.nonzero(ink.any(0))[0]
        if len(rows) == 0:
            return np.zeros((10, 6), dtype=np.float32)
        ink = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        img = Image.fromarray(ink.astype(np.uint8) * 255).resize((6, 10), Image.NEAREST)
        return (np.asarray(img) > 127).astype(np.float32)

    def read(self, screen: np.ndarray) -> int | None:
        ink = self._ink(_crop(screen, self.region))
        segs = _segments(ink.any(axis=0))
        if not segs:
            return None
        digits = []
        for a, b in segs:
            glyph = self._normalise(ink[:, a:b])
            best_d, best_s = None, -1.0
            for d, tpl in self.templates.items():
                score = 1.0 - float(np.abs(glyph - tpl).mean())
                if score > best_s:
                    best_d, best_s = d, score
            if best_s < self.min_score:
                return None  # unreadable (animation, overlap): keep the previous value upstream
            digits.append(best_d)
        return int("".join(digits))


def _match_template(region_img: np.ndarray, tpl: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Best normalised cross-correlation and its top-left (x, y) inside the region."""
    a = _gray(region_img)
    t = _gray(tpl)
    if a.shape[0] < t.shape[0] or a.shape[1] < t.shape[1]:
        return 0.0, (0, 0)
    if cv2 is not None:
        res = cv2.matchTemplate(a.astype(np.float32), t.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        return float(maxv), (int(maxloc[0]), int(maxloc[1]))
    win = np.lib.stride_tricks.sliding_window_view(a, t.shape)  # (H', W', th, tw)
    tz = t - t.mean()
    tn = np.sqrt((tz**2).sum()) + 1e-6
    wm = win.mean(axis=(2, 3), keepdims=True)
    wz = win - wm
    num = (wz * tz).sum(axis=(2, 3))
    den = np.sqrt((wz**2).sum(axis=(2, 3))) * tn + 1e-6
    res = num / den
    iy, ix = np.unravel_index(int(np.argmax(res)), res.shape)
    return float(res[iy, ix]), (int(ix), int(iy))


class TemplateProbe(Probe):
    kind = "template"

    def __init__(self, name, region, template: np.ndarray, threshold: float = 0.85, calibrated=True):
        super().__init__(name, region, calibrated)
        self.template = np.asarray(template)
        self.threshold = float(threshold)

    def score(self, screen: np.ndarray) -> tuple[float, tuple[int, int]]:
        return _match_template(_crop(screen, self.region), self.template)

    def read(self, screen: np.ndarray) -> int:
        return int(self.score(screen)[0] >= self.threshold)


class LocateProbe(TemplateProbe):
    """Centre (x, y) of the best template match in captured-image pixels, or None."""

    kind = "locate"

    def read(self, screen: np.ndarray) -> tuple[int, int] | None:  # type: ignore[override]
        s, (x, y) = self.score(screen)
        if s < self.threshold:
            return None
        th, tw = self.template.shape[:2]
        return self.region[0] + x + tw // 2, self.region[1] + y + th // 2


class RegionChangeProbe(Probe):
    kind = "change"

    def __init__(self, name, region, calibrated=True):
        super().__init__(name, region, calibrated)
        self._prev: np.ndarray | None = None

    def read(self, screen: np.ndarray) -> float:
        cur = _gray(_crop(screen, self.region))
        prev, self._prev = self._prev, cur
        return 0.0 if prev is None else float(np.abs(cur - prev).mean())


# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------

_BUILTIN_TEMPLATES: dict[str, Callable[[], dict[str, np.ndarray]]] = {}


def register_builtin_templates(name: str, fn: Callable[[], dict[str, np.ndarray]]) -> None:
    _BUILTIN_TEMPLATES[name] = fn


def load_digit_templates(spec: str, base_dir: Path | None = None) -> dict[str, np.ndarray]:
    if spec.startswith("builtin:"):
        key = spec.split(":", 1)[1]
        if key == "dojo" and "dojo" not in _BUILTIN_TEMPLATES:
            importlib.import_module("hakisa.envs.dojo_render")  # registers the simulator font
        return _BUILTIN_TEMPLATES[key]()
    path = Path(spec)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    out = {}
    for d in "0123456789":
        f = path / f"{d}.png"
        if f.exists():
            out[d] = np.asarray(Image.open(f).convert("RGB"))
    if len(out) < 10:
        raise FileNotFoundError(f"digit templates 0.png..9.png not found in {path}")
    return out


def _load_image(spec: str, base_dir: Path | None) -> np.ndarray:
    path = Path(spec)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return np.asarray(Image.open(path).convert("RGB"))


def build_probe(spec: dict[str, Any], base_dir: Path | None = None) -> Probe:
    kind = spec["type"]
    common = dict(name=spec["name"], region=spec["region"], calibrated=bool(spec.get("calibrated", True)))
    if kind == "pixel_count":
        return PixelCountProbe(color=spec["color"], tol=spec.get("tol", 30), unit_pixels=spec.get("unit_pixels", 1),
                               max_value=spec.get("max_value"), **common)
    if kind == "digits":
        return DigitTemplateProbe(templates=load_digit_templates(spec["templates"], base_dir),
                                  ink_threshold=spec.get("ink_threshold", 128), min_score=spec.get("min_score", 0.7),
                                  invert=spec.get("invert", False), **common)
    if kind == "template":
        return TemplateProbe(template=_load_image(spec["template"], base_dir), threshold=spec.get("threshold", 0.85), **common)
    if kind == "locate":
        return LocateProbe(template=_load_image(spec["template"], base_dir), threshold=spec.get("threshold", 0.7), **common)
    if kind == "change":
        return RegionChangeProbe(**common)
    raise ValueError(f"unknown probe type {kind!r}")


def build_probes(specs: list[dict[str, Any]], base_dir: Path | None = None, require_calibrated: bool = False) -> dict[str, Probe]:
    probes: dict[str, Probe] = {}
    for spec in specs:
        if not spec.get("calibrated", True):
            if require_calibrated:
                raise RuntimeError(
                    f"probe {spec['name']!r} is not calibrated; run `hakisa calibrate` and set calibrated: true"
                )
            continue
        probes[spec["name"]] = build_probe(spec, base_dir)
    return probes


class HudReader:
    """Reads all probes and debounces discrete values (a change must persist ``stable_frames`` reads)."""

    def __init__(self, probes: dict[str, Probe], stable_frames: int = 2):
        self.probes = probes
        self.stable_frames = max(1, int(stable_frames))
        self._history: dict[str, deque] = {k: deque(maxlen=self.stable_frames) for k in probes}
        self._value: dict[str, Any] = {}

    def reset(self) -> None:
        for h in self._history.values():
            h.clear()
        self._value.clear()

    def read(self, screen: np.ndarray) -> dict[str, Any]:
        for name, probe in self.probes.items():
            v = probe.read(screen)
            if isinstance(probe, (RegionChangeProbe, LocateProbe)):
                self._value[name] = v
                continue
            hist = self._history[name]
            hist.append(v)
            if v is None:
                continue
            if name not in self._value or self._value[name] is None:
                self._value[name] = v
            elif len(hist) == hist.maxlen and all(h == v for h in hist):
                self._value[name] = v
        return dict(self._value)
