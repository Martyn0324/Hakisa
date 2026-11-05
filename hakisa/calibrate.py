"""Calibration helper: tell Hakisa where the playfield and the HUD are (once per game / resolution).

Interactive part (needs a matplotlib window): click two corners per region; click one icon pixel for
colour-based counters; type the number currently shown for digit templates.  The result is printed
and saved as a YAML snippet to paste into ``hakisa/games/<game>.yaml``.

The non-interactive helpers (glyph cutting, unit-pixel estimation) are unit-tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from hakisa.limbic.probes import _crop, _gray, _segments


def glyphs_from_region(img: np.ndarray, region, ink_threshold: float = 128.0, invert: bool = False) -> list[np.ndarray]:
    """Cuts the glyphs of a number displayed in ``region`` (same segmentation as the digit probe)."""
    patch = _crop(img, region)
    g = _gray(patch)
    ink = (g < ink_threshold) if invert else (g >= ink_threshold)
    out = []
    for a, b in _segments(ink.any(axis=0)):
        rows = np.nonzero(ink[:, a:b].any(1))[0]
        out.append(patch[rows[0] : rows[-1] + 1, a:b])
    return out


def save_digit_templates(img: np.ndarray, region, shown: str, out_dir: Path, ink_threshold: float = 128.0) -> list[str]:
    """Saves one template per character of ``shown`` (the number the user reads on screen)."""
    glyphs = glyphs_from_region(img, region, ink_threshold)
    if len(glyphs) != len(shown):
        raise ValueError(f"found {len(glyphs)} glyphs but the number has {len(shown)} digits; adjust the region/threshold")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for d, g in zip(shown, glyphs):
        Image.fromarray(g).save(out_dir / f"{d}.png")
        saved.append(d)
    return saved


def unit_pixels(img: np.ndarray, region, color, tol: int, icons_visible: int) -> float:
    patch = _crop(img, region)[..., :3].astype(np.int16)
    count = int((np.abs(patch - np.asarray(color, np.int16)) <= tol).all(axis=-1).sum())
    return count / max(icons_visible, 1)


def calibrate(cfg, screenshot: str | None = None, out_dir: Path = Path("calibration"), digits: bool = False) -> dict[str, Any]:  # pragma: no cover
    import matplotlib.pyplot as plt

    from hakisa.thalamus.capture import make_dpi_aware, make_source, resolve_capture_region

    out_dir.mkdir(parents=True, exist_ok=True)
    if screenshot:
        img = np.asarray(Image.open(screenshot).convert("RGB"))
    else:
        make_dpi_aware()
        import time

        region = resolve_capture_region(cfg)
        print(f"Capturing {region} in 3 s — show the game with the HUD visible...")
        time.sleep(3)
        src = make_source(cfg.capture.backend, region)
        img = src.grab()
        src.close()
    shot_path = out_dir / f"{cfg.name}_screenshot.png"
    Image.fromarray(img).save(shot_path)
    print(f"screenshot saved to {shot_path} ({img.shape[1]}x{img.shape[0]})")

    def pick_box(title: str) -> list[int]:
        plt.figure(figsize=(12, 7))
        plt.imshow(img)
        plt.title(title + " — click TOP-LEFT then BOTTOM-RIGHT")
        pts = plt.ginput(2, timeout=0)
        plt.close()
        (x0, y0), (x1, y1) = pts
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        return [x0, y0, x1 - x0, y1 - y0]

    def pick_pixel(title: str) -> list[int]:
        plt.figure(figsize=(12, 7))
        plt.imshow(img)
        plt.title(title)
        (x, y), = plt.ginput(1, timeout=0)
        plt.close()
        return [int(v) for v in img[int(y), int(x), :3]]

    result: dict[str, Any] = {"obs": {"crop": pick_box("Effective window (the playfield the model should see)")}, "probes": []}
    for spec in cfg.probes:
        spec = dict(spec)
        spec["region"] = pick_box(f"Probe '{spec['name']}' ({spec['type']}) region")
        if spec["type"] == "pixel_count":
            spec["color"] = pick_pixel(f"Click one pixel of a '{spec['name']}' icon")
            n = int(input(f"How many '{spec['name']}' icons are visible now? "))
            spec["unit_pixels"] = round(unit_pixels(img, spec["region"], spec["color"], int(spec.get("tol", 30)), n), 1)
        elif spec["type"] == "digits" and digits:
            shown = input(f"Type the number currently shown by '{spec['name']}': ").strip()
            tdir = out_dir / f"{cfg.name}_digits"
            save_digit_templates(img, spec["region"], shown, tdir)
            have = sorted(p.stem for p in tdir.glob("*.png"))
            print(f"digit templates so far: {have} (repeat with other screenshots until 0-9 exist)")
            spec["templates"] = str(tdir)
        elif spec["type"] in ("template", "locate"):
            box = pick_box(f"Crop the reference image for '{spec['name']}'")
            tpl_path = out_dir / f"{cfg.name}_{spec['name']}.png"
            Image.fromarray(_crop(img, box)).save(tpl_path)
            spec["template"] = str(tpl_path)
        spec["calibrated"] = True
        result["probes"].append(spec)
    snippet = yaml.safe_dump(result, sort_keys=False)
    (out_dir / f"{cfg.name}_calibration.yaml").write_text(snippet)
    print("\nPaste into the game YAML (obs.crop and probes):\n")
    print(snippet)
    return result
