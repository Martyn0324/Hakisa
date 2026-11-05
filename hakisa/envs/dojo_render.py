"""Rendering helpers for DanmakuDojo: vectorised disc splatting, a tiny bitmap font and the HUD."""

from __future__ import annotations

import numpy as np

from hakisa.limbic.probes import register_builtin_templates

# 5x7 bitmap font (digits + the few letters the HUD needs)
_FONT_ROWS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    " ": ["00000"] * 7,
}
FONT = {ch: np.array([[c == "1" for c in row] for row in rows], dtype=bool) for ch, rows in _FONT_ROWS.items()}


def draw_text(img: np.ndarray, text: str, x: int, y: int, color, scale: int = 2, spacing: int = 1) -> None:
    for ch in text.upper():
        glyph = FONT.get(ch)
        if glyph is None:
            x += (5 + spacing) * scale
            continue
        big = np.kron(glyph, np.ones((scale, scale), dtype=bool))
        h, w = big.shape
        region = img[y : y + h, x : x + w]
        region[big[: region.shape[0], : region.shape[1]]] = color
        x += (5 + spacing) * scale


def _dojo_digit_templates() -> dict[str, np.ndarray]:
    return {d: np.kron(FONT[d], np.ones((2, 2), dtype=bool)) for d in "0123456789"}


register_builtin_templates("dojo", _dojo_digit_templates)


def disc_offsets(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    keep = xs**2 + ys**2 <= radius_px**2 + 0.25
    return np.stack([ys[keep], xs[keep]], axis=1)


def splat(frames: np.ndarray, env_idx: np.ndarray, ys: np.ndarray, xs: np.ndarray, colors: np.ndarray, offsets: np.ndarray) -> None:
    """Draw discs into ``frames`` (N, H, W, 3) at integer centres; ``colors`` is (M, 3) uint8."""
    if len(env_idx) == 0:
        return
    n, h, w, _ = frames.shape
    flat = frames.reshape(-1, 3)
    for dy, dx in offsets:
        yy = ys + dy
        xx = xs + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        if not ok.any():
            continue
        idx = (env_idx[ok] * h + yy[ok]) * w + xx[ok]
        flat[idx] = colors[ok]


def fill_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color) -> None:
    img[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)] = color
