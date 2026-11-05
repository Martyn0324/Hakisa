"""Small utilities: seeding, devices, fixed-rate loops, image resizing, JSON logging."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

try:  # OpenCV is optional: fastest resize, but Pillow works everywhere
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - depends on the machine
    cv2 = None
    _HAS_CV2 = False

from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover
        pass


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resize_rgb(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Area-style downscale of an (H, W, C) uint8 image to (height, width, C)."""
    if img.shape[0] == height and img.shape[1] == width:
        return img
    if _HAS_CV2:
        interp = cv2.INTER_AREA if (img.shape[0] > height or img.shape[1] > width) else cv2.INTER_LINEAR
        out = cv2.resize(img, (width, height), interpolation=interp)
        return out if out.ndim == 3 else out[:, :, None]
    pil = Image.fromarray(img)
    return np.asarray(pil.resize((width, height), Image.BOX if img.shape[0] > height else Image.BILINEAR))


def crop(img: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return img
    x, y, w, h = box
    return img[y : y + h, x : x + w]


class RateKeeper:
    """Keeps a loop at a fixed frequency and reports how late each tick was."""

    def __init__(self, hz: float, clock=time.perf_counter, sleep=time.sleep):
        self.period = 1.0 / float(hz)
        self.clock = clock
        self.sleep = sleep
        self.next_t = clock()
        self.late = 0.0

    def wait(self) -> float:
        self.next_t += self.period
        now = self.clock()
        delay = self.next_t - now
        if delay > 0:
            self.sleep(delay)
            self.late = 0.0
        else:
            self.late = -delay
            if self.late > 5 * self.period:  # we fell far behind: re-anchor instead of bursting
                self.next_t = now
        return self.late


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in record.items()}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")
