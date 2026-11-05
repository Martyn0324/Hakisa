"""Sensory relay: screen capture backends and window/region helpers.

Backends
--------
``dxcam``  Desktop Duplication API (Windows, fastest; https://github.com/ra1nty/DXcam)
``mss``    portable GDI/X11 capture (slower, works everywhere)
``sim`` / ``CallableSource``  frames produced by Python (DanmakuDojo, tests)

On Windows call :func:`make_dpi_aware` before reading any coordinate, otherwise scaled displays
(125 %, 150 %) make capture regions and mouse positions disagree.
"""

from __future__ import annotations

import sys
from typing import Callable, Protocol

import numpy as np

from hakisa.types import Region


class FrameSource(Protocol):
    region: Region

    def grab(self) -> np.ndarray: ...  # (H, W, 3) uint8 RGB

    def close(self) -> None: ...


class CallableSource:
    """Wraps any ``fn() -> (H, W, 3) uint8`` (simulator, pygame window surface, tests)."""

    def __init__(self, fn: Callable[[], np.ndarray], region: Region | None = None):
        self.fn = fn
        first = fn()
        self.region = region or Region(0, 0, first.shape[1], first.shape[0])
        self._first: np.ndarray | None = first

    def grab(self) -> np.ndarray:
        if self._first is not None:
            out, self._first = self._first, None
            return out
        return self.fn()

    def close(self) -> None:
        pass


class MssSource:  # pragma: no cover - needs a display
    def __init__(self, region: Region):
        import mss

        self.region = region
        self._sct = mss.mss()
        self._monitor = {"left": region.left, "top": region.top, "width": region.width, "height": region.height}

    def grab(self) -> np.ndarray:
        shot = self._sct.grab(self._monitor)
        bgra = np.asarray(shot)  # (H, W, 4) BGRA
        return np.ascontiguousarray(bgra[:, :, [2, 1, 0]])

    def close(self) -> None:
        self._sct.close()


class DxcamSource:  # pragma: no cover - needs Windows
    """dxcam in single-grab mode; ``grab`` returns the last frame when the screen did not change."""

    def __init__(self, region: Region, output_idx: int = 0):
        import dxcam

        self.region = region
        self._cam = dxcam.create(output_idx=output_idx, output_color="RGB")
        self._box = (region.left, region.top, region.right, region.bottom)
        self._last: np.ndarray | None = None

    def grab(self) -> np.ndarray:
        frame = self._cam.grab(region=self._box)
        if frame is None:  # no new frame since the last grab
            if self._last is None:
                for _ in range(100):
                    frame = self._cam.grab(region=self._box)
                    if frame is not None:
                        break
                if frame is None:
                    raise RuntimeError("dxcam returned no frame; is the region on the selected monitor?")
            else:
                return self._last
        self._last = np.asarray(frame)
        return self._last

    def close(self) -> None:
        try:
            self._cam.release()
        except Exception:
            pass


def make_dpi_aware() -> None:
    """Make this process per-monitor DPI aware (Windows). No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:  # pragma: no cover - Windows only
        ctypes = __import__("ctypes")
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
            return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def find_window_client_region(title_substring: str) -> Region:  # pragma: no cover - Windows only
    """Client area (without borders/title bar) of the first visible window whose title contains the text."""
    if sys.platform != "win32":
        raise RuntimeError("window lookup by title is only implemented for Windows")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    matches: list[int] = []
    needle = title_substring.lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if needle in buf.value.lower():
            matches.append(hwnd)
        return True

    user32.EnumWindows(_enum, 0)
    if not matches:
        raise RuntimeError(f"no visible window with {title_substring!r} in its title")
    hwnd = matches[0]
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return Region(pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top)


def primary_screen_region() -> Region:  # pragma: no cover - needs a display
    if sys.platform == "win32":
        import ctypes

        u = ctypes.windll.user32
        return Region(0, 0, u.GetSystemMetrics(0), u.GetSystemMetrics(1))
    import mss

    with mss.mss() as sct:
        mon = sct.monitors[1]
        return Region(mon["left"], mon["top"], mon["width"], mon["height"])


def resolve_capture_region(cfg, window_lookup: Callable[[str], Region] | None = None) -> Region:
    """Absolute screen region to capture for a game config."""
    cap = cfg.capture
    if cap.window_title:
        client = (window_lookup or find_window_client_region)(cap.window_title)
        if cap.region:
            x, y, w, h = cap.region
            return Region(client.left + x, client.top + y, w, h)
        return client
    if cap.region:
        return Region(*cap.region)
    return primary_screen_region()


def sub_regions(cfg, capture_region: Region) -> tuple[Region, Region | None]:
    """Absolute regions of the model's effective window (obs crop) and of the minimap."""
    if cfg.obs.crop:
        x, y, w, h = cfg.obs.crop
        obs_region = Region(capture_region.left + x, capture_region.top + y, w, h)
    else:
        obs_region = capture_region
    minimap_region = None
    if cfg.obs.minimap is not None:
        x, y, w, h = cfg.obs.minimap.region
        minimap_region = Region(capture_region.left + x, capture_region.top + y, w, h)
    return obs_region, minimap_region


def make_source(backend: str, region: Region) -> FrameSource:  # pragma: no cover - platform dependent
    if backend == "auto":
        if sys.platform == "win32":
            try:
                return DxcamSource(region)
            except Exception:
                pass
        return MssSource(region)
    if backend == "dxcam":
        return DxcamSource(region)
    if backend == "mss":
        return MssSource(region)
    raise ValueError(f"unknown capture backend {backend!r}")
