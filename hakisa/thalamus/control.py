"""Motor relay: turns a :class:`DesiredInput` into OS input events.

The :class:`Controller` is *diff-based*: only keys whose state changes are pressed/released, so the
game sees exactly what a human holding keys would produce.  Taps are non-blocking: pressed at this
tick and released at the next one (games poll input once per frame, so a 0 ms tap can be missed).

Safety: ``release_all()`` releases everything Hakisa holds; ``Agent`` calls it on exit and on the
kill switch (F12 by default).
"""

from __future__ import annotations

import ctypes
import sys
import time
from typing import Callable, Protocol

from hakisa.thalamus.keys import normalize_key_name, scan_code
from hakisa.types import DesiredInput, Tap

_MODIFIER_ORDER = {"ctrl": 0, "shift": 1, "alt": 2}


class InputBackend(Protocol):
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def button_down(self, button: str) -> None: ...
    def button_up(self, button: str) -> None: ...
    def move_to(self, x: int, y: int) -> None: ...


class LoggingBackend:
    """Records calls instead of touching the OS (tests, dry runs, the simulator)."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[str, object]] = []
        self.fail_on = fail_on

    def _rec(self, kind: str, arg: object) -> None:
        if self.fail_on is not None and kind == self.fail_on:
            self.fail_on = None
            raise RuntimeError(f"simulated backend failure on {kind}")
        self.calls.append((kind, arg))

    def key_down(self, key: str) -> None:
        self._rec("key_down", key)

    def key_up(self, key: str) -> None:
        self._rec("key_up", key)

    def button_down(self, button: str) -> None:
        self._rec("button_down", button)

    def button_up(self, button: str) -> None:
        self._rec("button_up", button)

    def move_to(self, x: int, y: int) -> None:
        self._rec("move_to", (int(x), int(y)))


# ---------------------------------------------------------------------------------------------
# Windows SendInput (scan codes).  Structures use fixed-width types so their layout matches the
# x64 Windows ABI on any platform (lets us unit-test the packing on Linux).
# ---------------------------------------------------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

_BUTTON_FLAGS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_uint64),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_uint64),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_uint32), ("wParamL", ctypes.c_uint16), ("wParamH", ctypes.c_uint16)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("u", _INPUTUNION)]


def make_key_input(key: str, up: bool) -> INPUT:
    code, extended = scan_code(key)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.u.ki = KEYBDINPUT(0, code, flags, 0, 0)
    return inp


def make_button_input(button: str, up: bool) -> INPUT:
    down_flag, up_flag = _BUTTON_FLAGS[button]
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.u.mi = MOUSEINPUT(0, 0, 0, up_flag if up else down_flag, 0, 0)
    return inp


class WindowsSendInputBackend:  # pragma: no cover - needs Windows
    """Injects scan-code keyboard events and absolute mouse moves with ``SendInput``."""

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("WindowsSendInputBackend only works on Windows")
        self.user32 = ctypes.windll.user32
        self.user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
        self._refresh_virtual_screen()

    def _refresh_virtual_screen(self) -> None:
        gsm = self.user32.GetSystemMetrics
        self.vx, self.vy = gsm(76), gsm(77)  # SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN
        self.vw, self.vh = max(gsm(78), 1), max(gsm(79), 1)  # SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN

    def _send(self, inp: INPUT) -> None:
        if self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise OSError(ctypes.get_last_error() or "SendInput failed (blocked by UIPI? run as admin)")

    def key_down(self, key: str) -> None:
        self._send(make_key_input(key, up=False))

    def key_up(self, key: str) -> None:
        self._send(make_key_input(key, up=True))

    def button_down(self, button: str) -> None:
        self._send(make_button_input(button, up=False))

    def button_up(self, button: str) -> None:
        self._send(make_button_input(button, up=True))

    def move_to(self, x: int, y: int) -> None:
        nx = int(round((x - self.vx) * 65535 / max(self.vw - 1, 1)))
        ny = int(round((y - self.vy) * 65535 / max(self.vh - 1, 1)))
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.u.mi = MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, 0)
        try:
            self._send(inp)
        except OSError:
            self.user32.SetCursorPos(int(x), int(y))


class PynputBackend:  # pragma: no cover - needs a display
    """Cross-platform fallback (uses virtual-key events; some DirectInput games ignore them)."""

    def __init__(self):
        from pynput import keyboard, mouse

        self._kb = keyboard.Controller()
        self._mouse = mouse.Controller()
        self._Key = keyboard.Key
        self._Button = mouse.Button

    def _key(self, name: str):
        name = normalize_key_name(name)
        if len(name) == 1:
            return name
        alias = {"ctrl": "ctrl_l", "alt": "alt_l"}.get(name, name)
        return getattr(self._Key, alias)

    def key_down(self, key: str) -> None:
        self._kb.press(self._key(key))

    def key_up(self, key: str) -> None:
        self._kb.release(self._key(key))

    def button_down(self, button: str) -> None:
        self._mouse.press(getattr(self._Button, button))

    def button_up(self, button: str) -> None:
        self._mouse.release(getattr(self._Button, button))

    def move_to(self, x: int, y: int) -> None:
        self._mouse.position = (int(x), int(y))


def make_backend(name: str = "auto") -> InputBackend:  # pragma: no cover - platform dependent
    if name == "log":
        return LoggingBackend()
    if name in ("auto", "sendinput") and sys.platform == "win32":
        return WindowsSendInputBackend()
    if name == "sendinput":
        raise RuntimeError("sendinput backend requires Windows")
    return PynputBackend()


def _key_order(k: str) -> tuple[int, str]:
    return (_MODIFIER_ORDER.get(k, 9), k)


class Controller:
    def __init__(self, backend: InputBackend, sleep: Callable[[float], None] = time.sleep, tap_hold_s: float = 0.0):
        self.backend = backend
        self.sleep = sleep
        self.tap_hold_s = tap_hold_s  # >0 = blocking taps (press, wait, release in the same tick)
        self.held_keys: set[str] = set()
        self.held_buttons: set[str] = set()
        self.cursor: tuple[int, int] | None = None
        self._pending_keys: list[str] = []  # tap keys to release at the next apply()
        self._pending_buttons: list[str] = []

    # -- low level --------------------------------------------------------------------------
    def _press_key(self, k: str) -> None:
        self.backend.key_down(k)
        self.held_keys.add(k)

    def _release_key(self, k: str) -> None:
        self.held_keys.discard(k)
        self.backend.key_up(k)

    def _press_button(self, b: str) -> None:
        self.backend.button_down(b)
        self.held_buttons.add(b)

    def _release_button(self, b: str) -> None:
        self.held_buttons.discard(b)
        self.backend.button_up(b)

    def move_to(self, xy: tuple[int, int]) -> None:
        if self.cursor != xy:
            self.backend.move_to(*xy)
            self.cursor = xy

    # -- tick ---------------------------------------------------------------------------------
    def _flush_pending(self, desired: DesiredInput) -> None:
        for k in reversed(self._pending_keys):
            if k not in desired.keys and k in self.held_keys:
                self._release_key(k)
        for b in self._pending_buttons:
            if b not in desired.buttons and b in self.held_buttons:
                self._release_button(b)
        self._pending_keys.clear()
        self._pending_buttons.clear()

    def tap(self, tap: Tap, desired_keys: frozenset[str] = frozenset()) -> None:
        if tap.cursor is not None:
            self.move_to(tap.cursor)
        pressed: list[str] = []
        for k in tap.keys:
            if k in self.held_keys and k not in desired_keys:
                self._release_key(k)  # re-press to create a fresh edge
            if k not in self.held_keys:
                self._press_key(k)
                pressed.append(k)
        if tap.button is not None:
            if tap.button in self.held_buttons:
                self._release_button(tap.button)
            self._press_button(tap.button)
        if self.tap_hold_s > 0:
            self.sleep(self.tap_hold_s)
            if tap.button is not None:
                self._release_button(tap.button)
            for k in reversed(pressed):
                self._release_key(k)
        else:
            self._pending_keys.extend(k for k in pressed if k not in desired_keys)
            if tap.button is not None:
                self._pending_buttons.append(tap.button)

    def apply(self, desired: DesiredInput) -> None:
        self._flush_pending(desired)
        if desired.cursor is not None:
            self.move_to(desired.cursor)
        for k in sorted(self.held_keys - set(desired.keys), key=_key_order, reverse=True):
            self._release_key(k)
        for b in sorted(self.held_buttons - set(desired.buttons)):
            self._release_button(b)
        for k in sorted(set(desired.keys) - self.held_keys, key=_key_order):
            self._press_key(k)
        for b in sorted(set(desired.buttons) - self.held_buttons):
            self._press_button(b)
        multiple = len(desired.taps) > 1
        for i, tap in enumerate(desired.taps):
            if multiple and i < len(desired.taps) - 1:
                # several commands in one tick: make the earlier ones complete taps
                saved, self.tap_hold_s = self.tap_hold_s, max(self.tap_hold_s, 0.02)
                try:
                    self.tap(tap, desired.keys)
                finally:
                    self.tap_hold_s = saved
            else:
                self.tap(tap, desired.keys)

    def release_all(self) -> list[Exception]:
        """Release everything; never raises (returns the errors it swallowed)."""
        errors: list[Exception] = []
        for k in sorted(set(self.held_keys) | set(self._pending_keys), key=_key_order, reverse=True):
            try:
                self.backend.key_up(k)
            except Exception as exc:  # keep going: never leave keys stuck
                errors.append(exc)
        for b in sorted(set(self.held_buttons) | set(self._pending_buttons)):
            try:
                self.backend.button_up(b)
            except Exception as exc:
                errors.append(exc)
        self.held_keys.clear()
        self.held_buttons.clear()
        self._pending_keys.clear()
        self._pending_buttons.clear()
        return errors
