"""Afferent relay for *human* input: global keyboard/mouse hooks (pynput) -> held state + event log.

Unlike Hakisa 1.0's keylogger (which blocked until a key was pressed and kept only that key), the
tracker runs in the background and the recorder samples its full state at every capture tick, so
"doing nothing" and "holding a key" are recorded too.

Windows note: hooks do not see input sent to a window running as administrator unless Python runs
as administrator as well (UIPI).
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from hakisa.thalamus.keys import normalize_pynput_key
from hakisa.types import InputEvent, InputSnapshot


class InputTracker:
    def __init__(self, clock: Callable[[], float] = time.perf_counter, move_hz: float = 60.0):
        self.clock = clock
        self.move_period = 1.0 / move_hz if move_hz > 0 else 0.0
        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._buttons: set[str] = set()
        self._cursor: tuple[int, int] = (0, 0)
        self._events: list[InputEvent] = []
        self._last_move_t = -1e9
        self._hotkeys: dict[str, Callable[[], None]] = {}
        self._listeners: list[object] = []

    # -- state updates (thread-safe) -------------------------------------------------------------
    def feed(self, ev: InputEvent) -> None:
        hotkey: Callable[[], None] | None = None
        with self._lock:
            if ev.x is not None and ev.y is not None:
                self._cursor = (int(ev.x), int(ev.y))
            if ev.kind == "move":
                if ev.t - self._last_move_t < self.move_period:
                    return
                self._last_move_t = ev.t
            elif ev.kind == "key":
                if ev.pressed:
                    if ev.name in self._keys:  # OS auto-repeat: not a new press
                        return
                    self._keys.add(ev.name)
                    hotkey = self._hotkeys.get(ev.name)
                else:
                    self._keys.discard(ev.name)
            elif ev.kind == "button":
                if ev.pressed:
                    self._buttons.add(ev.name)
                else:
                    self._buttons.discard(ev.name)
            self._events.append(ev)
        if hotkey is not None:
            hotkey()

    def snapshot(self) -> InputSnapshot:
        with self._lock:
            return InputSnapshot(self.clock(), frozenset(self._keys), frozenset(self._buttons), self._cursor)

    def drain_events(self) -> list[InputEvent]:
        with self._lock:
            out, self._events = self._events, []
        return out

    def on_hotkey(self, key: str, callback: Callable[[], None]) -> None:
        """Call ``callback`` (from the hook thread) whenever ``key`` is pressed."""
        self._hotkeys[key.lower()] = callback

    # -- OS hooks -------------------------------------------------------------------------------
    def start(self) -> "InputTracker":  # pragma: no cover - needs a display / OS hooks
        from pynput import keyboard, mouse

        def on_press(key, *_):
            name = normalize_pynput_key(key)
            if name:
                self.feed(InputEvent(self.clock(), "key", name, True))

        def on_release(key, *_):
            name = normalize_pynput_key(key)
            if name:
                self.feed(InputEvent(self.clock(), "key", name, False))

        def on_click(x, y, button, pressed, *_):
            self.feed(InputEvent(self.clock(), "button", button.name, bool(pressed), int(x), int(y)))

        def on_move(x, y, *_):
            self.feed(InputEvent(self.clock(), "move", "", True, int(x), int(y)))

        def on_scroll(x, y, dx, dy, *_):
            self.feed(InputEvent(self.clock(), "scroll", "up" if dy > 0 else "down", True, int(x), int(y)))

        kl = keyboard.Listener(on_press=on_press, on_release=on_release)
        ml = mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll)
        kl.start()
        ml.start()
        try:
            ctrl = mouse.Controller()
            x, y = ctrl.position
            self._cursor = (int(x), int(y))
        except Exception:
            pass
        self._listeners = [kl, ml]
        return self

    def stop(self) -> None:  # pragma: no cover - needs OS hooks
        for listener in self._listeners:
            try:
                listener.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._listeners = []
