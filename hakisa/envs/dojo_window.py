"""DanmakuDojo dressed up as a *real game window*.

It is simultaneously an :class:`~hakisa.thalamus.control.InputBackend` (it receives key/button/mouse
events exactly like an OS would deliver them) and a :class:`~hakisa.thalamus.capture.FrameSource`
(``grab()`` advances the game one tick with the currently held inputs and returns the 544x448 screen
with its HUD).  The whole real-game stack — Controller -> game -> capture -> crop/resize -> HUD
probes -> reward — can therefore be exercised without Windows, Steam or a display.
"""

from __future__ import annotations

import numpy as np

from hakisa.actions import InputTimeline
from hakisa.envs.dojo import DanmakuDojo
from hakisa.types import InputEvent, InputSnapshot, Region


class DojoVirtualWindow:
    def __init__(self, scheme: str = "keyboard", seed: int = 0, difficulty: float = 1.0, origin: tuple[int, int] = (0, 0)):
        self.env = DanmakuDojo(num_envs=1, scheme=scheme, seed=seed, difficulty=difficulty)
        self.region = Region(origin[0], origin[1], 544, 448)
        self.space = self.env.space
        crop = self.env.cfg.obs.crop or (0, 0, 544, 448)
        self.play_region = Region(origin[0] + crop[0], origin[1] + crop[1], crop[2], crop[3])
        self.keys: set[str] = set()
        self.buttons: set[str] = set()
        self.cursor = (self.play_region.left + self.play_region.width // 2, self.play_region.bottom - 40)
        self._edges: list[InputEvent] = []
        self.ticks = 0

    # -- InputBackend ---------------------------------------------------------------------------
    def key_down(self, key: str) -> None:
        self.keys.add(key)
        self._edges.append(InputEvent(0.0, "key", key, True, *self.cursor))

    def key_up(self, key: str) -> None:
        self.keys.discard(key)

    def button_down(self, button: str) -> None:
        self.buttons.add(button)
        self._edges.append(InputEvent(0.0, "button", button, True, *self.cursor))

    def button_up(self, button: str) -> None:
        self.buttons.discard(button)

    def move_to(self, x: int, y: int) -> None:
        self.cursor = (int(x), int(y))

    # -- FrameSource ----------------------------------------------------------------------------
    def current_action(self) -> dict[str, int]:
        snap = InputSnapshot(0.0, frozenset(self.keys), frozenset(self.buttons), self.cursor)
        timeline = InputTimeline([snap], self._edges)
        saved = self.space.lookahead_s
        self.space.lookahead_s = 0.0
        try:
            return self.space.labels_from_timeline(timeline, 0.0, self.play_region)
        finally:
            self.space.lookahead_s = saved

    def grab(self) -> np.ndarray:
        action = self.current_action()
        self._edges.clear()
        self.env.step({k: np.array([v]) for k, v in action.items()})
        self.ticks += 1
        return self.env.render_screen(0)

    def close(self) -> None:
        pass
