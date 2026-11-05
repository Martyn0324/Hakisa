"""Play DanmakuDojo yourself (keyboard: WASD + Shift + Z + X; mouse: move, hold left, right = bomb).

Optionally records your play with the *same* recorder used for real games, so you can test the whole
Study pipeline on your own demonstrations.  Needs ``pip install pygame``.
"""

from __future__ import annotations

import time

from hakisa.config import load_game_config
from hakisa.envs.dojo_window import DojoVirtualWindow
from hakisa.record import Recorder
from hakisa.thalamus.tracker import InputTracker
from hakisa.types import InputEvent


def play_dojo(scheme: str = "keyboard", difficulty: float = 1.0, record_dir: str | None = None, fps: int = 20) -> None:  # pragma: no cover
    import pygame

    cfg = load_game_config("dojo_keyboard" if scheme == "keyboard" else "dojo_mouse")
    window = DojoVirtualWindow(scheme, difficulty=difficulty)
    pygame.init()
    screen = pygame.display.set_mode((544, 448))
    pygame.display.set_caption("DanmakuDojo")
    clock = pygame.time.Clock()
    tracker = InputTracker()
    recorder = Recorder(cfg, record_dir, window.region, tracker, source="human") if record_dir else None
    names = {pygame.K_w: "w", pygame.K_a: "a", pygame.K_s: "s", pygame.K_d: "d", pygame.K_z: "z", pygame.K_x: "x",
             pygame.K_LSHIFT: "shift", pygame.K_RSHIFT: "shift"}
    buttons = {1: "left", 3: "right"}
    running = True
    try:
        while running:
            now = time.perf_counter()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
                elif ev.type in (pygame.KEYDOWN, pygame.KEYUP) and ev.key in names:
                    down = ev.type == pygame.KEYDOWN
                    (window.key_down if down else window.key_up)(names[ev.key])
                    tracker.feed(InputEvent(now, "key", names[ev.key], down, *window.cursor))
                elif ev.type == pygame.MOUSEMOTION:
                    window.move_to(*ev.pos)
                    tracker.feed(InputEvent(now, "move", "", True, *ev.pos))
                elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and ev.button in buttons:
                    down = ev.type == pygame.MOUSEBUTTONDOWN
                    window.move_to(*ev.pos)
                    (window.button_down if down else window.button_up)(buttons[ev.button])
                    tracker.feed(InputEvent(now, "button", buttons[ev.button], down, *ev.pos))
            frame = window.grab()
            if recorder is not None:
                recorder.tick(frame)
            screen.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
            pygame.display.flip()
            clock.tick(fps)
    finally:
        if recorder is not None and recorder.ticks:
            print("saved", recorder.close())
        pygame.quit()
