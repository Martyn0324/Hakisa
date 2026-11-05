"""A real game behind the screen, exposed as a 1-environment vector env (leading batch dim of 1).

Every tick: apply the action (diff-based input injection) -> wait for the next tick (fixed rate) ->
grab the screen -> crop/resize the effective window -> read the HUD -> reward.

Safety hotkeys (``runtime`` section of the game YAML), handled by :class:`Safety`:
F12 kill (releases every key and stops), F9 pause/resume, F8 human takeover, F7 "stage started".
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

from hakisa.config import GameConfig
from hakisa.hippocampus.dataset import cursor_heatmap
from hakisa.limbic.rewards import BulletHellReward
from hakisa.thalamus.capture import sub_regions
from hakisa.thalamus.control import Controller
from hakisa.types import Region, Tap
from hakisa.utils import RateKeeper, crop, resize_rgb


class HakisaStop(Exception):
    """Raised when the kill switch is pressed."""


class Safety:
    def __init__(self, runtime: dict[str, Any] | None = None, controller: Controller | None = None, log_fn=print):
        runtime = runtime or {}
        self.keys = {
            "kill": str(runtime.get("kill_key", "f12")),
            "pause": str(runtime.get("pause_key", "f9")),
            "takeover": str(runtime.get("takeover_key", "f8")),
            "start": str(runtime.get("start_key", "f7")),
        }
        self.kill = threading.Event()
        self.paused = threading.Event()
        self.takeover = threading.Event()
        self.started = threading.Event()
        self.controller = controller
        self.log_fn = log_fn

    def attach(self, tracker) -> "Safety":
        tracker.on_hotkey(self.keys["kill"], self.trigger_kill)
        tracker.on_hotkey(self.keys["pause"], self.toggle_pause)
        tracker.on_hotkey(self.keys["takeover"], self.toggle_takeover)
        tracker.on_hotkey(self.keys["start"], self.started.set)
        return self

    def trigger_kill(self) -> None:
        self.kill.set()
        if self.controller is not None:
            self.controller.release_all()
        self.log_fn(f"[safety] kill switch ({self.keys['kill']}) — releasing all inputs")

    def toggle_pause(self) -> None:
        if self.paused.is_set():
            self.paused.clear()
            self.log_fn("[safety] resumed")
        else:
            self.paused.set()
            self.log_fn("[safety] paused")

    def toggle_takeover(self) -> None:
        if self.takeover.is_set():
            self.takeover.clear()
            self.log_fn("[safety] Hakisa has control again")
        else:
            self.takeover.set()
            self.log_fn("[safety] human takeover — your corrections are being recorded")

    def check(self) -> None:
        if self.kill.is_set():
            raise HakisaStop()


class RealtimeGameEnv:
    def __init__(
        self,
        cfg: GameConfig,
        source,
        controller: Controller,
        capture_region: Region,
        hud=None,
        reward_fn=None,
        safety: Safety | None = None,
        cursor_fn: Callable[[], tuple[int, int]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        log_fn: Callable[[str], None] = print,
    ):
        self.cfg = cfg
        self.space = cfg.action_space
        self.source = source
        self.controller = controller
        self.capture_region = capture_region
        self.obs_region, self.minimap_region = sub_regions(cfg, capture_region)
        self.hud = hud
        self.reward_fn = reward_fn or BulletHellReward(cfg.reward)
        self.safety = safety or Safety(cfg.runtime, controller, log_fn)
        self.cursor_fn = cursor_fn
        self.clock, self.sleep = clock, sleep
        self.log_fn = log_fn
        self.H, self.W, self.stack = cfg.obs.height, cfg.obs.width, cfg.obs.stack
        self.frames: deque = deque(maxlen=self.stack)
        self.mm_frames: deque = deque(maxlen=self.stack)
        self.max_episode_steps = cfg.runtime.get("max_episode_steps")
        self.last_screen: np.ndarray | None = None
        self.last_hud: dict[str, Any] = {}
        self.late_s = 0.0
        self._reset_episode_stats()
        self.rate = RateKeeper(cfg.tick_hz, clock, sleep)

    # -- observation ----------------------------------------------------------------------------
    def grab(self) -> np.ndarray:
        self.last_screen = self.source.grab()
        return self.last_screen

    def _cursor_norm(self) -> tuple[float, float]:
        xy = self.cursor_fn() if self.cursor_fn is not None else self.controller.cursor
        if xy is None:
            return 0.5, 0.5
        return self.obs_region.to_norm(*xy)

    def aux_vector(self) -> np.ndarray | None:
        return None

    def observe(self, screen: np.ndarray, fill: bool = False):
        img = resize_rgb(crop(screen, self.cfg.obs.crop), self.H, self.W)
        for _ in range(self.stack if fill else 1):
            self.frames.append(img)
        obs = np.concatenate([f.transpose(2, 0, 1) for f in self.frames])[None]
        if self.cfg.obs.cursor_channel:
            u, v = self._cursor_norm()
            obs = np.concatenate([obs, cursor_heatmap(self.H, self.W, u, v)[None, None]], axis=1)
        mm = None
        if self.cfg.obs.minimap is not None:
            m = self.cfg.obs.minimap
            mimg = resize_rgb(crop(screen, m.region), m.height, m.width)
            for _ in range(self.stack if fill else 1):
                self.mm_frames.append(mimg)
            mm = np.concatenate([f.transpose(2, 0, 1) for f in self.mm_frames])[None]
        aux = self.aux_vector()
        if mm is None and aux is None:
            return np.ascontiguousarray(obs)
        out = {"obs": np.ascontiguousarray(obs)}
        if aux is not None:
            out["aux"] = aux[None].astype(np.float32)
        if mm is not None:
            out["minimap"] = np.ascontiguousarray(mm)
        return out

    def read_hud(self, screen: np.ndarray) -> dict[str, Any]:
        self.last_hud = self.hud.read(screen) if self.hud is not None else {}
        return self.last_hud

    # -- episode control --------------------------------------------------------------------------
    def _reset_episode_stats(self) -> None:
        self.ep_return, self.ep_len, self.ep_deaths, self.ep_bombs = 0.0, 0, 0, 0

    def _wait(self, seconds: float) -> None:
        end = self.clock() + seconds
        while self.clock() < end:
            self.safety.check()
            self.sleep(min(0.05, max(end - self.clock(), 0)))

    def run_reset_macro(self) -> bool:
        macro = self.cfg.runtime.get("reset_macro") or []
        for step in macro:
            self.safety.check()
            keys = tuple(step["key"]) if isinstance(step["key"], list) else (str(step["key"]),)
            self._tap(keys)
            self._wait(float(step.get("delay", 0.5)))
        return bool(macro)

    def _tap(self, keys: tuple[str, ...]) -> None:
        saved = self.controller.tap_hold_s
        self.controller.tap_hold_s = max(saved, 0.05)
        try:
            self.controller.tap(Tap(keys=keys))
        finally:
            self.controller.tap_hold_s = saved

    def reset(self, seed: int | None = None):
        self.controller.release_all()
        if not self.run_reset_macro():
            if not self.safety.started.is_set():
                self.log_fn(f"[env] press {self.safety.keys['start']} when the stage starts")
            while not self.safety.started.is_set():
                self._wait(0.05)
            self.safety.started.clear()  # consumed: the next episode needs a new signal
        screen = self.grab()
        if self.hud is not None:
            self.hud.reset()
        obs = self.observe(screen, fill=True)
        self.reward_fn.reset(self.read_hud(screen))
        self._reset_episode_stats()
        self.rate = RateKeeper(self.cfg.tick_hz, self.clock, self.sleep)
        return obs, {"hud": dict(self.last_hud)}

    # -- step -----------------------------------------------------------------------------------
    def step(self, action: dict[str, Any]):
        self.safety.check()
        while self.safety.paused.is_set():
            self.controller.release_all()
            self._wait(0.05)
            self.rate = RateKeeper(self.cfg.tick_hz, self.clock, self.sleep)
        a = {k: int(np.asarray(v).reshape(-1)[0]) for k, v in action.items()}
        self.controller.apply(self.space.to_desired(a, self.obs_region, self.minimap_region))
        self.late_s = self.rate.wait()
        screen = self.grab()
        obs = self.observe(screen)
        r, rinfo = self.reward_fn(self.read_hud(screen))
        self.ep_return += r
        self.ep_len += 1
        self.ep_deaths += int(bool(rinfo.get("life_lost")))
        self.ep_bombs += int(bool(rinfo.get("bomb_used")))
        terminated = bool(rinfo.get("game_over") or rinfo.get("stage_clear") or rinfo.get("terminal"))
        truncated = bool(self.max_episode_steps and self.ep_len >= int(self.max_episode_steps)) and not terminated
        info: dict[str, Any] = {k: np.array([v]) for k, v in rinfo.items() if isinstance(v, (bool, int, float))}
        info["late_ms"] = np.array([self.late_s * 1000.0])
        if terminated or truncated:
            info.update(
                episode_return=np.array([self.ep_return]), episode_length=np.array([self.ep_len]),
                episode_deaths=np.array([self.ep_deaths]), episode_bombs=np.array([self.ep_bombs]),
                episode_clear=np.array([bool(rinfo.get("stage_clear"))]),
                episode_score=np.array([float(self.last_hud.get("score") or 0)]),
            )
            obs, _ = self.reset()
        return obs, np.array([r], np.float32), np.array([terminated]), np.array([truncated]), info

    def close(self) -> None:
        self.controller.release_all()
        try:
            self.source.close()
        except Exception:
            pass
