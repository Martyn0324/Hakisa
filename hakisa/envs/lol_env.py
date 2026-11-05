"""League of Legends environment (real time).

Differences from the bullet-hell environments:
* the reward and the 32-d game-state vector come from Riot's local Live Client Data API, not the HUD;
* observations are a dict: main view, minimap stream and game-state vector;
* an episode is a whole game (it ends on the ``GameEnd`` event); ``reset`` waits until a game is
  running (the API answers only while you are in a game);
* duo-partner instructions (F1-F6 by default) are kept for System 2.

RISK NOTICE: automated input in League of Legends violates Riot's Terms of Service and Vanguard may
detect it; use Practice Tool / custom games against bots at your own risk. No evasion is implemented.
"""

from __future__ import annotations

import numpy as np

from hakisa.envs.realtime import RealtimeGameEnv
from hakisa.limbic.lol_client import LiveClientPoller, LoLStats
from hakisa.limbic.lol_reward import LoLReward


class LoLEnv(RealtimeGameEnv):
    def __init__(self, cfg, source, controller, capture_region, safety=None, poller=None, wait_timeout_s: float | None = None, **kw):
        if poller is None:
            poller = LiveClientPoller(cfg.reward.get("live_client_url", "https://127.0.0.1:2999/liveclientdata/allgamedata"), interval=0.25)
        if isinstance(poller, LiveClientPoller) and not poller.is_alive():
            poller.start()
        self.poller = poller
        self.last_stats: LoLStats | None = None
        self.instruction: str | None = None
        self.wait_timeout_s = wait_timeout_s
        super().__init__(cfg, source, controller, capture_region, hud=None, reward_fn=LoLReward(cfg.reward), safety=safety, **kw)

    # -- duo partner -> System 2 --------------------------------------------------------------------
    def attach_instructions(self, tracker) -> None:
        for key, text in (self.cfg.runtime.get("instruction_keys") or {}).items():
            tracker.on_hotkey(str(key), lambda text=text: self.set_instruction(text))

    def set_instruction(self, text: str | None) -> None:
        self.instruction = text
        self.log_fn(f"[lol] partner says: {text}")

    # -- live client --------------------------------------------------------------------------------
    def _stats(self) -> LoLStats | None:
        _, stats, _ = self.poller.latest()
        return stats

    def aux_vector(self) -> np.ndarray:
        stats = self._stats()
        return stats.vector() if stats is not None else np.zeros(self.cfg.obs.aux_dim, np.float32)

    def read_hud(self, screen: np.ndarray):  # type: ignore[override]
        stats = self._stats()
        if stats is not None:
            self.last_stats = stats
        self.last_hud = {"stats": stats.values if stats else None, "text": stats.text() if stats else None,
                         "instruction": self.instruction}
        return stats

    def reset(self, seed: int | None = None):
        self.controller.release_all()
        waited = 0.0
        while self._stats() is None or self._stats().result is not None:  # wait for a running game
            if waited == 0.0:
                self.log_fn("[lol] waiting for a game (Live Client Data API not answering yet)...")
            self._wait(0.5)
            waited += 0.5
            if self.wait_timeout_s is not None and waited >= self.wait_timeout_s:
                raise TimeoutError("no League of Legends game detected")
        screen = self.grab()
        obs = self.observe(screen, fill=True)
        self.reward_fn.reset(self.read_hud(screen))
        self._reset_episode_stats()
        from hakisa.utils import RateKeeper

        self.rate = RateKeeper(self.cfg.tick_hz, self.clock, self.sleep)
        return obs, {"hud": dict(self.last_hud)}

    def close(self) -> None:
        super().close()
        if isinstance(self.poller, LiveClientPoller):
            self.poller.stop()


def lol_state_text(stats: LoLStats | None, instruction: str | None = None) -> str:
    text = stats.text() if stats is not None else "no live data"
    if instruction:
        text += f". Partner says: {instruction}"
    return text


__all__ = ["LoLEnv", "lol_state_text"]

