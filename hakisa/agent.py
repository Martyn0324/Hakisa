"""Play mode: Hakisa plays a real game; you can take over at any time (and teach her while doing so).

Loop per tick: observation -> System 1 policy (-> intent from System 2 if a planner runs) -> inputs.
Human takeover (F8) implements the HG-DAgger idea (Kelly et al., ICRA 2019): when Hakisa is about
to fail, the human plays, and exactly those corrective frames are recorded for the next Study run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from hakisa.consolidation import split_obs
from hakisa.envs.realtime import HakisaStop, RealtimeGameEnv
from hakisa.models.policy import HakisaPolicy
from hakisa.utils import RateKeeper


class Agent:
    def __init__(
        self,
        policy: HakisaPolicy,
        env: RealtimeGameEnv,
        device: torch.device,
        deterministic: bool = False,
        temperature: float = 1.0,
        planner=None,
        recorder_factory: Callable[[], object] | None = None,
        log_every: int = 200,
        log_fn: Callable[[str], None] = print,
    ):
        self.policy = policy.to(device).eval()
        self.env = env
        self.device = device
        self.deterministic = deterministic
        self.temperature = temperature
        self.planner = planner
        self.recorder_factory = recorder_factory
        self.log_every = log_every
        self.log_fn = log_fn
        self.latencies: list[float] = []
        self.saved: list[Path] = []

    def _tensor(self, x):
        return torch.from_numpy(x).to(self.device) if x is not None else None

    def _takeover(self) -> None:
        env = self.env
        env.controller.release_all()
        rec = self.recorder_factory() if self.recorder_factory else None
        rate = RateKeeper(env.cfg.tick_hz)
        try:
            while env.safety.takeover.is_set():
                env.safety.check()
                screen = env.grab()
                if rec is not None:
                    rec.tick(screen)
                env.observe(screen)  # keep the frame stack fresh for when Hakisa resumes
                rate.wait()
        finally:
            if rec is not None and getattr(rec, "ticks", 0):
                self.saved.append(rec.close())
                self.log_fn(f"[agent] saved {self.saved[-1]} — add it to your training data")

    def run(self, max_steps: int | None = None) -> dict[str, float]:
        env = self.env
        obs, _ = env.reset()
        steps, total_r = 0, 0.0
        try:
            while max_steps is None or steps < max_steps:
                env.safety.check()
                if env.safety.takeover.is_set():
                    self._takeover()
                    obs = env.observe(env.grab())
                    continue
                t0 = time.perf_counter()
                o, aux, mm = (self._tensor(x) for x in split_obs(obs))
                intent = self.planner.latest_intent(self.device) if self.planner is not None else None
                with torch.no_grad():
                    actions, _, value, out = self.policy.act(o, aux, mm, intent, deterministic=self.deterministic,
                                                             temperature=self.temperature)
                self.latencies.append(time.perf_counter() - t0)
                if self.planner is not None:
                    self.planner.observe(self.policy, o, aux, mm, env.last_hud)
                obs, r, term, trunc, info = env.step({k: v.cpu().numpy() for k, v in actions.items()})
                total_r += float(r[0])
                steps += 1
                if steps % self.log_every == 0:
                    lat = np.array(self.latencies[-self.log_every :]) * 1000
                    self.log_fn(
                        f"[agent] step {steps} return {total_r:.2f} decide {np.median(lat):.1f} ms (p90 {np.percentile(lat, 90):.1f}) "
                        f"late {float(info['late_ms'][0]):.1f} ms hud {env.last_hud}"
                    )
        except HakisaStop:
            self.log_fn("[agent] stopped by the kill switch")
        finally:
            env.controller.release_all()
        lat = np.array(self.latencies or [0.0]) * 1000
        return {"steps": steps, "return": total_r, "decide_ms_median": float(np.median(lat))}
