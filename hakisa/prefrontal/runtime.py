"""Asynchronous System 2 at play time: plans ~once a second while System 1 acts every tick.

The Agent calls :meth:`AsyncPlanner.observe` after each decision (cheap: it only stores the latest
inputs) and :meth:`latest_intent` before each decision.  The planner thread turns the newest
observation into visual tokens, asks the LLM for a PLAN and publishes it.  A plan older than
``max_age_s`` is dropped and System 1 falls back to the null intent, so a slow or crashed LLM can
never freeze Hakisa's reflexes.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import torch

from hakisa.prefrontal.intents import IntentSchema, bullet_hell_state_text
from hakisa.prefrontal.reasoner import Prefrontal, make_prompt, s1_visual_tokens


class AsyncPlanner:
    def __init__(self, prefrontal: Prefrontal, hz: float = 1.0, max_age_s: float = 3.0, think: bool = True,
                 state_text_fn: Callable[[dict], str] = bullet_hell_state_text, log_fn: Callable[[str], None] = print):
        self.pf = prefrontal.eval()
        self.schema: IntentSchema = prefrontal.schema
        self.period = 1.0 / hz
        self.max_age_s = max_age_s
        self.think = think
        self.state_text_fn = state_text_fn
        self.log_fn = log_fn
        self._lock = threading.Lock()
        self._latest_obs: tuple | None = None
        self._plan: tuple[float, torch.Tensor, str] | None = None
        self._previous: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.instruction: str | None = None

    def observe(self, policy, obs, aux, minimap, hud: dict[str, Any]) -> None:
        with self._lock:
            self._latest_obs = (policy, obs.detach(), minimap.detach() if minimap is not None else None, dict(hud or {}))

    def latest_intent(self, device) -> torch.Tensor | None:
        with self._lock:
            plan = self._plan
        if plan is None or time.perf_counter() - plan[0] > self.max_age_s:
            return None
        return plan[1].to(device)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                item = self._latest_obs
            if item is not None:
                try:
                    policy, obs, mm, hud = item
                    vis = s1_visual_tokens(policy, obs[:1], mm[:1] if mm is not None else None)
                    with torch.no_grad():  # System 1's amygdala tells System 2 how dangerous it is
                        hud.setdefault("danger_p", float(torch.sigmoid(policy(obs[:1], minimap=mm[:1] if mm is not None else None).danger)[0]))
                    instruction = hud.get("instruction") or self.instruction
                    text = hud.get("text") or self.state_text_fn(hud)
                    prompt = make_prompt(self.pf.title, text, instruction, self._previous)
                    out, plan = self.pf.plan(vis.to(self.pf.device), [prompt], max_new_tokens=160 if self.think else 32)[0]
                    if plan is not None:
                        ids = torch.from_numpy(self.schema.encode(plan))[None]
                        with self._lock:
                            self._plan = (time.perf_counter(), ids, out)
                        self._previous = self.schema.format_plan(plan)
                        self.log_fn(f"[system2] {self._previous}  ({time.perf_counter() - t0:.2f}s)")
                except Exception as exc:  # never take System 1 down with us
                    self.log_fn(f"[system2] error: {exc}")
            self._stop.wait(max(self.period - (time.perf_counter() - t0), 0.01))

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def load_planner(directory: str, cfg, device: torch.device, hz: float = 1.0) -> AsyncPlanner:  # pragma: no cover
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pf = Prefrontal.load(directory, device=device, dtype=dtype)

    def lol_state(hud: dict) -> str:
        return hud.get("text") or "no live data"

    return AsyncPlanner(pf, hz=hz, state_text_fn=lol_state if cfg.name == "lol" else bullet_hell_state_text)
