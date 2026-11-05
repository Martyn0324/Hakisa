"""DanmakuDojo helpers shared by the CLI and the experiment scripts: demo recording and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from hakisa.envs.dojo import DanmakuDojo
from hakisa.envs.dojo_expert import ScriptedExpert
from hakisa.hippocampus.storage import EpisodeWriter


def record_dojo_demos(
    out_dir: str | Path,
    scheme: str,
    total_steps: int,
    num_envs: int = 32,
    difficulty: float = 2.0,
    seed: int = 0,
    lapse_prob: float = 0.08,
    noise_prob: float = 0.03,
    chunk_size: int = 512,
    log_fn: Callable[[str], None] = print,
) -> list[Path]:
    """Records the scripted demonstrator in exactly the format human recordings use.

    Per tick: the newest frame, the expert's action as ``label/<head>``, HUD values, the cursor
    (mouse scheme) and privileged info used later for System 2's hindsight intents.
    """
    env = DanmakuDojo(num_envs=num_envs, scheme=scheme, difficulty=difficulty, seed=seed)
    expert = ScriptedExpert(env, lapse_prob=lapse_prob, noise_prob=noise_prob, seed=seed)
    env.reset(seed)
    meta = {
        "game": env.cfg.name, "source": "expert", "scheme": scheme, "difficulty": difficulty, "tick_hz": env.cfg.tick_hz,
        "obs_region": [0, 0, env.W, env.H],
    }
    writers = [EpisodeWriter(out_dir, dict(meta, env=i), chunk_size=chunk_size, background=False) for i in range(num_envs)]
    paths: list[Path] = []
    steps = 0
    while steps < total_steps:
        info_before = env._info()
        frames = env.frames[:, -1].copy()
        cursor = env.cursor.copy()
        actions = expert.act()
        _, rewards, term, trunc, info = env.step(actions)
        for i in range(num_envs):
            step = {f"label/{k}": int(v[i]) for k, v in actions.items()}
            step.update(
                {
                    "reward": float(rewards[i]),
                    "life_lost": bool(info["life_lost"][i]),
                    "bomb_used": bool(info["bomb_used"][i]),
                    "hud/lives": int(info_before["lives"][i]),
                    "hud/bombs": int(info_before["bombs"][i]),
                    "hud/score": int(info_before["score"][i]),
                    "cursor_norm": cursor[i].astype(np.float32),
                    "info/player_xy": info_before["player_xy"][i],
                    "info/boss_hp": float(info_before["boss_hp"][i]),
                    "info/danger": float(info_before["danger"][i]),
                    "info/t": int(info_before["t"][i]),
                }
            )
            writers[i].add(frames[i], step)
        steps += num_envs
        done = term | trunc
        for i in np.nonzero(done)[0]:
            paths.append(writers[i].close())
            writers[i] = EpisodeWriter(out_dir, dict(meta, env=int(i)), chunk_size=chunk_size, background=False)
        if steps % (num_envs * 500) == 0:
            log_fn(f"[dojo-demos] {steps}/{total_steps} frames, {len(paths)} finished episodes")
    for w in writers:
        if w.length > 0:
            paths.append(w.close())
    return paths


def dojo_env_factory(scheme: str, num_envs: int, difficulty: float) -> Callable[[int], DanmakuDojo]:
    def make(seed: int) -> DanmakuDojo:
        return DanmakuDojo(num_envs=num_envs, scheme=scheme, difficulty=difficulty, seed=seed)

    return make
