"""Scripted demonstrator for DanmakuDojo — the stand-in for the owner's own gameplay.

It *cheats* (reads bullet positions/velocities from the simulator) and searches a short horizon:
for every candidate input it predicts the ship and bullet positions for ``horizon`` ticks and keeps
the candidate with the largest clearance, preferring to stay under the current target (to shoot it)
and near the bottom of the screen.  Human-like imperfections: ``lapse_prob`` (keeps the previous
input, like a late reaction) and ``noise_prob`` (random input).  It bombs only when every candidate
predicts a hit.
"""

from __future__ import annotations

import numpy as np

from hakisa.envs.dojo import FIELD_H, FIELD_W, HIT_R, DanmakuDojo

_KEEP = 96  # nearest bullets considered per env


class ScriptedExpert:
    def __init__(
        self,
        env: DanmakuDojo,
        horizon: int = 10,
        lapse_prob: float = 0.03,
        noise_prob: float = 0.01,
        bomb_threshold: float = 0.25,
        seed: int = 0,
    ):
        self.env = env
        self.horizon = int(horizon)
        self.lapse_prob = lapse_prob
        self.noise_prob = noise_prob
        self.bomb_threshold = bomb_threshold
        self.rng = np.random.default_rng(seed)
        if env.scheme == "keyboard":
            move = env._move_vec  # (9, 2) unit-ish directions in move-choice order
            self.cand_move = np.repeat(np.arange(len(move)), 2)
            self.cand_focus = np.tile(np.array([0, 1]), len(move))
            speed = np.where(self.cand_focus == 1, 1.4, 3.0)[:, None]
            self.cand_vel = (move[self.cand_move] * speed).astype(np.float32)  # (C, 2)
        else:
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            offs = [(0.0, 0.0)] + [(np.cos(a) * d, np.sin(a) * d) for d in (10.0, 26.0) for a in angles]
            offs = np.array(offs, dtype=np.float32)
            self.cand_off = np.repeat(offs, 2, axis=0)  # (C, 2)
            self.cand_focus = np.tile(np.array([0, 1]), len(offs))
        self.n_cand = len(self.cand_focus)
        self.prev = np.zeros(env.N, np.int64)

    def _future_positions(self) -> tuple[np.ndarray, np.ndarray]:
        env = self.env
        ks = np.arange(1, self.horizon + 1, dtype=np.float32)  # (T,)
        p = np.stack([env.px, env.py], 1)[:, None, None, :]  # (N,1,1,2)
        if env.scheme == "keyboard":
            pos = p + ks[None, None, :, None] * self.cand_vel[None, :, None, :]
            target = np.broadcast_to(p[:, :, 0, :] + self.cand_vel[None] * 6, (env.N, self.n_cand, 2)).copy()
        else:
            target = p[:, :, 0, :] + self.cand_off[None]  # (N, C, 2)
            target[..., 0] = np.clip(target[..., 0], 4, FIELD_W - 4)
            target[..., 1] = np.clip(target[..., 1], 4, FIELD_H - 4)
            delta = target - p[:, :, 0, :]
            dist = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-6
            speed = np.where(self.cand_focus == 1, 2.0, 4.5)[None, :, None]
            travel = np.minimum(ks[None, None, :, None] * speed[..., None], dist[:, :, None, :])
            pos = p + (delta / dist)[:, :, None, :] * travel
        pos[..., 0] = np.clip(pos[..., 0], 4, FIELD_W - 4)
        pos[..., 1] = np.clip(pos[..., 1], 4, FIELD_H - 4)
        return pos.astype(np.float32), target.astype(np.float32)

    def _clearance(self, pos: np.ndarray) -> np.ndarray:
        """min over time of (distance to the nearest bullet - radii), with mild time discounting."""
        env = self.env
        d0 = np.where(env.balive, np.hypot(env.bx - env.px[:, None], env.by - env.py[:, None]), np.inf)
        keep = min(_KEEP, d0.shape[1])
        idx = np.argpartition(d0, keep - 1, axis=1)[:, :keep]
        rows = np.arange(env.N)[:, None]
        alive = np.isfinite(d0[rows, idx])
        bx, by = env.bx[rows, idx], env.by[rows, idx]
        bvx, bvy, br = env.bvx[rows, idx], env.bvy[rows, idx], env.br[rows, idx]
        ks = np.arange(1, self.horizon + 1, dtype=np.float32)
        fbx = bx[:, None, :] + ks[None, :, None] * bvx[:, None, :]  # (N, T, K)
        fby = by[:, None, :] + ks[None, :, None] * bvy[:, None, :]
        dx = pos[:, :, :, 0:1] - fbx[:, None]  # (N, C, T, K)
        dy = pos[:, :, :, 1:2] - fby[:, None]
        clear = np.sqrt(dx * dx + dy * dy) - br[:, None, None, :] - HIT_R
        clear = np.where(alive[:, None, None, :], clear, 1e3).min(-1)  # (N, C, T)
        return (clear + 0.15 * ks[None, None, :]).min(-1)  # earlier collisions matter more

    def _aim_x(self) -> np.ndarray:
        env = self.env
        boss = env.ealive[:, 0]
        on = env.ealive & (env.ey > 0)
        lowest = np.argmax(np.where(on, env.ey, -1e9), axis=1)
        x = np.where(on.any(1), env.ex[np.arange(env.N), lowest], FIELD_W / 2)
        return np.where(boss, env.ex[:, 0], x)

    def act(self) -> dict[str, np.ndarray]:
        env = self.env
        pos, target = self._future_positions()
        safe = self._clearance(pos)  # (N, C)
        end = pos[:, :, -1, :]
        align = -np.abs(end[..., 0] - self._aim_x()[:, None]) / FIELD_W
        home = -np.abs(end[..., 1] - 0.82 * FIELD_H) / FIELD_H
        score = np.minimum(safe, 8.0) + 3.0 * align + 2.0 * home
        score += 0.4 * (np.arange(self.n_cand)[None, :] == self.prev[:, None])
        score -= 0.3 * self.cand_focus[None, :] * (safe > 6.0)  # focus only when it helps
        choice = score.argmax(1)
        lapse = self.rng.random(env.N) < self.lapse_prob
        choice = np.where(lapse, self.prev, choice)
        noise = self.rng.random(env.N) < self.noise_prob
        choice = np.where(noise, self.rng.integers(0, self.n_cand, env.N), choice)
        self.prev = choice
        best_safe = safe.max(1)
        bomb = (best_safe < self.bomb_threshold) & (env.bombs > 0) & (env.invuln == 0) & (env.bomb_cd == 0)
        focus = self.cand_focus[choice]
        if env.scheme == "keyboard":
            return {
                "move": self.cand_move[choice],
                "focus": focus,
                "shoot": np.ones(env.N, np.int64),
                "bomb": bomb.astype(np.int64),
            }
        tgt = target[np.arange(env.N), choice]
        fx, fy = env.encode_target(tgt[:, 0], tgt[:, 1])
        return {
            "pointer_x": fx,
            "pointer_y": fy,
            "fire": np.ones(env.N, np.int64),
            "bomb": bomb.astype(np.int64),
            "slow": focus,
        }
