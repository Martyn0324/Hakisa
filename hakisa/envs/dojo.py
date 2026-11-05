"""DanmakuDojo — Hakisa's own (original) vectorised bullet-hell simulator, the "third game".

Why it exists: Jigoku Kisetsukan / Bullet Heaven 2 / LoL cannot run in a cloud container, but every
part of Hakisa (dataset format, models, BC, RL, reward probes, System 2) can be exercised end-to-end
here, quickly, with numbers that can be trusted.

Game rules (all in "field units", the playfield is 192 x 224):
* the player has ``lives`` (default 3) and ``bombs`` (2 per life); hitbox radius 2, graze radius 10
* keyboard scheme: 8 directions, focus = slow (1.4 vs 3.0 units/tick) + narrow, stronger beam
* mouse scheme: the ship flies towards the cursor (max 4.5 units/tick, 2.0 when ``slow``)
* shooting is a beam hitting the nearest enemy above the ship; bombs clear bullets, hurt enemies and
  give 2.5 s of invulnerability; dying also clears the screen and respawns at the bottom
* fairies arrive in waves (aimed fans, rings, rain), a 3-phase boss arrives at ``boss_time``
  (spirals, aimed fans, rings, walls with gaps, rain); defeating it clears the stage
* score: 10/damage, +500 per fairy, +2000 per boss phase, +10000 boss, +20 per grazed bullet

All randomness inside an episode is a hash of (episode seed, tick, slot), so cloned environments
evolve identically under identical actions (common random numbers for System 2's search).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from hakisa.config import GameConfig, load_game_config
from hakisa.envs.dojo_render import FONT, disc_offsets, draw_text, fill_rect, splat
from hakisa.hippocampus.dataset import cursor_heatmap
from hakisa.limbic.rewards import BulletHellReward

FIELD_W, FIELD_H = 192.0, 224.0
MAX_BULLETS = 512
MAX_BODIES = 8  # slot 0 = boss, 1..7 = fairies
HIT_R, GRAZE_R = 2.0, 10.0

AIMED, RING, SPIRAL, RAIN, WALL = range(5)
PATTERN_K = {AIMED: 5, RING: 16, SPIRAL: 3, RAIN: 2, WALL: 14}
PATTERN_SPEED = {AIMED: 2.2, RING: 1.4, SPIRAL: 1.6, RAIN: 2.0, WALL: 1.0}
PATTERN_RADIUS = {AIMED: 3.0, RING: 3.5, SPIRAL: 3.0, RAIN: 2.5, WALL: 4.0}
PATTERN_COLOR = {
    AIMED: (255, 80, 80),
    RING: (80, 200, 255),
    SPIRAL: (255, 120, 255),
    RAIN: (255, 230, 90),
    WALL: (240, 240, 240),
}
BOSS_PHASES = [  # (pattern, period, dphase) x 2 emitters
    [(SPIRAL, 4, 0.27), (AIMED, 30, 0.0)],
    [(RING, 24, 0.20), (WALL, 50, 0.0)],
    [(SPIRAL, 3, 0.33), (RAIN, 6, 0.0)],
]
FAIRY_PATTERNS = [(AIMED, 36, 0.0), (RING, 40, 0.15), (RAIN, 10, 0.0)]

_MOVES = {
    "none": (0.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
    "up_left": (-1.0, -1.0), "up_right": (1.0, -1.0), "down_left": (-1.0, 1.0), "down_right": (1.0, 1.0),
}

_M32 = np.uint64(0xFFFFFFFF)


def hash_uniform(*keys: np.ndarray | int) -> np.ndarray:
    """Deterministic uniform [0, 1) from integer keys (vectorised; splitmix-style mixing)."""
    h = np.uint64(0x9E3779B97F4A7C15)
    shape = np.broadcast(*[np.asarray(k) for k in keys]).shape
    acc = np.full(shape, h, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i, k in enumerate(keys):
            acc ^= (np.asarray(k).astype(np.uint64) + np.uint64(0x632BE59BD9B4E019 + i)) * np.uint64(0xBF58476D1CE4E5B9)
            acc ^= acc >> np.uint64(31)
            acc *= np.uint64(0x94D049BB133111EB)
            acc ^= acc >> np.uint64(29)
    return ((acc & _M32).astype(np.float64)) / 4294967296.0


class DanmakuDojo:
    def __init__(
        self,
        num_envs: int = 16,
        scheme: str = "keyboard",
        obs_hw: tuple[int, int] | None = None,
        stack: int | None = None,
        difficulty: float = 1.0,
        seed: int = 0,
        max_steps: int = 1800,
        boss_time: int = 600,
        lives: int = 3,
        bombs: int = 2,
        cfg: GameConfig | None = None,
    ):
        if scheme not in ("keyboard", "mouse"):
            raise ValueError("scheme must be 'keyboard' or 'mouse'")
        self.scheme = scheme
        self.cfg = cfg or load_game_config("dojo_keyboard" if scheme == "keyboard" else "dojo_mouse")
        self.space = self.cfg.action_space
        self.N = int(num_envs)
        self.H, self.W = obs_hw or (self.cfg.obs.height, self.cfg.obs.width)
        self.stack = int(stack or self.cfg.obs.stack)
        self.difficulty = float(difficulty)
        self.max_steps = int(max_steps)
        self.boss_time = int(boss_time)
        self.start_lives = int(lives)
        self.start_bombs = int(bombs)
        self.base_seed = int(seed)
        self._episode_counter = 0
        if scheme == "keyboard":
            move = self.space.head("move")
            self._move_vec = np.array([_MOVES[c.name] for c in move.choices], dtype=np.float32)
            norm = np.linalg.norm(self._move_vec, axis=1, keepdims=True)
            self._move_vec = self._move_vec / np.maximum(norm, 1.0)
        else:
            self._fine_hw = self.space.head("pointer").fine_hw
        self._reward_fns = [BulletHellReward(self.cfg.reward) for _ in range(self.N)]
        self._offsets = {r: disc_offsets(r) for r in (0.8, 1.2, 1.6, 2.0, 2.5, 4.5)}
        self._alloc()
        self.reset(seed)

    # ------------------------------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------------------------------
    def _alloc(self) -> None:
        n, b, e = self.N, MAX_BULLETS, MAX_BODIES
        f32 = np.float32
        self.px = np.zeros(n, f32)
        self.py = np.zeros(n, f32)
        self.lives = np.zeros(n, np.int64)
        self.bombs = np.zeros(n, np.int64)
        self.invuln = np.zeros(n, np.int64)
        self.bomb_cd = np.zeros(n, np.int64)
        self.score = np.zeros(n, np.int64)
        self.t = np.zeros(n, np.int64)
        self.focus = np.zeros(n, bool)
        self.shooting = np.zeros(n, bool)
        self.cursor = np.zeros((n, 2), f32)  # normalised target of the mouse scheme
        self.ep_seed = np.zeros(n, np.int64)
        self.bg = np.zeros((n, 3), np.int16)
        self.bx = np.zeros((n, b), f32)
        self.by = np.zeros((n, b), f32)
        self.bvx = np.zeros((n, b), f32)
        self.bvy = np.zeros((n, b), f32)
        self.br = np.zeros((n, b), f32)
        self.bpat = np.zeros((n, b), np.int8)
        self.balive = np.zeros((n, b), bool)
        self.bgrazed = np.zeros((n, b), bool)
        self.bptr = np.zeros(n, np.int64)
        self.ealive = np.zeros((n, e), bool)
        self.ex = np.zeros((n, e), f32)
        self.ey = np.zeros((n, e), f32)
        self.ehp = np.zeros((n, e), f32)
        self.emaxhp = np.ones((n, e), f32)
        self.etimer = np.zeros((n, e), np.int64)
        self.estop = np.zeros((n, e), f32)
        self.ehome = np.zeros((n, e), f32)
        self.epat = np.full((n, e, 2), -1, np.int64)
        self.eperiod = np.ones((n, e, 2), np.int64)
        self.ecd = np.zeros((n, e, 2), np.int64)
        self.ephase = np.zeros((n, e, 2), f32)
        self.edphase = np.zeros((n, e, 2), f32)
        self.boss_phase = np.full(n, -1, np.int64)
        self.boss_done = np.zeros(n, bool)
        self.waves: list[list[tuple[int, int]]] = [[] for _ in range(n)]  # (tick, count)
        self.ep_return = np.zeros(n, np.float64)
        self.ep_deaths = np.zeros(n, np.int64)
        self.ep_bombs = np.zeros(n, np.int64)
        self.beam_top = np.zeros(n, f32)
        self.frames = np.zeros((n, self.stack, self.H, self.W, 3), np.uint8)
        self._per_env = [k for k, v in self.__dict__.items() if not k.startswith("_") and isinstance(v, np.ndarray) and v.shape[:1] == (n,)]

    def _reset_env(self, i: int) -> None:
        self._episode_counter += 1
        seed = self.base_seed * 1000003 + self._episode_counter * 7919 + i
        self.ep_seed[i] = seed
        rng = np.random.default_rng(seed)
        self.px[i], self.py[i] = FIELD_W / 2, FIELD_H - 24
        self.lives[i], self.bombs[i] = self.start_lives, self.start_bombs
        self.invuln[i], self.bomb_cd[i], self.score[i], self.t[i] = 30, 0, 0, 0
        self.focus[i] = self.shooting[i] = False
        self.cursor[i] = (self.px[i] / FIELD_W, self.py[i] / FIELD_H)
        self.bg[i] = (int(rng.integers(6, 30)), int(rng.integers(4, 24)), int(rng.integers(20, 50)))
        self.balive[i] = False
        self.bgrazed[i] = False
        self.bptr[i] = 0
        self.ealive[i] = False
        self.epat[i] = -1
        self.boss_phase[i] = -1
        self.boss_done[i] = False
        waves, tick = [], 20
        while tick < self.boss_time - 60:
            waves.append((tick, int(rng.integers(2, 4))))
            tick += int(rng.integers(55, 90) / max(self.difficulty, 0.25))
        self.waves[i] = waves
        self.ep_return[i] = 0.0
        self.ep_deaths[i] = 0
        self.ep_bombs[i] = 0
        self._reward_fns[i].reset(self._hud(i))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.base_seed = int(seed)
            self._episode_counter = 0
        for i in range(self.N):
            self._reset_env(i)
        frame = self._render(np.arange(self.N))
        self.frames[:] = frame[:, None]
        return self._obs(), self._info()

    def _hud(self, i: int) -> dict[str, Any]:
        return {"lives": int(self.lives[i]), "bombs": int(self.bombs[i]), "score": int(self.score[i]),
                "game_over": bool(self.lives[i] <= 0), "stage_clear": bool(self.boss_done[i])}

    # ------------------------------------------------------------------------------------------
    # bullets
    # ------------------------------------------------------------------------------------------
    def _alloc_slots(self, env_ids: np.ndarray, k: int) -> np.ndarray:
        order = np.argsort(env_ids, kind="stable")
        sorted_env = env_ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_env)) + 1]
        lengths = np.diff(np.r_[starts, len(sorted_env)])
        rank_sorted = np.arange(len(sorted_env)) - np.repeat(starts, lengths)
        rank = np.empty_like(rank_sorted)
        rank[order] = rank_sorted
        base = self.bptr[env_ids] + rank * k
        counts = np.bincount(env_ids, minlength=self.N)
        self.bptr = (self.bptr + counts * k) % MAX_BULLETS
        return (base[:, None] + np.arange(k)[None, :]) % MAX_BULLETS

    def _emit(self, pattern: int, env: np.ndarray, x: np.ndarray, y: np.ndarray, phase: np.ndarray, salt: np.ndarray) -> None:
        if len(env) == 0:
            return
        k = PATTERN_K[pattern]
        spd = PATTERN_SPEED[pattern] * math.sqrt(self.difficulty)
        ks = np.arange(k)[None, :]
        alive = np.ones((len(env), k), bool)
        bx = np.repeat(x[:, None], k, 1).astype(np.float32)
        by = np.repeat(y[:, None], k, 1).astype(np.float32)
        if pattern == AIMED:
            base = np.arctan2(self.py[env] - y, self.px[env] - x)
            ang = base[:, None] + np.linspace(-0.35, 0.35, k)[None, :]
            vx, vy = np.cos(ang) * spd, np.sin(ang) * spd
        elif pattern in (RING, SPIRAL):
            ang = phase[:, None] + 2 * np.pi * ks / k
            vx, vy = np.cos(ang) * spd, np.sin(ang) * spd
        elif pattern == RAIN:
            u = hash_uniform(self.ep_seed[env][:, None], self.t[env][:, None], salt[:, None], ks)
            w = hash_uniform(self.ep_seed[env][:, None], self.t[env][:, None], salt[:, None], ks + 97)
            bx = (u * FIELD_W).astype(np.float32)
            by = np.zeros_like(bx)
            ang = np.pi / 2 + (w - 0.5) * 0.6
            vx, vy = np.cos(ang) * spd, np.sin(ang) * spd
        elif pattern == WALL:
            bx = np.repeat(((ks + 0.5) * FIELD_W / k).astype(np.float32), len(env), 0)
            gap = (hash_uniform(self.ep_seed[env], self.t[env], salt) * (k - 3)).astype(np.int64)
            alive = ~((ks >= gap[:, None]) & (ks < gap[:, None] + 3))
            vx, vy = np.zeros((len(env), k)), np.full((len(env), k), spd)
        else:  # pragma: no cover
            raise ValueError(pattern)
        slots = self._alloc_slots(env, k)
        rows = np.repeat(env[:, None], k, 1)
        self.bx[rows, slots] = bx
        self.by[rows, slots] = by
        self.bvx[rows, slots] = vx
        self.bvy[rows, slots] = vy
        self.br[rows, slots] = PATTERN_RADIUS[pattern]
        self.bpat[rows, slots] = pattern
        self.balive[rows, slots] = alive
        self.bgrazed[rows, slots] = False

    def _fire_emitters(self) -> None:
        on_screen = self.ealive & (self.ey > 0) & (self.ey < FIELD_H * 0.75)
        self.ecd -= 1
        fire = (self.epat >= 0) & on_screen[:, :, None] & (self.ecd <= 0)
        for pattern in PATTERN_K:
            n, e, s = np.nonzero(fire & (self.epat == pattern))
            if len(n):
                self._emit(pattern, n, self.ex[n, e], self.ey[n, e], self.ephase[n, e, s], e * 2 + s)
        n, e, s = np.nonzero(fire)
        self.ecd[n, e, s] = np.maximum(1, (self.eperiod[n, e, s] / self.difficulty).astype(np.int64))
        self.ephase[n, e, s] += self.edphase[n, e, s]

    # ------------------------------------------------------------------------------------------
    # enemies
    # ------------------------------------------------------------------------------------------
    def _spawn_waves(self) -> None:
        for i in range(self.N):
            waves = self.waves[i]
            while waves and waves[0][0] <= self.t[i]:
                _, count = waves.pop(0)
                free = [j for j in range(1, MAX_BODIES) if not self.ealive[i, j]]
                for c, j in enumerate(free[:count]):
                    u = hash_uniform(self.ep_seed[i], self.t[i], j, c)
                    v = hash_uniform(self.ep_seed[i], self.t[i], j, c + 31)
                    w = hash_uniform(self.ep_seed[i], self.t[i], j, c + 57)
                    self.ealive[i, j] = True
                    self.ex[i, j] = 20 + float(u) * (FIELD_W - 40)
                    self.ey[i, j] = -8.0
                    self.estop[i, j] = 30 + float(v) * 60
                    self.ehp[i, j] = self.emaxhp[i, j] = 30.0
                    self.etimer[i, j] = 0
                    pat, period, dphase = FAIRY_PATTERNS[int(w * len(FAIRY_PATTERNS)) % len(FAIRY_PATTERNS)]
                    self.epat[i, j] = (pat, -1)
                    self.eperiod[i, j] = (period, 1)
                    self.ecd[i, j] = (15 + int(v * 20), 0)
                    self.ephase[i, j] = (float(u) * 6.28, 0.0)
                    self.edphase[i, j] = (dphase, 0.0)
            if self.t[i] == self.boss_time and not self.ealive[i, 0] and not self.boss_done[i]:
                self.ealive[i, 0] = True
                self.ex[i, 0], self.ey[i, 0] = FIELD_W / 2, -12.0
                self.ehome[i, 0] = FIELD_W / 2
                self.ehp[i, 0] = self.emaxhp[i, 0] = 700.0
                self.etimer[i, 0] = 0
                self._set_boss_phase(np.array([i]), 0)

    def _set_boss_phase(self, envs: np.ndarray, phase: int) -> None:
        for i in envs:
            self.boss_phase[i] = phase
            for s, (pat, period, dphase) in enumerate(BOSS_PHASES[phase]):
                self.epat[i, 0, s] = pat
                self.eperiod[i, 0, s] = period
                self.ecd[i, 0, s] = 20
                self.edphase[i, 0, s] = dphase

    def _move_bodies(self) -> None:
        self.etimer[self.ealive] += 1
        fairy = self.ealive.copy()
        fairy[:, 0] = False
        descending = fairy & (self.ey < self.estop) & (self.etimer < 200)
        self.ey[descending] += 1.2
        leaving = fairy & (self.etimer >= 200)
        self.ey[leaving] -= 1.0
        self.ex[leaving] += np.where(self.ex[leaving] < FIELD_W / 2, -0.8, 0.8)
        gone = fairy & ((self.ey < -20) | (self.ex < -20) | (self.ex > FIELD_W + 20))
        self.ealive[gone] = False
        boss = self.ealive[:, 0]
        if boss.any():
            tb = self.etimer[boss, 0].astype(np.float32)
            self.ey[boss, 0] = np.minimum(48.0, -12.0 + tb * 0.8)
            self.ex[boss, 0] = self.ehome[boss, 0] + 50.0 * np.sin(tb / 60.0)

    def _damage(self, n: np.ndarray, e: np.ndarray, dmg: np.ndarray) -> None:
        if len(n) == 0:
            return
        self.ehp[n, e] -= dmg
        np.add.at(self.score, n, np.round(dmg * 10).astype(np.int64))
        dead = self.ehp[n, e] <= 0
        for i, j in zip(n[dead], e[dead]):
            if not self.ealive[i, j]:
                continue
            self.ealive[i, j] = False
            if j == 0:
                self.score[i] += 10000
                self.boss_done[i] = True
                self.balive[i] = False
            else:
                self.score[i] += 500
        boss_hit = (e == 0) & ~dead
        for i in n[boss_hit]:
            frac = self.ehp[i, 0] / self.emaxhp[i, 0]
            phase = 0 if frac > 2 / 3 else (1 if frac > 1 / 3 else 2)
            if phase > self.boss_phase[i]:
                self.score[i] += 2000
                self.balive[i] = False  # spell break clears the screen
                self._set_boss_phase(np.array([i]), phase)

    # ------------------------------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------------------------------
    def decode_target(self, fx: np.ndarray, fy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fh, fw = self._fine_hw
        return (np.asarray(fx) + 0.5) / fw * FIELD_W, (np.asarray(fy) + 0.5) / fh * FIELD_H

    def encode_target(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fh, fw = self._fine_hw
        fx = np.clip((np.asarray(x) / FIELD_W * fw).astype(np.int64), 0, fw - 1)
        fy = np.clip((np.asarray(y) / FIELD_H * fh).astype(np.int64), 0, fh - 1)
        return fx, fy

    def step(self, actions: dict[str, np.ndarray]):
        n = self.N
        a = {k: np.asarray(v).reshape(n) for k, v in actions.items()}
        if self.scheme == "keyboard":
            self.focus = a["focus"].astype(bool)
            self.shooting = a["shoot"].astype(bool)
            vec = self._move_vec[a["move"]]
            speed = np.where(self.focus, 1.4, 3.0).astype(np.float32)
            self.px += vec[:, 0] * speed
            self.py += vec[:, 1] * speed
        else:
            self.focus = a["slow"].astype(bool)
            self.shooting = a["fire"].astype(bool)
            tx, ty = self.decode_target(a["pointer_x"], a["pointer_y"])
            self.cursor[:, 0], self.cursor[:, 1] = tx / FIELD_W, ty / FIELD_H
            dx, dy = tx - self.px, ty - self.py
            dist = np.sqrt(dx * dx + dy * dy) + 1e-6
            step = np.minimum(dist, np.where(self.focus, 2.0, 4.5))
            self.px += (dx / dist * step).astype(np.float32)
            self.py += (dy / dist * step).astype(np.float32)
        np.clip(self.px, 4, FIELD_W - 4, out=self.px)
        np.clip(self.py, 4, FIELD_H - 4, out=self.py)

        # bombs
        use = a["bomb"].astype(bool) & (self.bombs > 0) & (self.bomb_cd == 0)
        bomb_used = use.copy()
        if use.any():
            idx = np.nonzero(use)[0]
            self.bombs[idx] -= 1
            self.ep_bombs[idx] += 1
            self.balive[idx] = False
            self.invuln[idx] = np.maximum(self.invuln[idx], 50)
            self.bomb_cd[idx] = 50
            hit = self.ealive[idx] & (self.ey[idx] > -5)
            ii, jj = np.nonzero(hit)
            self._damage(idx[ii], jj, np.full(len(ii), 40.0, np.float32))

        # beam
        hw = np.where(self.focus, 3.0, 6.0)
        body_r = np.where(np.arange(MAX_BODIES) == 0, 9.0, 4.0)[None, :]
        cand = self.ealive & (self.ey < self.py[:, None]) & (self.ey > -5) & (np.abs(self.ex - self.px[:, None]) < hw[:, None] + body_r)
        cand &= self.shooting[:, None]
        has = cand.any(1)
        target = np.argmax(np.where(cand, self.ey, -1e9), axis=1)
        self.beam_top = np.where(has, self.ey[np.arange(n), target], 0.0).astype(np.float32)
        hit_env = np.nonzero(has)[0]
        self._damage(hit_env, target[hit_env], np.where(self.focus[hit_env], 1.5, 1.0).astype(np.float32))

        # enemies, emitters, bullets
        self._spawn_waves()
        self._move_bodies()
        self._fire_emitters()
        self.bx += self.bvx
        self.by += self.bvy
        self.balive &= (self.bx > -10) & (self.bx < FIELD_W + 10) & (self.by > -10) & (self.by < FIELD_H + 10)

        # collisions & graze
        dx = self.bx - self.px[:, None]
        dy = self.by - self.py[:, None]
        d2 = dx * dx + dy * dy
        vulnerable = self.invuln == 0
        hit = (self.balive & (d2 < (self.br + HIT_R) ** 2)).any(1) & vulnerable
        graze = self.balive & ~self.bgrazed & (d2 < (self.br + GRAZE_R) ** 2) & vulnerable[:, None]
        self.score += 20 * graze.sum(1)
        self.bgrazed |= graze
        life_lost = hit.copy()
        if hit.any():
            idx = np.nonzero(hit)[0]
            self.lives[idx] -= 1
            self.ep_deaths[idx] += 1
            self.balive[idx] = False
            self.invuln[idx] = 60
            self.bombs[idx] = np.maximum(self.bombs[idx], self.start_bombs)
            self.px[idx], self.py[idx] = FIELD_W / 2, FIELD_H - 24
        self.invuln = np.maximum(self.invuln - 1, 0)
        self.bomb_cd = np.maximum(self.bomb_cd - 1, 0)
        self.t += 1

        # rewards (same BulletHellReward as the real games, fed with the HUD values)
        rewards = np.zeros(n, np.float32)
        for i in range(n):
            r, rinfo = self._reward_fns[i](self._hud(i))
            rewards[i] = r
        self.ep_return += rewards
        game_over = self.lives <= 0
        stage_clear = self.boss_done.copy()
        terminated = game_over | stage_clear
        truncated = (self.t >= self.max_steps) & ~terminated
        info = self._info()
        info.update(life_lost=life_lost, bomb_used=bomb_used, game_over=game_over, stage_clear=stage_clear)
        done = terminated | truncated
        info["episode_return"] = np.where(done, self.ep_return, np.nan)
        info["episode_length"] = np.where(done, self.t, 0)
        info["episode_score"] = np.where(done, self.score, 0)
        info["episode_deaths"] = np.where(done, self.ep_deaths, 0)
        info["episode_bombs"] = np.where(done, self.ep_bombs, 0)
        info["episode_clear"] = np.where(done, stage_clear, False)
        for i in np.nonzero(done)[0]:
            self._reset_env(int(i))

        frame = self._render(np.arange(n))
        self.frames[:, :-1] = self.frames[:, 1:]
        self.frames[:, -1] = frame
        if done.any():
            self.frames[done] = frame[done][:, None]
        return self._obs(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------------------------------
    # observation / info
    # ------------------------------------------------------------------------------------------
    def danger(self) -> np.ndarray:
        dx = self.bx - self.px[:, None]
        dy = self.by - self.py[:, None]
        clear = np.sqrt(dx * dx + dy * dy) - self.br - HIT_R
        return np.where(self.balive, clear, 1e3).min(1).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        boss = self.ealive[:, 0]
        return {
            "lives": self.lives.copy(),
            "bombs": self.bombs.copy(),
            "score": self.score.copy(),
            "t": self.t.copy(),
            "player_xy": np.stack([self.px / FIELD_W, self.py / FIELD_H], 1).astype(np.float32),
            "cursor_xy": self.cursor.copy(),
            "boss_hp": np.where(boss, self.ehp[:, 0] / self.emaxhp[:, 0], 0.0).astype(np.float32),
            "danger": self.danger(),
            "focus": self.focus.copy(),
            "invuln": self.invuln.copy(),
        }

    def _obs(self) -> np.ndarray:
        obs = self.frames.transpose(0, 1, 4, 2, 3).reshape(self.N, self.stack * 3, self.H, self.W)
        if self.scheme == "mouse" and self.cfg.obs.cursor_channel:
            cur = np.stack([cursor_heatmap(self.H, self.W, float(u), float(v)) for u, v in self.cursor])[:, None]
            obs = np.concatenate([obs, cur], axis=1)
        return np.ascontiguousarray(obs)

    def _render(self, envs: np.ndarray, scale: float | None = None, height: int | None = None, width: int | None = None) -> np.ndarray:
        h, w = height or self.H, width or self.W
        sx, sy = w / FIELD_W, h / FIELD_H
        m = len(envs)
        frames = np.empty((m, h, w, 3), np.uint8)
        frames[:] = self.bg[envs][:, None, None, :].astype(np.uint8)
        stripe = (np.arange(h)[None, :] + (self.t[envs][:, None] * 2)) % 24 == 0  # scrolling background
        frames[stripe] = np.minimum(frames[stripe].astype(np.int16) + 18, 255).astype(np.uint8)
        local = np.arange(m)
        # beam
        for k, i in enumerate(envs):
            if self.shooting[i]:
                half = max(1, int(round((3.0 if self.focus[i] else 6.0) * sx * 0.5)))
                cx = int(self.px[i] * sx)
                top = int(max(self.beam_top[i], 0) * sy)
                bot = int(self.py[i] * sy)
                frames[k, top:bot, max(cx - half, 0) : cx + half] = (70, 160, 110)
        # bodies
        for r_units, kind_mask, color in ((9.0, np.arange(MAX_BODIES) == 0, (200, 120, 255)), (4.0, np.arange(MAX_BODIES) > 0, (120, 255, 120))):
            sel = self.ealive[envs] & kind_mask[None, :]
            ii, jj = np.nonzero(sel)
            if len(ii):
                offs = disc_offsets(max(r_units * sx, 0.8))
                ys = (self.ey[envs[ii], jj] * sy).astype(np.int64)
                xs = (self.ex[envs[ii], jj] * sx).astype(np.int64)
                splat(frames, local[ii], ys, xs, np.tile(np.array(color, np.uint8), (len(ii), 1)), offs)
        # bullets
        for pattern, radius in PATTERN_RADIUS.items():
            sel = self.balive[envs] & (self.bpat[envs] == pattern)
            ii, jj = np.nonzero(sel)
            if len(ii):
                offs = disc_offsets(max(radius * sx, 0.8))
                ys = (self.by[envs[ii], jj] * sy).astype(np.int64)
                xs = (self.bx[envs[ii], jj] * sx).astype(np.int64)
                splat(frames, local[ii], ys, xs, np.tile(np.array(PATTERN_COLOR[pattern], np.uint8), (len(ii), 1)), offs)
        # player (blinks while invulnerable) + hitbox when focused
        show = ~((self.invuln[envs] > 0) & ((self.t[envs] // 3) % 2 == 1))
        ii = np.nonzero(show)[0]
        if len(ii):
            ys = (self.py[envs[ii]] * sy).astype(np.int64)
            xs = (self.px[envs[ii]] * sx).astype(np.int64)
            splat(frames, local[ii], ys, xs, np.tile(np.array((120, 200, 255), np.uint8), (len(ii), 1)), disc_offsets(max(4.0 * sx, 1.0)))
            foc = self.focus[envs[ii]]
            if foc.any():
                splat(frames, local[ii][foc], ys[foc], xs[foc], np.tile(np.array((255, 255, 255), np.uint8), (int(foc.sum()), 1)), disc_offsets(max(HIT_R * sx, 0.5)))
        # boss HP bar
        for k, i in enumerate(envs):
            if self.ealive[i, 0]:
                frac = max(self.ehp[i, 0] / self.emaxhp[i, 0], 0.0)
                frames[k, 0 : max(1, int(round(2 * sy))), : int(frac * w)] = (255, 70, 70)
        return frames

    def render_screen(self, i: int = 0) -> np.ndarray:
        """Full 544x448 'window' with the HUD panel (for probes, recording demos, human play)."""
        play = self._render(np.array([i]), height=448, width=384)[0]
        screen = np.zeros((448, 544, 3), np.uint8)
        screen[:, :384] = play
        screen[:, 384:] = (20, 20, 32)
        draw_text(screen, "SCORE", 400, 24, (200, 200, 200))
        draw_text(screen, str(int(self.score[i])), 400, 44, (255, 255, 255))
        draw_text(screen, "LIVES", 400, 80, (200, 200, 200))
        for k in range(min(int(self.lives[i]), 8)):
            fill_rect(screen, 400 + 16 * k, 100, 12, 12, (255, 64, 64))
        draw_text(screen, "BOMBS", 400, 130, (200, 200, 200))
        for k in range(min(int(self.bombs[i]), 8)):
            fill_rect(screen, 400 + 16 * k, 150, 12, 12, (64, 220, 96))
        if self.lives[i] <= 0:
            draw_text(screen, "GAME OVER", 120, 200, (255, 255, 255), scale=3)
        return screen

    # ------------------------------------------------------------------------------------------
    # cloning (System 2 search) and helpers
    # ------------------------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in ("cfg", "space")})

    def restore(self, snap: dict[str, Any]) -> None:
        self.__dict__.update(copy.deepcopy(snap))

    def replicate(self, i: int, copies: int) -> "DanmakuDojo":
        """A new env whose ``copies`` members are all exact clones of env ``i`` (same episode seed)."""
        clone = DanmakuDojo.__new__(DanmakuDojo)
        snap = self.snapshot()
        for k, v in snap.items():
            if k in self._per_env:
                v = np.repeat(v[i : i + 1], copies, axis=0)
            elif k == "waves":
                v = [copy.deepcopy(v[i]) for _ in range(copies)]
            elif k == "_reward_fns":
                v = [copy.deepcopy(v[i]) for _ in range(copies)]
            clone.__dict__[k] = v
        clone.cfg, clone.space, clone.N = self.cfg, self.space, copies
        return clone

    @property
    def action_space(self):
        return self.space

    def random_actions(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        out = {}
        for h in self.space.categorical_heads:
            out[h.name] = rng.integers(0, h.num_classes, self.N)
        if "bomb" in out:
            out["bomb"] = (rng.random(self.N) < 0.01).astype(np.int64)
        for p in self.space.pointer_heads:
            fh, fw = p.fine_hw
            out[f"{p.name}_x"] = rng.integers(0, fw, self.N)
            out[f"{p.name}_y"] = rng.integers(0, fh, self.N)
        return out


__all__ = ["DanmakuDojo", "FIELD_W", "FIELD_H", "FONT", "hash_uniform"]
