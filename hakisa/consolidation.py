"""Consolidation — reinforcement-learning fine-tuning of the studied (BC) policy.

"She remembers what she studied, and learns from it" (Hakisa 1.0, after memory consolidation).

Algorithm: PPO (clipped surrogate, GAE, entropy bonus) with three safeguards for fine-tuning a
behaviour-cloned policy:

1. **Critic warm-up** — BC gives an actor but no critic; a random critic produces garbage advantages
   that wreck the actor in the first updates (PIRLNav, CVPR 2023).  The first iterations train only
   the value head while acting with the frozen BC policy.
2. **KL anchor** — exact per-head KL(pi || pi_BC) penalty (AlphaStar regularised RL towards its
   supervised policy) so the policy improves without forgetting the human prior.
3. **Demo replay** — a decaying behaviour-cloning loss on the human demonstrations (DAPG, 2018).

BatchNorm layers run on frozen running statistics during RL so acting and learning see the same net.
``AsyncActorLearner`` runs the game at a fixed rate in one thread while PPO updates in another
(real games cannot be paused cheaply nor run in parallel copies).
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hakisa.hippocampus.dataset import BCDataset, worker_seed_init
from hakisa.models.games import save_policy
from hakisa.models.policy import HakisaPolicy
from hakisa.study import batch_to
from hakisa.utils import JsonlLogger


@dataclass
class PPOConfig:
    rollout_steps: int = 128
    num_envs: int = 16
    epochs: int = 3
    minibatch_size: int = 512
    lr: float = 2e-4
    lr_end_frac: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    kl_coef: float = 0.03
    kl_decay: float = 1.0
    demo_coef: float = 0.05
    demo_decay: float = 0.995
    demo_batch: int = 128
    max_grad_norm: float = 0.5
    critic_warmup_iters: int = 10
    target_kl: float | None = 0.05
    reward_scale: bool = True
    update_mode: str = "sync"  # sync (simulator) | async | pause (real games)
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PPOConfig":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda). ``dones[t]`` = the episode ended after the action at step t. Shapes (T, N)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot


class RewardScaler:
    """Divides rewards by the running std of the discounted return (PPO implementation detail #?)."""

    def __init__(self, num_envs: int, gamma: float):
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.rms = RunningMeanStd()

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + rewards
        self.rms.update(self.ret)
        self.ret[dones.astype(bool)] = 0.0
        return rewards / np.sqrt(self.rms.var + 1e-8)


@dataclass
class Rollout:
    obs: torch.Tensor  # (T, N, C, H, W) uint8
    actions: dict[str, torch.Tensor]  # (T, N)
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_value: torch.Tensor
    aux: torch.Tensor | None = None
    minimap: torch.Tensor | None = None
    intent: torch.Tensor | None = None


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


class PPOTrainer:
    def __init__(
        self,
        policy: HakisaPolicy,
        cfg: PPOConfig,
        device: torch.device,
        ref_policy: HakisaPolicy | None = None,
        demo_ds: BCDataset | None = None,
        out_dir: str | Path | None = None,
        log_fn: Callable[[str], None] = print,
    ):
        self.policy = policy.to(device)
        self.policy.set_bn_frozen(True)
        self.ref = (ref_policy or copy.deepcopy(policy)).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=0.0)
        self.value_opt = torch.optim.AdamW(self.policy.value_head.parameters(), lr=cfg.lr * 5, eps=1e-5)
        self.iteration = 0
        self.kl_coef = cfg.kl_coef
        self.demo_coef = cfg.demo_coef if demo_ds is not None and len(demo_ds) else 0.0
        self._demo_iter: Iterator | None = None
        self.demo_ds = demo_ds
        self.scaler: RewardScaler | None = None
        self.out_dir = Path(out_dir) if out_dir else None
        self.logger = JsonlLogger(self.out_dir / "consolidation_log.jsonl") if self.out_dir else None
        self.log_fn = log_fn

    # -- demos ----------------------------------------------------------------------------------
    def _demo_batch(self) -> dict | None:
        if self.demo_ds is None or self.demo_coef <= 0:
            return None
        if self._demo_iter is None:
            self._demo_iter = iter(DataLoader(self.demo_ds, batch_size=self.cfg.demo_batch, shuffle=True, drop_last=True,
                                              worker_init_fn=worker_seed_init))
        try:
            return next(self._demo_iter)
        except StopIteration:
            self._demo_iter = None
            return self._demo_batch()

    # -- collection (vectorised simulator) --------------------------------------------------------
    @torch.no_grad()
    def collect(self, env, obs: np.ndarray, steps: int, episode_cb: Callable[[dict], None] | None = None) -> tuple[Rollout, np.ndarray]:
        self.policy.eval()
        n = obs.shape[0]
        if self.scaler is None or len(self.scaler.ret) != n:
            self.scaler = RewardScaler(n, self.cfg.gamma)
        warm = self.iteration < self.cfg.critic_warmup_iters
        actor = self.ref if warm else self.policy
        buf_obs = torch.empty((steps, *obs.shape), dtype=torch.uint8)
        acts: dict[str, list[torch.Tensor]] = {}
        logp, values, rewards, dones = (torch.empty(steps, n) for _ in range(4))
        for t in range(steps):
            o = torch.from_numpy(obs)
            buf_obs[t] = o
            a, lp, v, _ = actor.act(o.to(self.device))
            if warm:  # value from the critic being trained, actions from the frozen BC policy
                v = self._value(o)
            np_a = {k: x.cpu().numpy() for k, x in a.items()}
            obs, r, term, trunc, info = env.step(np_a)
            done = term | trunc
            for k, x in a.items():
                acts.setdefault(k, []).append(x.cpu())
            logp[t], values[t] = lp.cpu(), v.cpu()
            rewards[t] = torch.from_numpy(self.scaler(r, done) if self.cfg.reward_scale else r).float()
            dones[t] = torch.from_numpy(done.astype(np.float32))
            if episode_cb is not None and done.any():
                for i in np.nonzero(done)[0]:
                    episode_cb({k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in info.items() if k.startswith("episode_")})
        last_value = self._value(torch.from_numpy(obs))
        roll = Rollout(buf_obs, {k: torch.stack(v) for k, v in acts.items()}, logp, values, rewards, dones, last_value)
        return roll, obs

    @torch.no_grad()
    def _value(self, obs: torch.Tensor, aux=None, minimap=None, intent=None) -> torch.Tensor:
        feats = self.policy.features(obs.to(self.device), aux, minimap, intent)
        return self.policy.value_head(feats.trunk)[:, 0].cpu()

    # -- update -----------------------------------------------------------------------------------
    def update(self, roll: Rollout) -> dict[str, float]:
        cfg = self.cfg
        warm = self.iteration < cfg.critic_warmup_iters
        adv, ret = compute_gae(roll.rewards, roll.values, roll.dones, roll.last_value, cfg.gamma, cfg.gae_lambda)
        b_obs = _flat(roll.obs)
        b_act = {k: _flat(v) for k, v in roll.actions.items()}
        b_logp, b_adv, b_ret, b_val = _flat(roll.logp), _flat(adv), _flat(ret), _flat(roll.values)
        b_aux = _flat(roll.aux) if roll.aux is not None else None
        b_mm = _flat(roll.minimap) if roll.minimap is not None else None
        b_int = _flat(roll.intent) if roll.intent is not None else None
        total = b_obs.shape[0]
        mb = min(cfg.minibatch_size, total)
        frac = min(self.iteration / max(getattr(self, "total_iterations", 1), 1), 1.0)
        lr = cfg.lr * (1.0 - (1.0 - cfg.lr_end_frac) * frac)
        for g in self.opt.param_groups:
            g["lr"] = lr
        stats: dict[str, list[float]] = {}
        self.policy.train()
        dev = self.device
        stop = False
        for _ in range(cfg.epochs):
            perm = torch.randperm(total)
            for start in range(0, total, mb):
                idx = perm[start : start + mb]
                obs = b_obs[idx].to(dev)
                actions = {k: v[idx].to(dev) for k, v in b_act.items()}
                aux = b_aux[idx].to(dev) if b_aux is not None else None
                mm = b_mm[idx].to(dev) if b_mm is not None else None
                it = b_int[idx].to(dev) if b_int is not None else None
                ret_mb = b_ret[idx].to(dev)
                if warm:
                    with torch.no_grad():
                        feats = self.policy.features(obs, aux, mm, it)
                    v = self.policy.value_head(feats.trunk)[:, 0]
                    v_loss = 0.5 * F.mse_loss(v, ret_mb)
                    self.value_opt.zero_grad(set_to_none=True)
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.value_head.parameters(), cfg.max_grad_norm)
                    self.value_opt.step()
                    stats.setdefault("v_loss", []).append(float(v_loss))
                    continue
                logp, ent, v, dist = self.policy.evaluate_actions(obs, actions, aux, mm, it)
                adv_mb = b_adv[idx].to(dev)
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                log_ratio = logp - b_logp[idx].to(dev)
                ratio = log_ratio.exp()
                pg = torch.max(-adv_mb * ratio, -adv_mb * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * F.mse_loss(v, ret_mb)
                with torch.no_grad():
                    ref_out = self.ref(obs, aux, mm, it, condition=actions)
                ref_dist = self.ref.distribution(ref_out)
                kl_ref = dist.kl(ref_dist, actions).mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent.mean() + self.kl_coef * kl_ref
                demo = self._demo_batch()
                if demo is not None:
                    demo = batch_to(demo, dev)
                    out = self.policy(demo["obs"], demo.get("aux"), demo.get("minimap"), demo.get("intent"), condition=demo["labels"])
                    d_losses = self.policy.distribution(out).bc_losses(demo["labels"])
                    d_loss = sum(d_losses.values())
                    loss = loss + self.demo_coef * d_loss
                    stats.setdefault("demo_loss", []).append(float(d_loss))
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                for k, x in (("pg_loss", pg), ("v_loss", v_loss), ("entropy", ent.mean()), ("kl_ref", kl_ref),
                             ("approx_kl", approx_kl), ("clipfrac", ((ratio - 1).abs() > cfg.clip).float().mean())):
                    stats.setdefault(k, []).append(float(x))
                if cfg.target_kl is not None and float(approx_kl) > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.iteration += 1
        if not warm:
            self.kl_coef *= cfg.kl_decay
            self.demo_coef *= cfg.demo_decay
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        var_y = float(b_ret.var())
        out["explained_var"] = float(1 - (b_ret - b_val).var() / var_y) if var_y > 0 else 0.0
        out.update(iteration=self.iteration, warmup=float(warm), lr=lr, kl_coef=self.kl_coef, demo_coef=self.demo_coef)
        return out

    # -- full loop (vectorised simulator) ---------------------------------------------------------
    def train(
        self,
        env,
        total_steps: int,
        eval_fn: Callable[[HakisaPolicy], dict[str, float]] | None = None,
        eval_every: int = 20,
        save_every: int = 20,
    ) -> list[dict[str, Any]]:
        obs, _ = env.reset()
        per_iter = self.cfg.rollout_steps * obs.shape[0]
        self.total_iterations = max(1, total_steps // per_iter)
        history: list[dict[str, Any]] = []
        episodes: list[dict] = []
        t0 = time.time()
        for it in range(self.total_iterations):
            roll, obs = self.collect(env, obs, self.cfg.rollout_steps, episodes.append)
            stats = self.update(roll)
            stats["env_steps"] = (it + 1) * per_iter
            stats["elapsed_s"] = time.time() - t0
            if episodes:
                recent = episodes[-64:]
                for key in ("episode_return", "episode_length", "episode_deaths", "episode_score", "episode_clear"):
                    vals = [float(e[key]) for e in recent if key in e]
                    if vals:
                        stats[f"train/{key}"] = float(np.mean(vals))
            if eval_fn is not None and (it + 1) % eval_every == 0:
                stats.update({f"eval/{k}": v for k, v in eval_fn(self.policy).items()})
            history.append(stats)
            if self.logger:
                self.logger.log(stats)
            if (it + 1) % 5 == 0 or it == 0:
                self.log_fn(
                    f"[consolidation] it {it + 1}/{self.total_iterations} steps {stats['env_steps']} "
                    f"ret {stats.get('train/episode_return', float('nan')):.2f} kl_ref {stats.get('kl_ref', 0):.3f} "
                    f"v_loss {stats.get('v_loss', 0):.3f} ev {stats['explained_var']:.2f}"
                )
            if self.out_dir and ((it + 1) % save_every == 0 or it + 1 == self.total_iterations):
                save_policy(self.policy, self.out_dir / "last.pt", iteration=it + 1, stats=stats, config=asdict(self.cfg))
        return history


class AsyncActorLearner:
    """Real-time training: an actor thread plays at a fixed rate while the learner runs PPO updates.

    The actor always uses the newest weights it has been given and records *behaviour* log-probs, so
    PPO's importance ratio stays correct even though the policy lags by up to one update.
    """

    def __init__(self, trainer: PPOTrainer, env, pause_fn: Callable[[], None] | None = None,
                 resume_fn: Callable[[], None] | None = None):
        self.trainer = trainer
        self.env = env
        self.actor = copy.deepcopy(trainer.policy).eval()
        self.lock = threading.Lock()
        self.rollouts: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.pause_fn, self.resume_fn = pause_fn, resume_fn
        self.episodes: list[dict] = []

    def _sync_actor(self) -> None:
        with self.lock:
            self.actor.load_state_dict(self.trainer.policy.state_dict())

    def _actor_loop(self, steps: int) -> None:
        dev = self.trainer.device
        obs, _ = self.env.reset()
        scaler = RewardScaler(obs.shape[0], self.trainer.cfg.gamma)
        while not self.stop_event.is_set():
            buf: dict[str, list] = {"obs": [], "logp": [], "values": [], "rewards": [], "dones": []}
            acts: dict[str, list] = {}
            for _ in range(steps):
                if self.stop_event.is_set():
                    return
                o = torch.from_numpy(obs)
                with self.lock, torch.no_grad():
                    warm = self.trainer.iteration < self.trainer.cfg.critic_warmup_iters
                    pol = self.trainer.ref if warm else self.actor
                    a, lp, v, _ = pol.act(o.to(dev))
                obs, r, term, trunc, info = self.env.step({k: x.cpu().numpy() for k, x in a.items()})
                done = term | trunc
                buf["obs"].append(o)
                buf["logp"].append(lp.cpu())
                buf["values"].append(v.cpu())
                buf["rewards"].append(torch.from_numpy(scaler(r, done)).float())
                buf["dones"].append(torch.from_numpy(done.astype(np.float32)))
                for k, x in a.items():
                    acts.setdefault(k, []).append(x.cpu())
                if done.any():
                    self.episodes.append({k: v for k, v in info.items() if k.startswith("episode_")})
            with self.lock, torch.no_grad():
                last_v = self.trainer._value(torch.from_numpy(obs))
            roll = Rollout(
                torch.stack(buf["obs"]), {k: torch.stack(v) for k, v in acts.items()}, torch.stack(buf["logp"]),
                torch.stack(buf["values"]), torch.stack(buf["rewards"]), torch.stack(buf["dones"]), last_v,
            )
            self.rollouts.put(roll)

    def run(self, iterations: int, on_update: Callable[[dict], None] | None = None) -> list[dict]:
        trainer = self.trainer
        trainer.total_iterations = iterations
        mode = trainer.cfg.update_mode
        history = []
        actor = threading.Thread(target=self._actor_loop, args=(trainer.cfg.rollout_steps,), daemon=True)
        actor.start()
        try:
            for _ in range(iterations):
                roll = self.rollouts.get()
                if mode == "pause" and self.pause_fn:
                    self.pause_fn()
                with self.lock:
                    stats = trainer.update(roll)
                self._sync_actor()
                if mode == "pause" and self.resume_fn:
                    self.resume_fn()
                history.append(stats)
                if on_update:
                    on_update(stats)
        finally:
            self.stop_event.set()
            actor.join(timeout=10)
        return history


@torch.no_grad()
def evaluate_policy(
    policy: HakisaPolicy,
    make_env: Callable[[int], Any],
    episodes: int = 32,
    num_envs: int = 16,
    deterministic: bool = False,
    device: torch.device | str = "cpu",
    max_steps: int = 100000,
    actor: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
    seed: int = 12345,
) -> dict[str, float]:
    """Plays ``episodes`` complete episodes (fixed seeds) and averages the episode statistics."""
    env = make_env(seed)
    obs, _ = env.reset(seed)
    if policy is not None:
        policy.eval()
    done_eps: list[dict] = []
    for _ in range(max_steps):
        if actor is not None:
            a = actor(obs)
        else:
            act, *_ = policy.act(torch.from_numpy(obs).to(device), deterministic=deterministic)
            a = {k: v.cpu().numpy() for k, v in act.items()}
        obs, r, term, trunc, info = env.step(a)
        for i in np.nonzero(term | trunc)[0]:
            done_eps.append({k: float(v[i]) for k, v in info.items() if k.startswith("episode_")})
        if len(done_eps) >= episodes:
            break
    done_eps = done_eps[:episodes]
    out: dict[str, float] = {"episodes": float(len(done_eps))}
    for key in ("episode_return", "episode_length", "episode_deaths", "episode_score", "episode_clear", "episode_bombs"):
        vals = np.array([e[key] for e in done_eps if key in e], dtype=np.float64)
        if len(vals):
            out[key] = float(vals.mean())
            out[key + "_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return out
