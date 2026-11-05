"""Training System 2: supervised plans from hindsight, then GRPO against game outcomes.

1. ``build_s2_samples``  — visual tokens from System 1's encoder + state text + hindsight PLAN (+ rationale)
2. ``train_s2_sl``       — next-token cross-entropy on the completion (LoRA for Qwen3, full for the tiny model)
3. ``grpo_train``        — Group Relative Policy Optimization (DeepSeekMath, 2024): for each game state,
   sample a *group* of plans, score each plan by letting System 1 play it from an exact clone of the state
   (DanmakuDojo is clonable with common random numbers), normalise the returns inside the group and push
   up the log-likelihood of the better plans, with a KL penalty to the supervised reference.
   (Real games cannot be cloned; there the intent-conditioned critic V(s, intent) of System 1 can score
   the group instead — see docs.)
4. ``play_with_planner`` — evaluation of System 1 + System 2 playing together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from hakisa.config import GameConfig
from hakisa.hippocampus.dataset import BCDataset
from hakisa.prefrontal.intents import (
    IntentSchema,
    bullet_hell_hud,
    bullet_hell_rationale,
    bullet_hell_state_text,
    hindsight_for_episode,
    lol_rationale,
    zone_of,
)
from hakisa.limbic.lol_client import stats_from_vector
from hakisa.prefrontal.reasoner import Prefrontal, make_completion, make_prompt, s1_visual_tokens


@dataclass
class S2Sample:
    vis: torch.Tensor  # (n_vis, C)
    prompt: tuple[str, str]
    completion: str
    intent: np.ndarray  # (S,)


@torch.no_grad()
def build_s2_samples(episodes, cfg: GameConfig, s1_policy, schema: IntentSchema, stride: int = 10, think: bool = True,
                     device: torch.device | str = "cpu", batch: int = 128) -> list[S2Sample]:
    s1_policy.eval()
    ds = BCDataset(episodes, cfg)
    intents = [hindsight_for_episode(ep, cfg, schema) for ep in episodes]
    chosen = [i for i, (e, t) in enumerate(ds.index) if t % stride == 0]
    samples: list[S2Sample] = []
    for s in range(0, len(chosen), batch):
        idx = chosen[s : s + batch]
        items = [ds.item_numpy(i) for i in idx]
        obs = torch.from_numpy(np.stack([it["obs"] for it in items])).to(device)
        mm = torch.from_numpy(np.stack([it["minimap"] for it in items])).to(device) if "minimap" in items[0] else None
        vis = s1_visual_tokens(s1_policy, obs, mm).cpu()
        danger_p = s1_danger(s1_policy, obs, mm)
        for j, i in enumerate(idx):
            e, t = ds.index[i]
            ep = episodes[e]
            plan = schema.decode(intents[e][t])
            prev = schema.format_plan(schema.decode(intents[e][t - stride])) if t >= stride else None
            if cfg.name == "lol":
                text = stats_from_vector(ep.array("aux")[t]).text() if ep.has("aux") else "no live data"
                rationale = lol_rationale(text, plan)
            else:
                hud = bullet_hell_hud(ep, t)
                hud.pop("clearance", None)  # privileged in the simulator: use System 1's own danger estimate
                hud["danger_p"] = float(danger_p[j])
                text = bullet_hell_state_text(hud)
                rationale = bullet_hell_rationale(hud, plan)
            prompt = make_prompt(cfg.title, text, None, prev)
            comp = make_completion(schema.format_plan(plan), rationale, think)
            samples.append(S2Sample(vis[j], prompt, comp, intents[e][t]))
    return samples


@torch.no_grad()
def s1_danger(s1_policy, obs: torch.Tensor, minimap: torch.Tensor | None = None) -> np.ndarray:
    """System 1's amygdala: P(life lost within the horizon) — the danger signal System 2 reads."""
    out = s1_policy(obs, minimap=minimap)
    return torch.sigmoid(out.danger).float().cpu().numpy()


def _batches(n: int, size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for s in range(0, n, size):
        yield order[s : s + size]


@torch.no_grad()
def evaluate_s2(pf: Prefrontal, samples: list[S2Sample], max_samples: int = 128, batch: int = 32) -> dict[str, float]:
    pf.eval()
    samples = samples[:max_samples]
    parsed = exact = 0
    slot_hits = np.zeros(len(pf.schema.slots))
    for s in range(0, len(samples), batch):
        chunk = samples[s : s + batch]
        vis = torch.stack([c.vis for c in chunk])
        outs = pf.plan(vis, [c.prompt for c in chunk], max_new_tokens=200)
        for (text, plan), c in zip(outs, chunk):
            if plan is None:
                continue
            parsed += 1
            ids = pf.schema.encode(plan)
            slot_hits += ids == c.intent
            exact += int((ids == c.intent).all())
    n = max(len(samples), 1)
    out = {"parse_rate": parsed / n, "exact": exact / n}
    for k, h in zip(pf.schema.names, slot_hits):
        out[f"slot_acc/{k}"] = float(h / n)
    return out


def train_s2_sl(pf: Prefrontal, train: list[S2Sample], val: list[S2Sample] | None, epochs: int = 3, batch_size: int = 16,
                lr: float = 2e-4, lr_projector: float = 1e-3, device: torch.device | str = "cpu",
                log_fn: Callable[[str], None] = print, eval_every_epoch: bool = True, seed: int = 0) -> list[dict[str, Any]]:
    proj = list(pf.projector.parameters()) + [pf.vis_pos]
    proj_ids = {id(p) for p in proj}
    other = [p for p in pf.trainable_parameters() if id(p) not in proj_ids]
    opt = torch.optim.AdamW([{"params": proj, "lr": lr_projector}, {"params": other, "lr": lr}], weight_decay=0.0)
    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        pf.train()
        losses = []
        for idx in _batches(len(train), batch_size, rng):
            chunk = [train[i] for i in idx]
            vis = torch.stack([c.vis for c in chunk]).to(device)
            loss = pf.loss(vis, [c.prompt for c in chunk], [c.completion for c in chunk])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pf.trainable_parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        rec: dict[str, Any] = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_s": time.time() - t0}
        if val and eval_every_epoch:
            rec.update({f"val/{k}": v for k, v in evaluate_s2(pf, val).items()})
        history.append(rec)
        log_fn(f"[system2-sl] epoch {epoch} " + ", ".join(f"{k}={v:.3f}" for k, v in rec.items() if isinstance(v, float)))
    return history


# ---------------------------------------------------------------------------------------------
# DanmakuDojo helpers
# ---------------------------------------------------------------------------------------------


def dojo_huds(info: dict[str, np.ndarray], danger_p: np.ndarray) -> list[dict[str, Any]]:
    """HUD facts System 2 may read: what probes give in real games + System 1's danger estimate."""
    out = []
    for i in range(len(info["lives"])):
        x, y = info["player_xy"][i]  # real games: a calibrated 'locate' probe on the ship sprite
        out.append({
            "lives": int(info["lives"][i]), "bombs": int(info["bombs"][i]), "score": int(info["score"][i]),
            "boss_hp": float(info["boss_hp"][i]), "danger_p": float(danger_p[i]), "zone": zone_of(float(x), float(y)),
        })
    return out


@torch.no_grad()
def plan_for_envs(pf: Prefrontal, s1, obs: np.ndarray, info: dict, previous: list[str | None], device, temperature: float = 0.0,
                  max_new_tokens: int = 200) -> tuple[np.ndarray, list[str | None], int]:
    o = torch.from_numpy(obs).to(device)
    vis = s1_visual_tokens(s1, o)
    huds = dojo_huds(info, s1_danger(s1, o))
    prompts = [make_prompt(pf.title, bullet_hell_state_text(h), None, previous[i]) for i, h in enumerate(huds)]
    results = pf.plan(vis, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
    ids, prev, invalid = [], [], 0
    for text, plan in results:
        if plan is None:
            invalid += 1
            ids.append(pf.schema.encode(None))
            prev.append(None)
        else:
            ids.append(pf.schema.encode(plan))
            prev.append(pf.schema.format_plan(plan))
    return np.stack(ids), prev, invalid


@torch.no_grad()
def play_with_planner(s1, pf: Prefrontal | None, make_env, episodes: int, schema: IntentSchema, replan_every: int = 10,
                      device: torch.device | str = "cpu", seed: int = 777, max_steps: int = 100_000,
                      fixed_intent: dict[str, str] | None = None) -> dict[str, float]:
    """System 1 alone (pf=None), with a fixed intent, or steered by System 2 every ``replan_every`` ticks."""
    s1.eval()
    env = make_env(seed)
    obs, info = env.reset(seed)
    n = obs.shape[0]
    intents = np.tile(schema.encode(fixed_intent), (n, 1))
    previous: list[str | None] = [None] * n
    done_eps: list[dict] = []
    invalid = plans = 0
    for step in range(max_steps):
        if pf is not None and step % replan_every == 0:
            intents, previous, bad = plan_for_envs(pf, s1, obs, info, previous, device)
            invalid += bad
            plans += n
        a, *_ = s1.act(torch.from_numpy(obs).to(device), intent=torch.from_numpy(intents).to(device))
        obs, r, term, trunc, info = env.step({k: v.cpu().numpy() for k, v in a.items()})
        for i in np.nonzero(term | trunc)[0]:
            done_eps.append({k: float(v[i]) for k, v in info.items() if k.startswith("episode_")})
            previous[i] = None
        if len(done_eps) >= episodes:
            break
    done_eps = done_eps[:episodes]
    out: dict[str, float] = {"episodes": float(len(done_eps))}
    for key in ("episode_return", "episode_length", "episode_deaths", "episode_score", "episode_clear", "episode_bombs"):
        vals = np.array([e[key] for e in done_eps if key in e])
        if len(vals):
            out[key] = float(vals.mean())
            out[key + "_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    if plans:
        out["invalid_plan_rate"] = invalid / plans
    return out


def grpo_train(pf: Prefrontal, s1, schema: IntentSchema, make_driver: Callable[[int], Any], steps: int = 200, group: int = 4,
               horizon: int = 30, beta: float = 0.04, lr: float = 5e-5, temperature: float = 1.0, advance: int = 15,
               invalid_penalty: float = 0.5, device: torch.device | str = "cpu", seed: int = 0,
               log_fn: Callable[[str], None] = print) -> list[dict[str, float]]:
    ref_state = pf.adapter_state()
    opt = torch.optim.AdamW(pf.trainable_parameters(), lr=lr, weight_decay=0.0)
    driver = make_driver(seed)
    obs, info = driver.reset(seed)
    n = obs.shape[0]
    previous: list[str | None] = [None] * n
    history = []
    s1.eval()
    for step in range(steps):
        t0 = time.time()
        pf.eval()
        with torch.no_grad():
            o = torch.from_numpy(obs).to(device)
            vis = s1_visual_tokens(s1, o)
            huds = dojo_huds(info, s1_danger(s1, o))
            prompts = [make_prompt(pf.title, bullet_hell_state_text(h), None, previous[i]) for i, h in enumerate(huds)]
            rep_vis = vis.repeat_interleave(group, 0)
            rep_prompts = [p for p in prompts for _ in range(group)]
            texts = pf.generate(rep_vis, rep_prompts, max_new_tokens=200, temperature=temperature)
        plans = [schema.parse_plan(t) for t in texts]
        intents = np.stack([schema.encode(p) for p in plans])
        # score every plan by playing it from an exact clone of the state (common random numbers)
        clones = [driver.replicate(i, group) for i in range(n)]
        returns = np.zeros(n * group)
        alive = np.ones(n * group, bool)
        cobs = np.concatenate([c._obs() for c in clones])
        it = torch.from_numpy(intents).to(device)
        with torch.no_grad():
            for _ in range(horizon):
                a, *_ = s1.act(torch.from_numpy(cobs).to(device), intent=it)
                a = {k: v.cpu().numpy() for k, v in a.items()}
                outs = []
                for i, c in enumerate(clones):
                    sl = slice(i * group, (i + 1) * group)
                    o, r, term, trunc, _ = c.step({k: v[sl] for k, v in a.items()})
                    returns[sl] += r * alive[sl]
                    alive[sl] &= ~(term | trunc)
                    outs.append(o)
                cobs = np.concatenate(outs)
        returns -= invalid_penalty * np.array([p is None for p in plans])
        R = returns.reshape(n, group)
        std = R.std(1, keepdims=True)
        adv = np.where(std > 1e-6, (R - R.mean(1, keepdims=True)) / (std + 1e-6), 0.0).reshape(-1)
        # policy gradient (single on-policy step) + KL to the supervised reference
        pf.train()
        with torch.no_grad(), pf.swapped(ref_state):  # reference first: swapping weights after the
            ref_lp, _ = pf.token_logprobs(rep_vis, rep_prompts, texts)  # forward would corrupt autograd
        lp, valid = pf.token_logprobs(rep_vis, rep_prompts, texts)
        ntok = valid.sum(-1).clamp_min(1)
        seq_lp = lp.sum(-1) / ntok
        d = (ref_lp - lp) * valid
        kl = ((torch.exp(d) - d - 1) * valid).sum(-1) / ntok
        adv_t = torch.tensor(adv, dtype=seq_lp.dtype, device=seq_lp.device)
        loss = (-(adv_t * seq_lp) + beta * kl).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pf.trainable_parameters(), 1.0)
        opt.step()
        # move the driver on with the best plan of every group
        best = intents.reshape(n, group, -1)[np.arange(n), R.argmax(1)]
        with torch.no_grad():
            for _ in range(advance):
                a, *_ = s1.act(torch.from_numpy(obs).to(device), intent=torch.from_numpy(best).to(device))
                obs, r, term, trunc, info = driver.step({k: v.cpu().numpy() for k, v in a.items()})
        for i in range(n):
            plan = plans[i * group + int(R[i].argmax())]
            previous[i] = schema.format_plan(plan) if plan is not None else None
        rec = {"step": step, "return_mean": float(R.mean()), "return_spread": float(std.mean()), "kl": float(kl.detach().mean()),
               "invalid": float(np.mean([p is None for p in plans])), "seconds": time.time() - t0}
        history.append(rec)
        if step % 10 == 0 or step == steps - 1:
            log_fn(f"[system2-grpo] step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in rec.items() if k != "step"))
    return history
