"""On-policy diagnostics for DanmakuDojo checkpoints (exposes the BC "copycat" failure mode).

For states visited by the *policy itself*: how often its action matches the (noise-free) expert, how
often it repeats its own previous action, and how much safety it gives up vs. the expert's best move.

    python scripts/onpolicy_agreement.py --scheme keyboard --ckpt runs/dojo_keyboard/bc/best.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hakisa.envs.dojo import DanmakuDojo
from hakisa.envs.dojo_expert import ScriptedExpert
from hakisa.models import load_policy


def diagnose(ckpt: str, scheme: str, difficulty: float = 2.0, steps: int = 300, seed: int = 5) -> dict[str, float]:
    pol = load_policy(ckpt).eval()
    env = DanmakuDojo(num_envs=16, scheme=scheme, difficulty=difficulty, seed=seed)
    ex = ScriptedExpert(env, lapse_prob=0.0, noise_prob=0.0, seed=0)
    obs, _ = env.reset(seed)
    agree, repeat, gap = [], [], []
    prev = None
    key = "move" if scheme == "keyboard" else None
    for _ in range(steps):
        lab = ex.act()
        pos, target = ex._future_positions()
        safe = np.minimum(ex._clearance(pos), 8.0)
        with torch.no_grad():
            a, *_ = pol.act(torch.from_numpy(obs))
        a = {k: v.numpy() for k, v in a.items()}
        if key:
            agree.append((a[key] == lab[key]).mean())
            cand = a["move"] * 2 + a["focus"]
            cur = a["move"]
        else:  # mouse: agreement = target within 3 fine cells of the expert's target
            d = np.abs(a["pointer_x"] - lab["pointer_x"]) + np.abs(a["pointer_y"] - lab["pointer_y"])
            agree.append((d <= 3).mean())
            tx, ty = env.decode_target(a["pointer_x"], a["pointer_y"])
            cand_pos = target  # (N, C, 2) candidate targets of the expert
            dist = np.hypot(cand_pos[..., 0] - tx[:, None], cand_pos[..., 1] - ty[:, None])
            cand = np.argmin(dist + 1e3 * (ex.cand_focus[None, :] != a["slow"][:, None]), axis=1)
            cur = np.stack([a["pointer_x"], a["pointer_y"]], 1)
        gap.append(float(np.mean(safe.max(1) - safe[np.arange(env.N), cand])))
        if prev is not None:
            repeat.append(float(np.mean(np.all(np.atleast_2d(cur.T).T == np.atleast_2d(prev.T).T, axis=-1)
                                        if cur.ndim > 1 else cur == prev)))
        prev = cur
        obs, *_ = env.step(a)
    return {"onpolicy_agreement": float(np.mean(agree)), "repeats_previous": float(np.mean(repeat)),
            "safety_gap": float(np.mean(gap))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["keyboard", "mouse"], default="keyboard")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--difficulty", type=float, default=2.0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    for c in args.ckpt:
        print(json.dumps({"ckpt": c, **diagnose(c, args.scheme, args.difficulty)}))


if __name__ == "__main__":
    main()
