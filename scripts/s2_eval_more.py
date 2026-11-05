"""Larger paired evaluation of System 1 alone vs System 1 + System 2 (fresh seeds, more episodes)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hakisa.config import load_game_config
from hakisa.envs.dojo import DanmakuDojo
from hakisa.models import load_policy
from hakisa.prefrontal.intents import IntentSchema
from hakisa.prefrontal.reasoner import Prefrontal
from hakisa.prefrontal.train import play_with_planner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="docs/results/system2/results_more_episodes.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_game_config("dojo_keyboard")
    schema = IntentSchema.from_config(cfg)
    s1 = load_policy("runs/dojo_keyboard/s1_intent/best.pt").eval()

    def make(seed: int) -> DanmakuDojo:
        return DanmakuDojo(num_envs=16, scheme="keyboard", difficulty=2.0, seed=seed)

    res = {"episodes": args.episodes, "seed": args.seed}
    res["s1_alone"] = play_with_planner(s1, None, make, args.episodes, schema, seed=args.seed)
    print("S1 alone", res["s1_alone"], flush=True)
    for name in ("s2_sl", "s2_grpo"):
        pf = Prefrontal.load(f"runs/dojo_keyboard/{name}").eval()
        res[f"s1_{name}"] = play_with_planner(s1, pf, make, args.episodes, schema, seed=args.seed)
        print(name, res[f"s1_{name}"], flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
