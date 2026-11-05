"""Evaluate the latest RL checkpoint of a (possibly stopped-early) DanmakuDojo run and update its results.

    python scripts/finalize_rl.py --scheme keyboard

Reads runs/<game>/rl/last.pt and runs/<game>/rl/consolidation_log.jsonl, evaluates with the same protocol
as scripts/dojo_experiments.py (fixed seeds, 32 episodes, sampled + greedy) and writes results["rl"].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hakisa.consolidation import evaluate_policy
from hakisa.dojo_tools import dojo_env_factory
from hakisa.models import load_policy
from hakisa.utils import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["keyboard", "mouse"], required=True)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--difficulty", type=float, default=2.0)
    ap.add_argument("--eval-seed", type=int, default=777)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    game = f"dojo_{args.scheme}"
    run = Path(f"runs/{game}/rl")
    out = Path(f"docs/results/{game}")
    results = json.loads((out / "results.json").read_text())
    ckpt = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    policy = load_policy(run / "last.pt").to_channels_last()
    make = dojo_env_factory(args.scheme, 16, args.difficulty)
    log = read_jsonl(run / "consolidation_log.jsonl")
    curve = [{k: v for k, v in h.items() if k.startswith(("env_steps", "train/", "eval/", "kl_ref", "explained_var", "warmup"))}
             for h in log]
    rl = results.get("rl", {})
    rl.update({
        "checkpoint_iteration": ckpt.get("extra", {}).get("iteration"),
        "env_steps": (ckpt.get("extra", {}).get("stats") or {}).get("env_steps"),
        "stopped_early": True,
        "curve": curve,
    })
    rl["play_sampled"] = evaluate_policy(policy, make, args.episodes, seed=args.eval_seed)
    print("RL (sampled):", rl["play_sampled"], flush=True)
    rl["play_greedy"] = evaluate_policy(policy, make, args.episodes, deterministic=True, seed=args.eval_seed)
    print("RL (greedy):", rl["play_greedy"], flush=True)
    rl["play_t05"] = evaluate_policy(policy, make, args.episodes, seed=args.eval_seed, temperature=0.5)
    print("RL (T=0.5):", rl["play_t05"], flush=True)
    results["rl"] = rl
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    import importlib.util

    spec = importlib.util.spec_from_file_location("dojo_experiments", Path(__file__).with_name("dojo_experiments.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.write_report(results, out)


if __name__ == "__main__":
    main()
