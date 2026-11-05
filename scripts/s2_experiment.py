"""System 1 + System 2 on DanmakuDojo (keyboard scheme).

    python scripts/s2_experiment.py --llm tiny-qwen3          # what runs in the cloud container
    python scripts/s2_experiment.py --llm Qwen/Qwen3-0.6B --device cuda   # the real thing (GPU)

Steps: hindsight plans -> intent-conditioned System 1 (fine-tuned from the BC checkpoint) ->
System 2 SL -> play (S1 alone vs S1+S2) -> GRPO -> play again.  Results: docs/results/system2/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hakisa.config import load_game_config
from hakisa.envs.dojo import DanmakuDojo
from hakisa.hippocampus.dataset import AugmentConfig, BCDataset, split_episodes
from hakisa.hippocampus.storage import load_episodes
from hakisa.models import load_policy
from hakisa.prefrontal.intents import IntentSchema, hindsight_for_episode
from hakisa.prefrontal.llm import load_llm, tiny_qwen3, ByteTokenizer
from hakisa.prefrontal.reasoner import Prefrontal
from hakisa.prefrontal.train import build_s2_samples, evaluate_s2, grpo_train, play_with_planner, train_s2_sl
from hakisa.study import StudyConfig, train_bc


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="tiny-qwen3")
    ap.add_argument("--train-mode", default=None, help="lora | full (default: full for tiny, lora otherwise)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default="data/dojo_keyboard")
    ap.add_argument("--bc", default="runs/dojo_keyboard/bc/best.pt")
    ap.add_argument("--runs", default="runs/dojo_keyboard")
    ap.add_argument("--out", default="docs/results/system2")
    ap.add_argument("--difficulty", type=float, default=2.0)
    ap.add_argument("--s1-epochs", type=int, default=1)
    ap.add_argument("--s2-epochs", type=int, default=3)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--grpo-steps", type=int, default=150)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--replan-every", type=int, default=10)
    ap.add_argument("--think", action="store_true", default=True)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-s1", action="store_true")
    ap.add_argument("--episode-ticks", type=int, default=1800, help="episode length cap (smoke tests use less)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs)
    cfg = load_game_config("dojo_keyboard")
    schema = IntentSchema.from_config(cfg)
    results: dict = {"setup": vars(args)}

    def save():
        (out / "results.json").write_text(json.dumps(results, indent=2, default=float))

    eps = load_episodes([args.data])
    intents = [hindsight_for_episode(ep, cfg, schema) for ep in eps]
    counts = {name: np.bincount(np.concatenate([i[:, j] for i in intents]), minlength=len(opts)).tolist()
              for j, (name, opts) in enumerate(schema.slots.items())}
    results["hindsight_counts"] = counts
    log(f"hindsight intent counts: {counts}")

    # 1. intent-conditioned System 1 (continue from the BC checkpoint)
    s1_dir = runs / "s1_intent"
    if not args.skip_s1 or not (s1_dir / "best.pt").exists():
        idx = np.random.default_rng(0).permutation(len(eps))
        n_val = max(1, len(eps) // 10)
        val_ids, train_ids = idx[:n_val], idx[n_val:]
        tr = BCDataset([eps[i] for i in train_ids], cfg, augment=AugmentConfig(shift=3, flip=True, color=0.1),
                       intents=[intents[i] for i in train_ids], intent_flip=schema.flip)
        va = BCDataset([eps[i] for i in val_ids], cfg, intents=[intents[i] for i in val_ids])
        s1 = load_policy(args.bc).to_channels_last()
        scfg = StudyConfig(epochs=args.s1_epochs, batch_size=128, lr=3e-4, amp=False, intent_dropout=0.3, warmup_frac=0.02)
        t0 = time.time()
        train_bc(s1, tr, va, scfg, s1_dir, device, log_fn=log)
        results["s1_intent_train_s"] = time.time() - t0
        save()
    s1 = load_policy(s1_dir / "best.pt", device).eval()
    def make(seed: int) -> DanmakuDojo:
        return DanmakuDojo(num_envs=16, scheme="keyboard", difficulty=args.difficulty, seed=seed, max_steps=args.episode_ticks)

    results["s1_alone"] = play_with_planner(s1, None, make, args.episodes, schema, device=device)
    log(f"S1 alone (null intent): {results['s1_alone']}")
    save()

    # 2. System 2 supervised on hindsight plans
    train_eps, val_eps = split_episodes(eps, 0.1, seed=1)
    train_s = build_s2_samples(train_eps, cfg, s1, schema, stride=args.stride, think=args.think, device=device)
    val_s = build_s2_samples(val_eps, cfg, s1, schema, stride=args.stride * 3, think=args.think, device=device)
    log(f"System 2 samples: {len(train_s)} train / {len(val_s)} val")
    if args.llm == "tiny-qwen3":
        llm, tok = tiny_qwen3(ByteTokenizer().vocab_size, hidden=192, layers=3, heads=6), ByteTokenizer()
        llm = llm.to(device)
    else:
        llm, tok = load_llm(args.llm, device, torch.bfloat16 if device.type == "cuda" else torch.float32)
    mode = args.train_mode or ("full" if args.llm == "tiny-qwen3" else "lora")
    pf = Prefrontal(llm, tok, vis_dim=train_s[0].vis.shape[-1], schema=schema, title=cfg.title,
                    n_vis=train_s[0].vis.shape[0], spec=args.llm, train_mode=mode).to(device)
    results["llm"] = {"spec": args.llm, "mode": mode, "params_M": sum(p.numel() for p in pf.parameters()) / 1e6,
                      "trainable_M": sum(p.numel() for p in pf.trainable_parameters()) / 1e6}
    hist = train_s2_sl(pf, train_s, val_s, epochs=args.s2_epochs, batch_size=16,
                       lr=1e-3 if mode == "full" else 2e-4, device=device, log_fn=log)
    results["s2_sl"] = hist
    pf.save(runs / "s2_sl")
    results["s2_sl_val"] = evaluate_s2(pf, val_s, max_samples=256)
    log(f"S2 SL plan accuracy: {results['s2_sl_val']}")
    results["s1_s2_sl"] = play_with_planner(s1, pf, make, args.episodes, schema, replan_every=args.replan_every, device=device)
    log(f"S1 + S2 (SL): {results['s1_s2_sl']}")
    save()

    # 3. GRPO against game outcomes
    if args.grpo_steps > 0:
        t0 = time.time()
        g = grpo_train(pf, s1, schema, lambda seed: DanmakuDojo(num_envs=8, scheme="keyboard", difficulty=args.difficulty, seed=seed + 99),
                       steps=args.grpo_steps, group=4, horizon=30, beta=0.04, lr=1e-4 if mode == "full" else 2e-5,
                       device=device, log_fn=log)
        results["grpo"] = {"history": g, "seconds": time.time() - t0}
        pf.save(runs / "s2_grpo")
        results["s1_s2_grpo"] = play_with_planner(s1, pf, make, args.episodes, schema, replan_every=args.replan_every, device=device)
        log(f"S1 + S2 (GRPO): {results['s1_s2_grpo']}")
        save()
    write_report(results, out)
    save()


def write_report(results: dict, out: Path) -> None:
    def row(name, r):
        if not r:
            return ""
        ci = lambda k: 1.96 * r.get(k + "_se", 0.0)
        extra = f" {100 * r['invalid_plan_rate']:.1f}%" if "invalid_plan_rate" in r else " —"
        return (f"| {name} | {r['episode_return']:.2f} ± {ci('episode_return'):.2f} | {r['episode_deaths']:.2f} ± {ci('episode_deaths'):.2f} | "
                f"{r['episode_length'] / 20:.1f} | {r['episode_score']:.0f} | {r.get('episode_bombs', 0):.2f} |{extra} |")

    llm = results.get("llm", {})
    md = [
        "# System 1 + System 2 on DanmakuDojo (keyboard, difficulty %s)" % results["setup"].get("difficulty"),
        "",
        f"LLM: `{llm.get('spec')}` ({llm.get('params_M', 0):.2f} M parameters, {llm.get('mode')} training)."
        + (" **This is a tiny randomly initialised Qwen3-architecture model, NOT the pre-trained Qwen3-0.6B** "
           "(the Hugging Face Hub was unreachable from the container). It validates the pipeline only." if llm.get("spec") == "tiny-qwen3" else ""),
        "",
        f"Plan accuracy of the supervised planner vs. hindsight plans (validation): {json.dumps(results.get('s2_sl_val', {}))}",
        "",
        "| Agent | Return | Deaths / episode | Survival (s) | Score | Bombs | Invalid plans |",
        "|---|---|---|---|---|---|---|",
        row("System 1 alone (intent-conditioned policy, null intent)", results.get("s1_alone")),
        row("System 1 + System 2 (supervised planner)", results.get("s1_s2_sl")),
        row("System 1 + System 2 (after GRPO)", results.get("s1_s2_grpo")),
    ]
    (out / "README.md").write_text("\n".join(x for x in md if x is not None) + "\n")


if __name__ == "__main__":
    main()
