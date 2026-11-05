"""End-to-end evidence on DanmakuDojo: demonstrations -> Study (BC) -> Consolidation (PPO) -> evaluation.

    python scripts/dojo_experiments.py --scheme keyboard --stage all
    python scripts/dojo_experiments.py --scheme mouse --stage all

Writes ``<out>/results.json``, ``<out>/learning_curves.png`` and checkpoints under ``runs/``.
All numbers come from the runs this script performs (fixed evaluation seeds, 95% CIs = 1.96 * SE).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hakisa.config import load_game_config
from hakisa.consolidation import PPOConfig, PPOTrainer, evaluate_policy
from hakisa.dojo_tools import dojo_env_factory, record_dojo_dagger, record_dojo_demos
from hakisa.envs.dojo import DanmakuDojo
from hakisa.envs.dojo_expert import ScriptedExpert
from hakisa.hippocampus.dataset import AugmentConfig, BCDataset, split_episodes
from hakisa.hippocampus.storage import load_episodes
from hakisa.models import build_policy, load_policy
from hakisa.study import StudyConfig, evaluate_bc, train_bc


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def eval_baselines(args, make) -> dict:
    res = {}
    rng = np.random.default_rng(0)
    env = make(args.eval_seed)
    res["random"] = evaluate_policy(None, lambda s: env, args.eval_episodes, actor=lambda o: env.random_actions(rng), seed=args.eval_seed)
    log(f"random: {res['random']}")
    env2 = make(args.eval_seed)
    demo = ScriptedExpert(env2, lapse_prob=0.08, noise_prob=0.03, seed=1)
    res["demonstrator"] = evaluate_policy(None, lambda s: env2, args.eval_episodes, actor=lambda o: demo.act(), seed=args.eval_seed)
    log(f"demonstrator: {res['demonstrator']}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["keyboard", "mouse"], default="keyboard")
    ap.add_argument("--stage", default="all", help="all | demos | baselines | bc | rl | report")
    ap.add_argument("--difficulty", type=float, default=2.0)
    ap.add_argument("--demo-steps", type=int, default=100_000)
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--bc-batch", type=int, default=128)
    ap.add_argument("--rl-steps", type=int, default=1_000_000)
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--dagger-steps", type=int, default=40_000)
    ap.add_argument("--dagger-epochs", type=int, default=1)
    ap.add_argument("--eval-episodes", type=int, default=32)
    ap.add_argument("--eval-seed", type=int, default=777)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    game = "dojo_keyboard" if args.scheme == "keyboard" else "dojo_mouse"
    data = Path(args.data or f"data/{game}")
    runs = Path(args.runs or f"runs/{game}")
    out = Path(args.out or f"docs/results/{game}")
    out.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["setup"] = {"game": game, "difficulty": args.difficulty, "demo_steps": args.demo_steps, "bc_epochs": args.bc_epochs,
                        "rl_steps": args.rl_steps, "eval_episodes": args.eval_episodes, "eval_seed": args.eval_seed,
                        "device": "cpu", "threads": args.threads}
    cfg = load_game_config(game)
    make = dojo_env_factory(args.scheme, 16, args.difficulty)
    device = torch.device("cpu")

    def save():
        results_path.write_text(json.dumps(results, indent=2, default=float))

    stages = ["demos", "baselines", "bc", "dagger", "rl"] if args.stage == "all" else args.stage.split(",")

    if "demos" in stages:
        t0 = time.time()
        paths = record_dojo_demos(data, args.scheme, args.demo_steps, difficulty=args.difficulty, log_fn=log)
        results["demos"] = {"episodes": len(paths), "frames": args.demo_steps, "seconds": time.time() - t0}
        save()

    if "baselines" in stages:
        results["baselines"] = eval_baselines(args, make)
        save()

    if "bc" in stages:
        eps = load_episodes([data])
        train_eps, val_eps = split_episodes(eps, 0.1, seed=0)
        bc_cfg = cfg.training.get("bc", {})
        aug = AugmentConfig(shift=3, flip=bool(bc_cfg.get("flip", True)), color=0.1,
                            history_dropout=float(bc_cfg.get("history_dropout", 0.0)))
        train_ds = BCDataset(train_eps, cfg, augment=aug)
        val_ds = BCDataset(val_eps, cfg)
        log(f"BC data: {len(train_ds)} train / {len(val_ds)} val frames from {len(eps)} episodes")
        scfg = StudyConfig.from_dict(dict(bc_cfg, epochs=args.bc_epochs, batch_size=args.bc_batch, num_workers=0, amp=False))
        policy = build_policy(cfg).to_channels_last()
        t0 = time.time()
        hist = train_bc(policy, train_ds, val_ds, scfg, runs / "bc", device, log_fn=log)
        best = load_policy(runs / "bc" / "best.pt").to_channels_last()
        results["bc"] = {
            "train_seconds": time.time() - t0,
            "params": best.num_parameters(),
            "val": evaluate_bc(best, val_ds, device),
            "epochs": [{k: v for k, v in e.items() if k.startswith("val/nll/total") or k in ("epoch", "elapsed_s")} for e in hist["epochs"]],
        }
        results["bc"]["play_sampled"] = evaluate_policy(best, make, args.eval_episodes, seed=args.eval_seed)
        log(f"BC (sampled): {results['bc']['play_sampled']}")
        results["bc"]["play_greedy"] = evaluate_policy(best, make, args.eval_episodes, deterministic=True, seed=args.eval_seed)
        log(f"BC (greedy): {results['bc']['play_greedy']}")
        save()

    if "dagger" in stages:
        # "Play & teach": the learner plays, the expert labels the states the learner actually visits
        # (on real games: your F8 takeover corrections, HG-DAgger)
        current = runs / "bc" / "best.pt"
        dag_root = Path(str(data) + "_dagger")
        results["dagger"] = []
        aug = AugmentConfig(shift=3, flip=True, color=0.1, history_dropout=float(cfg.training.get("bc", {}).get("history_dropout", 0.0)))
        for k in range(args.dagger_rounds):
            learner = load_policy(current).to_channels_last().eval()

            def act_fn(obs, learner=learner):
                with torch.no_grad():
                    a, *_ = learner.act(torch.from_numpy(obs))
                return {key: v.numpy() for key, v in a.items()}

            t0 = time.time()
            record_dojo_dagger(dag_root / f"round{k}", args.scheme, act_fn, args.dagger_steps, difficulty=args.difficulty,
                               seed=1000 + 17 * k, log_fn=log)
            eps = load_episodes([data] + [dag_root / f"round{j}" for j in range(k + 1)])
            train_eps, val_eps = split_episodes(eps, 0.1, seed=0)
            train_ds = BCDataset(train_eps, cfg, augment=aug)
            val_ds = BCDataset(val_eps, cfg)
            log(f"DAgger round {k}: {len(train_ds)} train frames")
            scfg = StudyConfig.from_dict(dict(cfg.training.get("bc", {}), epochs=args.dagger_epochs, batch_size=args.bc_batch,
                                              lr=5e-4, warmup_frac=0.02, num_workers=0, amp=False))
            pol = load_policy(current).to_channels_last()
            train_bc(pol, train_ds, val_ds, scfg, runs / f"dagger{k}", device, log_fn=log)
            current = runs / f"dagger{k}" / "best.pt"
            best = load_policy(current).to_channels_last()
            rec = {"round": k, "train_frames": len(train_ds), "seconds": time.time() - t0,
                   "play_sampled": evaluate_policy(best, make, args.eval_episodes, seed=args.eval_seed),
                   "play_greedy": evaluate_policy(best, make, args.eval_episodes, deterministic=True, seed=args.eval_seed)}
            log(f"DAgger round {k} (sampled): {rec['play_sampled']}")
            results["dagger"].append(rec)
            results["dagger_final"] = str(current)
            save()

    if "rl" in stages:
        init = Path(results.get("dagger_final") or (runs / "bc" / "best.pt"))
        log(f"RL starts from {init}")
        bc_policy = load_policy(init).to_channels_last()
        ref = load_policy(init).to_channels_last()
        eps = load_episodes([data] + sorted(Path(str(data) + "_dagger").glob("round*")))
        demo_ds = BCDataset(eps, cfg, augment=AugmentConfig(shift=3, flip=True, color=0.1))
        pcfg = PPOConfig.from_dict(cfg.training.get("rl", {}))
        env = DanmakuDojo(num_envs=pcfg.num_envs, scheme=args.scheme, difficulty=args.difficulty, seed=1)
        trainer = PPOTrainer(bc_policy, pcfg, device, ref_policy=ref, demo_ds=demo_ds, out_dir=runs / "rl", log_fn=log)

        def quick_eval(p):
            return evaluate_policy(p, make, 16, seed=args.eval_seed + 1)

        t0 = time.time()
        hist = trainer.train(env, args.rl_steps, eval_fn=quick_eval, eval_every=25)
        rl_policy = trainer.policy
        results["rl"] = {
            "train_seconds": time.time() - t0,
            "config": vars(pcfg),
            "curve": [{k: v for k, v in h.items() if k.startswith(("env_steps", "train/", "eval/", "kl_ref", "explained_var", "warmup"))} for h in hist],
        }
        results["rl"]["play_sampled"] = evaluate_policy(rl_policy, make, args.eval_episodes, seed=args.eval_seed)
        log(f"BC+RL (sampled): {results['rl']['play_sampled']}")
        results["rl"]["play_greedy"] = evaluate_policy(rl_policy, make, args.eval_episodes, deterministic=True, seed=args.eval_seed)
        log(f"BC+RL (greedy): {results['rl']['play_greedy']}")
        save()

    if args.stage in ("all", "report"):
        write_report(results, out)
    save()


def write_report(results: dict, out: Path) -> None:
    rows = []
    def row(name, r):
        if not r:
            return
        ci = lambda k: 1.96 * r.get(k + "_se", 0.0)
        rows.append(
            f"| {name} | {r['episode_return']:.2f} ± {ci('episode_return'):.2f} | {r['episode_deaths']:.2f} ± {ci('episode_deaths'):.2f} | "
            f"{r['episode_length'] / 20:.1f} | {r['episode_score']:.0f} | {100 * r['episode_clear']:.0f}% | {r.get('episode_bombs', 0):.2f} |"
        )
    b = results.get("baselines", {})
    row("Random", b.get("random"))
    row("Demonstrator (scripted 'human')", b.get("demonstrator"))
    row("Study / BC (sampled)", results.get("bc", {}).get("play_sampled"))
    row("Study / BC (greedy)", results.get("bc", {}).get("play_greedy"))
    for d in results.get("dagger", []):
        row(f"+ Play & teach (DAgger round {d['round'] + 1}, sampled)", d.get("play_sampled"))
        row(f"+ Play & teach (DAgger round {d['round'] + 1}, greedy)", d.get("play_greedy"))
    row("BC + Consolidation / RL (sampled)", results.get("rl", {}).get("play_sampled"))
    row("BC + Consolidation / RL (greedy)", results.get("rl", {}).get("play_greedy"))
    row("BC + Consolidation / RL (temperature 0.5)", results.get("rl", {}).get("play_t05"))
    s = results.get("setup", {})
    md = [
        f"# {s.get('game')} — difficulty {s.get('difficulty')}",
        "",
        f"Evaluation: {s.get('eval_episodes')} full episodes per agent, fixed seeds, ±95% CI. Survival is in seconds (20 ticks/s, max 90 s).",
        (f"RL was stopped early at {results['rl'].get('env_steps')} environment steps (iteration "
         f"{results['rl'].get('checkpoint_iteration')}) because of the CPU budget; the last saved checkpoint is evaluated."
         if results.get("rl", {}).get("stopped_early") else ""),
        "",
        "| Agent | Return | Deaths / episode | Survival (s) | Score | Stage clear | Bombs |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ]
    (out / "README.md").write_text("\n".join(md) + "\n")
    curve = results.get("rl", {}).get("curve")
    if curve:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
            xs = [c["env_steps"] for c in curve if "train/episode_return" in c]
            ax[0].plot(xs, [c["train/episode_return"] for c in curve if "train/episode_return" in c], label="training episodes")
            ev = [c for c in curve if "eval/episode_return" in c]
            if ev:
                ax[0].plot([c["env_steps"] for c in ev], [c["eval/episode_return"] for c in ev], "o-", label="evaluation")
            ax[0].set_xlabel("environment steps")
            ax[0].set_ylabel("episode return")
            ax[0].legend()
            ax[1].plot([c["env_steps"] for c in curve], [c.get("kl_ref", 0) for c in curve])
            ax[1].set_xlabel("environment steps")
            ax[1].set_ylabel("KL(pi || pi_BC)")
            fig.tight_layout()
            fig.savefig(out / "learning_curves.png", dpi=120)
        except Exception as exc:  # pragma: no cover
            print("plot failed:", exc)


if __name__ == "__main__":
    main()
