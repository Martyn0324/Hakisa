"""Command-line interface: ``hakisa <command> --game <name> ...``.

Typical workflow for a real game (see docs/games/*.md)::

    hakisa calibrate --game jigoku            # screenshot + region picker, digit templates
    hakisa record    --game jigoku --out data/jigoku           # play! F12 stops, F7 new episode
    hakisa train-bc  --game jigoku --data data/jigoku --out runs/jigoku/bc
    hakisa play      --game jigoku --ckpt runs/jigoku/bc/best.pt   # F8 takeover, F9 pause, F12 kill
    hakisa train-rl  --game jigoku --init runs/jigoku/bc/best.pt --demos data/jigoku --out runs/jigoku/rl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from hakisa.config import available_games, load_game_config
from hakisa.utils import pick_device, seed_everything


def _cfg(args):
    return load_game_config(args.game, overrides=getattr(args, "set", None))


def cmd_games(args) -> None:
    for g in available_games():
        cfg = load_game_config(g)
        print(f"{g:16s} {cfg.title}  [{cfg.model.get('arch')}/{cfg.model.get('preset')}, {cfg.tick_hz:g} Hz]")
        if args.verbose:
            print("   " + cfg.action_space.describe().replace("\n", "\n   "))


def cmd_record(args) -> None:  # pragma: no cover - interactive
    from hakisa.record import record_session

    paths = record_session(_cfg(args), args.out, max_seconds=args.seconds, compress=args.compress)
    print(json.dumps([str(p) for p in paths], indent=2))


def cmd_relabel(args) -> None:
    from hakisa.hippocampus.labeling import relabel_episode
    from hakisa.hippocampus.storage import list_episodes

    cfg = _cfg(args)
    for p in list_episodes(args.data):
        labels = relabel_episode(p, cfg.action_space)
        print(p, {k: int((v >= 0).sum()) for k, v in labels.items()})


def cmd_calibrate(args) -> None:  # pragma: no cover - interactive
    from hakisa.calibrate import calibrate

    calibrate(_cfg(args), screenshot=args.screenshot, out_dir=Path(args.out), digits=args.digits)


def cmd_train_bc(args) -> None:
    from hakisa.hippocampus.dataset import AugmentConfig, BCDataset, split_episodes
    from hakisa.hippocampus.storage import load_episodes
    from hakisa.models import build_policy, load_policy
    from hakisa.study import StudyConfig, train_bc

    cfg = _cfg(args)
    seed_everything(args.seed)
    device = pick_device(args.device)
    eps = load_episodes(args.data)
    train_eps, val_eps = split_episodes(eps, args.val_fraction, seed=args.seed)
    bc = dict(cfg.training.get("bc", {}))
    for k in ("epochs", "batch_size", "lr"):
        if getattr(args, k) is not None:
            bc[k] = getattr(args, k)
    bc["num_workers"] = args.workers
    aug = AugmentConfig(shift=args.shift, flip=bool(bc.get("flip", False)), color=0.1)
    train_ds = BCDataset(train_eps, cfg, augment=aug)
    val_ds = BCDataset(val_eps, cfg) if val_eps else None
    print(f"{len(eps)} episodes: {len(train_ds)} train frames, {len(val_ds) if val_ds else 0} validation frames")
    policy = load_policy(args.init, cfg=cfg) if args.init else build_policy(cfg)
    if device.type == "cpu":
        policy.to_channels_last()
    print(f"{policy.arch}: {policy.num_parameters() / 1e6:.2f} M parameters on {device}")
    train_bc(policy, train_ds, val_ds, StudyConfig.from_dict(bc), args.out, device)


def _realtime_env(cfg, log_fn=print):  # pragma: no cover - needs a game
    from hakisa.envs.realtime import RealtimeGameEnv, Safety
    from hakisa.limbic.probes import HudReader, build_probes
    from hakisa.thalamus.capture import make_dpi_aware, make_source, resolve_capture_region
    from hakisa.thalamus.control import Controller, make_backend
    from hakisa.thalamus.tracker import InputTracker

    make_dpi_aware()
    region = resolve_capture_region(cfg)
    source = make_source(cfg.capture.backend, region)
    controller = Controller(make_backend("auto"))
    tracker = InputTracker().start()
    safety = Safety(cfg.runtime, controller, log_fn).attach(tracker)
    if cfg.name == "lol":
        from hakisa.envs.lol_env import LoLEnv

        env = LoLEnv(cfg, source, controller, region, safety=safety, log_fn=log_fn)
    else:
        base = cfg.source_path.parent if cfg.source_path else None
        probes = build_probes(cfg.probes, base)
        env = RealtimeGameEnv(cfg, source, controller, region, hud=HudReader(probes) if probes else None,
                              safety=safety, cursor_fn=lambda: tracker.snapshot().cursor, log_fn=log_fn)
    return env, tracker, region


def cmd_play(args) -> None:  # pragma: no cover - needs a game
    from hakisa.agent import Agent
    from hakisa.models import load_policy
    from hakisa.record import Recorder

    cfg = _cfg(args)
    device = pick_device(args.device)
    policy = load_policy(args.ckpt, device, cfg=cfg)
    env, tracker, region = _realtime_env(cfg)
    planner = None
    if args.planner:
        from hakisa.prefrontal.runtime import load_planner

        planner = load_planner(args.planner, cfg, device)
    factory = (lambda: Recorder(cfg, args.takeover_dir, region, tracker, source="takeover")) if args.takeover_dir else None
    agent = Agent(policy, env, device, deterministic=args.deterministic, planner=planner, recorder_factory=factory)
    try:
        print(agent.run(max_steps=args.steps))
    finally:
        env.close()
        tracker.stop()
        if planner is not None:
            planner.stop()


def cmd_train_rl(args) -> None:  # pragma: no cover - needs a game
    from hakisa.consolidation import AsyncActorLearner, PPOConfig, PPOTrainer
    from hakisa.hippocampus.dataset import AugmentConfig, BCDataset
    from hakisa.hippocampus.storage import load_episodes
    from hakisa.models import load_policy, save_policy

    cfg = _cfg(args)
    base = cfg.source_path.parent if cfg.source_path else None
    if cfg.name != "lol":
        from hakisa.limbic.probes import build_probes

        build_probes(cfg.probes, base, require_calibrated=True)  # RL needs a working reward
    device = pick_device(args.device)
    policy = load_policy(args.init, device, cfg=cfg)
    ref = load_policy(args.init, device, cfg=cfg)
    demo = BCDataset(load_episodes(args.demos), cfg, augment=AugmentConfig(flip=False)) if args.demos else None
    pcfg = PPOConfig.from_dict(cfg.training.get("rl", {}))
    trainer = PPOTrainer(policy, pcfg, device, ref_policy=ref, demo_ds=demo, out_dir=args.out)
    env, tracker, _ = _realtime_env(cfg)
    pause_key = cfg.runtime.get("game_pause_key")

    def tap_pause():
        if pause_key:
            env._tap((pause_key,))

    learner = AsyncActorLearner(trainer, env, pause_fn=tap_pause, resume_fn=tap_pause)
    try:
        def on_update(stats):
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})
            save_policy(trainer.policy, Path(args.out) / "last.pt", stats=stats)

        learner.run(args.iterations, on_update)
    finally:
        env.close()
        tracker.stop()


def cmd_dojo_demos(args) -> None:
    from hakisa.dojo_tools import record_dojo_demos

    paths = record_dojo_demos(args.out, args.scheme, args.steps, num_envs=args.envs, difficulty=args.difficulty, seed=args.seed)
    print(f"{len(paths)} episodes written to {args.out}")


def cmd_dojo_eval(args) -> None:
    from hakisa.consolidation import evaluate_policy
    from hakisa.dojo_tools import dojo_env_factory
    from hakisa.models import load_policy

    policy = load_policy(args.ckpt)
    res = evaluate_policy(policy, dojo_env_factory(args.scheme, 16, args.difficulty), args.episodes,
                          deterministic=args.deterministic, seed=args.seed)
    print(json.dumps(res, indent=2))


def cmd_dojo_play(args) -> None:  # pragma: no cover - needs a display
    from hakisa.dojo_play import play_dojo

    play_dojo(args.scheme, args.difficulty, record_dir=args.record)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hakisa", description="Hakisa 2.0 — see the screen, press the keys.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def game_arg(p, required=True):
        p.add_argument("--game", required=required, help=f"one of {available_games()} or a YAML path")
        p.add_argument("--set", nargs="*", default=[], help="config overrides like training.bc.epochs=5")

    p = sub.add_parser("games", help="list game configs")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_games)

    p = sub.add_parser("record", help="record your own gameplay")
    game_arg(p)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--compress", action="store_true", help="smaller files, slower training reads")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("relabel", help="recompute labels after changing bindings / lookahead")
    game_arg(p)
    p.add_argument("--data", required=True)
    p.set_defaults(fn=cmd_relabel)

    p = sub.add_parser("calibrate", help="screenshot + interactive region/colour/digit picker")
    game_arg(p)
    p.add_argument("--screenshot", default=None, help="use an existing screenshot instead of capturing")
    p.add_argument("--out", default="calibration")
    p.add_argument("--digits", action="store_true", help="also cut digit templates 0-9 for the score probe")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("train-bc", help="Study: behaviour cloning on recordings")
    game_arg(p)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None, help="continue from a checkpoint")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--shift", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train_bc)

    p = sub.add_parser("play", help="let Hakisa play (F8 takeover, F9 pause, F12 kill)")
    game_arg(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--planner", default=None, help="System 2 checkpoint directory (optional)")
    p.add_argument("--takeover-dir", default=None, help="where to save your corrections")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("train-rl", help="Consolidation: PPO fine-tuning on the real game (async actor-learner)")
    game_arg(p)
    p.add_argument("--init", required=True)
    p.add_argument("--demos", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_train_rl)

    p = sub.add_parser("dojo-demos", help="record the scripted demonstrator in DanmakuDojo")
    p.add_argument("--scheme", choices=["keyboard", "mouse"], default="keyboard")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--difficulty", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_dojo_demos)

    p = sub.add_parser("dojo-eval", help="evaluate a checkpoint in DanmakuDojo")
    p.add_argument("--scheme", choices=["keyboard", "mouse"], default="keyboard")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--difficulty", type=float, default=2.0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=777)
    p.set_defaults(fn=cmd_dojo_eval)

    p = sub.add_parser("dojo-play", help="play DanmakuDojo yourself in a window (needs pygame)")
    p.add_argument("--scheme", choices=["keyboard", "mouse"], default="keyboard")
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--record", default=None, help="record your play into this directory")
    p.set_defaults(fn=cmd_dojo_play)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
