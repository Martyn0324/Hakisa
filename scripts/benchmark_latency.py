"""Measure Hakisa's decision latency (one frame -> actions) for each game network.

    python scripts/benchmark_latency.py --device cpu
    python scripts/benchmark_latency.py --device cuda --half

Prints parameters, MACs-free wall-clock latency (median of N runs, batch 1) and throughput at batch 64.
Only numbers measured on *your* device mean anything; nothing here is estimated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from hakisa.config import load_game_config
from hakisa.models import build_policy


def bench(game: str, device: torch.device, runs: int, half: bool, preset: str | None) -> dict:
    overrides = [f"model.preset={preset}"] if preset else []
    cfg = load_game_config(game, overrides=overrides)
    policy = build_policy(cfg).to(device).eval()
    dtype = torch.float16 if half and device.type == "cuda" else torch.float32
    if dtype == torch.float16:
        policy = policy.half()

    def inputs(b: int):
        obs = torch.randint(0, 255, (b, cfg.obs.channels, cfg.obs.height, cfg.obs.width), dtype=torch.uint8, device=device)
        aux = torch.randn(b, cfg.obs.aux_dim, device=device, dtype=dtype) if cfg.obs.aux_dim else None
        mm = None
        if cfg.obs.minimap is not None:
            mm = torch.randint(0, 255, (b, 3 * cfg.obs.stack, cfg.obs.minimap.height, cfg.obs.minimap.width), dtype=torch.uint8, device=device)
        return obs, aux, mm

    obs, aux, mm = inputs(1)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=dtype == torch.float16):
        for _ in range(10):
            policy.act(obs, aux, mm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            policy.act(obs, aux, mm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        obs64, aux64, mm64 = inputs(64)
        t0 = time.perf_counter()
        for _ in range(5):
            policy.act(obs64, aux64, mm64)
        if device.type == "cuda":
            torch.cuda.synchronize()
        thr = 64 * 5 / (time.perf_counter() - t0)
    return {
        "game": game,
        "arch": policy.arch,
        "preset": cfg.model.get("preset"),
        "input": [cfg.obs.channels, cfg.obs.height, cfg.obs.width],
        "params_M": round(policy.num_parameters() / 1e6, 3),
        "latency_ms_median": round(statistics.median(times), 2),
        "latency_ms_p90": round(sorted(times)[int(0.9 * len(times)) - 1], 2),
        "throughput_fps_b64": round(thr, 1),
        "device": str(device),
        "threads": torch.get_num_threads(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--games", nargs="*", default=["jigoku", "bullet_heaven2", "lol", "dojo_keyboard", "dojo_mouse"])
    args = ap.parse_args()
    device = torch.device(args.device)
    for g in args.games:
        print(json.dumps(bench(g, device, args.runs, args.half, args.preset)))


if __name__ == "__main__":
    main()
