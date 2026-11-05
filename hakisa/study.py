"""Study mode — supervised learning (behaviour cloning) on human gameplay.

"She learns from us" (Hakisa 1.0).  Each frame is *classified* into the action the human took:
one cross-entropy per head (class-weighted so rare bombs matter, label-smoothed), the pointer's exact
X-then-Y|X likelihood, and the amygdala's danger loss.  AdamW + warm-up + cosine decay, an EMA copy
of the weights for evaluation, fp16 autocast on CUDA.

Liu Ruo-Ze et al. (HierNet-SC2) reported that more than ~10 SL epochs overfit; Hakisa therefore
keeps the best checkpoint by validation NLL (split by *episode*, never by frame).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from hakisa.hippocampus.dataset import BCDataset, compute_class_weights, keyframe_weights, worker_seed_init
from hakisa.models.games import save_policy
from hakisa.models.policy import HakisaPolicy
from hakisa.utils import JsonlLogger


@dataclass
class StudyConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    label_smoothing: float = 0.05
    class_weight_power: float = 0.0  # >0 boosts rare classes but miscalibrates sampling (bombs!)
    danger_weight: float = 0.5
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    num_workers: int = 0
    amp: bool = True
    intent_dropout: float = 0.3
    keyframe_boost: float = 1.0  # >1 oversamples ticks where the action changes (anti-copycat)
    log_every: int = 50
    max_steps: int | None = None
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StudyConfig":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def batch_to(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".pos") or "emb" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def bc_loss(
    policy: HakisaPolicy,
    batch: dict,
    class_weights: dict[str, torch.Tensor] | None,
    label_smoothing: float,
    danger_weight: float,
    intent_dropout: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    intent = batch.get("intent")
    if intent is not None and intent_dropout > 0:
        keep = (torch.rand(intent.shape[0], 1, device=intent.device) >= intent_dropout).long()
        intent = intent * keep
    labels = batch["labels"]
    out = policy(batch["obs"], batch.get("aux"), batch.get("minimap"), intent, condition=labels)
    losses = policy.distribution(out).bc_losses(labels, class_weights, label_smoothing)
    total = sum(policy.space.head(k).weight * v for k, v in losses.items())
    logs = {f"loss/{k}": float(v.detach()) for k, v in losses.items()}
    if out.danger is not None and danger_weight > 0:
        d = F.binary_cross_entropy_with_logits(out.danger.float(), batch["danger"].float())
        total = total + danger_weight * d
        logs["loss/danger"] = float(d.detach())
    logs["loss/total"] = float(total.detach())
    return total, logs


@torch.no_grad()
def evaluate_bc(policy: HakisaPolicy, ds: BCDataset, device: torch.device, batch_size: int = 128, max_batches: int | None = None) -> dict[str, float]:
    policy.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    space = policy.space
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}

    def add(key: str, value: float, n: float) -> None:
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0.0) + n

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = batch_to(batch, device)
        labels = batch["labels"]
        out = policy(batch["obs"], batch.get("aux"), batch.get("minimap"), batch.get("intent"), condition=labels)
        for h in space.categorical_heads:
            y = labels[h.name]
            valid = y >= 0
            logits = out.logits[h.name]
            nll = F.cross_entropy(logits, y.clamp_min(0), reduction="none")
            add(f"nll/{h.name}", float((nll * valid).sum()), float(valid.sum()))
            pred = logits.argmax(-1)
            add(f"acc/{h.name}", float(((pred == y) & valid).sum()), float(valid.sum()))
            if h.num_classes == 2:  # F1 of the "on" class (rare events like bombs)
                tp = float(((pred == 1) & (y == 1)).sum())
                add(f"_tp/{h.name}", tp, 0)
                add(f"_fp/{h.name}", float(((pred == 1) & (y == 0)).sum()), 0)
                add(f"_fn/{h.name}", float(((pred == 0) & (y == 1)).sum()), 0)
        for p in space.pointer_heads:
            fx, fy = labels[f"{p.name}_x"], labels[f"{p.name}_y"]
            valid = (fx >= 0) & (fy >= 0)
            if not valid.any():
                continue
            lp = out.pointer[p.name].log_prob(fx, fy)
            add(f"nll/{p.name}", float((-lp * valid).sum()), float(valid.sum()))
            px, py = out.pointer[p.name].sample(deterministic=True)
            err = torch.sqrt((px - fx).float() ** 2 + (py - fy).float() ** 2)
            add(f"err/{p.name}", float((err * valid).sum()), float(valid.sum()))
            add(f"acc2/{p.name}", float(((err <= 2) & valid).sum()), float(valid.sum()))
        if out.danger is not None:
            pred = (out.danger > 0).float()
            add("acc/danger", float((pred == batch["danger"]).sum()), float(len(pred)))
    metrics = {k: sums[k] / max(counts[k], 1.0) for k in sums if not k.startswith("_")}
    for h in space.categorical_heads:
        if h.num_classes == 2 and f"_tp/{h.name}" in sums:
            tp, fp, fn = sums[f"_tp/{h.name}"], sums[f"_fp/{h.name}"], sums[f"_fn/{h.name}"]
            metrics[f"f1/{h.name}"] = 2 * tp / max(2 * tp + fp + fn, 1.0)
    metrics["nll/total"] = sum(v for k, v in metrics.items() if k.startswith("nll/"))
    return metrics


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - self.decay)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)


def train_bc(
    policy: HakisaPolicy,
    train_ds: BCDataset,
    val_ds: BCDataset | None,
    cfg: StudyConfig,
    out_dir: str | Path,
    device: torch.device,
    log_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "study_log.jsonl")
    torch.manual_seed(cfg.seed)
    policy.to(device).train()
    sampler = None
    if cfg.keyframe_boost > 1.0:
        weights = keyframe_weights(train_ds, cfg.keyframe_boost)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(train_ds), replacement=True)
        log_fn(f"[study] keyframes (action changes): {(weights > 1).mean():.1%} of ticks, boosted x{cfg.keyframe_boost:g}")
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=len(train_ds) > cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_seed_init,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    class_weights = compute_class_weights(train_ds, cfg.class_weight_power) if cfg.class_weight_power > 0 else None
    opt = torch.optim.AdamW(param_groups(policy, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.99))
    steps_per_epoch = max(len(loader), 1)
    total = cfg.max_steps or cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_frac * total))

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg.lr * (step + 1) / warmup
        prog = (step - warmup) / max(total - warmup, 1)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(policy, cfg.ema_decay)
    best = float("inf")
    history: dict[str, Any] = {"config": asdict(cfg), "epochs": []}
    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        policy.train()
        running: dict[str, float] = {}
        for batch in loader:
            batch = batch_to(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, logs = bc_loss(policy, batch, class_weights, cfg.label_smoothing, cfg.danger_weight, cfg.intent_dropout)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(policy)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) * 0.98 + v * 0.02 if k in running else v
            step += 1
            if step % cfg.log_every == 0:
                rec = {"step": step, "epoch": epoch, "lr": lr_at(step), "elapsed_s": time.time() - t0, **running}
                logger.log(rec)
                log_fn(f"[study] step {step}/{total} epoch {epoch} loss {running.get('loss/total', 0):.4f}")
            if cfg.max_steps and step >= cfg.max_steps:
                break
        metrics = evaluate_bc(ema.module, val_ds, device) if val_ds is not None and len(val_ds) else {}
        rec = {"epoch": epoch, "step": step, "elapsed_s": time.time() - t0, **{f"val/{k}": v for k, v in metrics.items()}}
        history["epochs"].append(rec)
        logger.log(rec)
        score = metrics.get("nll/total", -epoch)  # without validation data keep the latest
        save_policy(ema.module, out_dir / "last.pt", epoch=epoch, metrics=metrics)
        if score <= best:
            best = score
            save_policy(ema.module, out_dir / "best.pt", epoch=epoch, metrics=metrics)
        log_fn(f"[study] epoch {epoch} done: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items()) if k.startswith(("acc", "nll/total", "f1", "err"))))
        if cfg.max_steps and step >= cfg.max_steps:
            break
    history["best_val_nll"] = best
    history["ema"] = ema.module
    return history
