"""One architecture per game (the owner asked for a specific net for each).

JigokuNet (keyboard bullet hell, vertical playfield)
    encoder -> **foveal attention pooling** (two learned glimpses at 1/8 resolution) + global vector
    -> move (9-way), focus, shoot, bomb + danger head. Keyboard control has no explicit target, so
    the decisive local evidence (bullets around the ship) is gathered by learned saliency.

BulletHeavenNet (mouse bullet hell)
    extra input channel with the current cursor position, encoder -> **superior-colliculus pointer
    decoder** (U-Net-lite back to 1/4 resolution, X then Y|X sampling) + fire/bomb/slow heads.

LoLNet (League of Legends)
    main-view encoder + **minimap encoder** + live-client **game-state MLP** -> action type
    (noop, move, attack-move, QWER, DF, items 1-9, recall, level-ups, ...) -> pointer heads
    **conditioned on the chosen action type** (main view and minimap), AlphaStar-style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hakisa.config import GameConfig, config_from_dict
from hakisa.models.policy import HakisaPolicy


class JigokuNet(HakisaPolicy):
    arch = "jigoku"
    use_fovea = True


class BulletHeavenNet(HakisaPolicy):
    arch = "bullet_heaven"
    use_fovea = False

    def __init__(self, cfg: GameConfig, hidden: int | None = None):
        if not cfg.action_space.pointer_heads:
            raise ValueError("BulletHeavenNet needs a pointer head in the action space")
        super().__init__(cfg, hidden)


class LoLNet(HakisaPolicy):
    arch = "lol"
    use_fovea = False

    def __init__(self, cfg: GameConfig, hidden: int | None = None):
        if cfg.obs.minimap is None:
            raise ValueError("LoLNet needs obs.minimap")
        super().__init__(cfg, hidden)


ARCHS: dict[str, type[HakisaPolicy]] = {
    "generic": HakisaPolicy,
    "jigoku": JigokuNet,
    "bullet_heaven": BulletHeavenNet,
    "lol": LoLNet,
}


def build_policy(cfg: GameConfig) -> HakisaPolicy:
    arch = cfg.model.get("arch", "generic")
    if arch not in ARCHS:
        raise ValueError(f"unknown model.arch {arch!r} (choose from {list(ARCHS)})")
    return ARCHS[arch](cfg)


def save_policy(policy: HakisaPolicy, path: str | Path, **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": policy.arch,
            "config": policy.cfg.raw,
            "state_dict": policy.state_dict(),
            "extra": extra,
        },
        path,
    )


def load_policy(path: str | Path, device: str | torch.device = "cpu", cfg: GameConfig | None = None) -> HakisaPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg or config_from_dict(ckpt["config"])
    policy = ARCHS[ckpt["arch"]](cfg)
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(device)
