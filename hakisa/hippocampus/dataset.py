"""Datasets for behaviour cloning ("study").

Each item is the model input at tick ``t`` (last ``stack`` frames, oldest first, never crossing an
episode boundary), the labels at ``t``, the amygdala's danger label and optional aux/intent vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from hakisa.config import GameConfig
from hakisa.hippocampus.labeling import danger_array
from hakisa.hippocampus.storage import Episode
from hakisa.types import Region
from hakisa.utils import resize_rgb


def cursor_heatmap(height: int, width: int, u: float, v: float, sigma: float = 1.5) -> np.ndarray:
    """uint8 (H, W) Gaussian blob at normalised position (u, v) — the extra 'where is my mouse' channel."""
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    sy, sx = sigma / height, sigma / width
    g = np.exp(-0.5 * (((ys[:, None] - v) / sy) ** 2 + ((xs[None, :] - u) / sx) ** 2))
    return (g * 255.0).astype(np.uint8)


@dataclass
class AugmentConfig:
    shift: int = 4  # max translation in pixels (pointer labels are shifted consistently)
    flip: bool = False  # horizontal mirror with left/right label swap
    color: float = 0.1  # brightness jitter amplitude


def _episode_cursor_norm(ep: Episode) -> np.ndarray:
    if ep.has("cursor_norm"):
        return ep.array("cursor_norm").astype(np.float32)
    if ep.has("cursor") and ep.meta.get("obs_region"):
        reg = Region(*ep.meta["obs_region"])
        c = ep.array("cursor").astype(np.float32)
        u = np.clip((c[:, 0] - reg.left) / max(reg.width, 1), 0, 1)
        v = np.clip((c[:, 1] - reg.top) / max(reg.height, 1), 0, 1)
        return np.stack([u, v], axis=1)
    return np.full((ep.length, 2), 0.5, dtype=np.float32)


class BCDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[Episode],
        cfg: GameConfig,
        augment: AugmentConfig | None = None,
        danger_horizon: int | None = None,
        intents: Sequence[np.ndarray] | None = None,
        intent_flip: Callable[[np.ndarray], np.ndarray] | None = None,
        stride: int = 1,
    ):
        self.cfg = cfg
        self.space = cfg.action_space
        self.episodes = list(episodes)
        self.stack = cfg.obs.stack
        self.h, self.w = cfg.obs.height, cfg.obs.width
        self.augment = augment
        self.intents = list(intents) if intents is not None else None
        self.intent_flip = intent_flip
        horizon = int(danger_horizon or cfg.model.get("danger_horizon", 20))
        self.labels = [ep.labels() for ep in self.episodes]
        self.danger = [danger_array(ep, horizon) for ep in self.episodes]
        self.cursor = [_episode_cursor_norm(ep) for ep in self.episodes] if cfg.obs.cursor_channel else None
        self.aux = None
        if cfg.obs.aux_dim > 0:
            self.aux = [
                ep.array("aux").astype(np.float32) if ep.has("aux") else np.zeros((ep.length, cfg.obs.aux_dim), np.float32)
                for ep in self.episodes
            ]
        self.flip_perms = {h.name: np.asarray(self.space.flip_permutation(h)) for h in self.space.categorical_heads}
        cat_keys = [h.name for h in self.space.categorical_heads]
        index: list[tuple[int, int]] = []
        for e, (ep, lab) in enumerate(zip(self.episodes, self.labels)):
            missing = [k for k in self.space.label_keys() if k not in lab]
            if missing:
                raise KeyError(f"episode {ep.path} has no labels for {missing}; run `hakisa relabel`")
            valid = np.ones(ep.length, dtype=bool)
            for k in cat_keys:
                valid &= lab[k] >= 0
            for t in np.nonzero(valid)[0][::stride]:
                index.append((e, int(t)))
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    # -- assembly -------------------------------------------------------------------------------
    def _frames(self, e: int, t: int) -> np.ndarray:
        ep = self.episodes[e]
        idx = np.array([max(t - k, 0) for k in reversed(range(self.stack))], dtype=np.int64)
        frames = ep.frames(idx)  # (stack, H, W, 3)
        if frames.shape[1] != self.h or frames.shape[2] != self.w:
            frames = np.stack([resize_rgb(f, self.h, self.w) for f in frames])
        return frames.transpose(0, 3, 1, 2).reshape(self.stack * 3, self.h, self.w)

    def _minimap(self, e: int, t: int) -> np.ndarray:
        ep = self.episodes[e]
        mm = self.cfg.obs.minimap
        assert mm is not None
        idx = np.array([max(t - k, 0) for k in reversed(range(self.stack))], dtype=np.int64)
        frames = ep.minimap(idx)
        if frames.shape[1] != mm.height or frames.shape[2] != mm.width:
            frames = np.stack([resize_rgb(f, mm.height, mm.width) for f in frames])
        return frames.transpose(0, 3, 1, 2).reshape(self.stack * 3, mm.height, mm.width)

    def item_numpy(self, i: int, rng: np.random.Generator | None = None) -> dict:
        e, t = self.index[i]
        obs = self._frames(e, t)
        if self.cursor is not None:
            u, v = self.cursor[e][t]
            obs = np.concatenate([obs, cursor_heatmap(self.h, self.w, float(u), float(v))[None]], axis=0)
        labels = {k: int(v[t]) for k, v in self.labels[e].items() if k in self.space.label_keys()}
        out = {
            "obs": obs,
            "labels": labels,
            "danger": np.float32(self.danger[e][t]),
        }
        if self.aux is not None:
            out["aux"] = self.aux[e][t]
        if self.cfg.obs.minimap is not None and self.episodes[e].has_minimap:
            out["minimap"] = self._minimap(e, t)
        if self.intents is not None:
            out["intent"] = np.asarray(self.intents[e][t], dtype=np.int64)
        if self.augment is not None:
            out = self._augment(out, rng or np.random.default_rng())
        return out

    def _augment(self, item: dict, rng: np.random.Generator) -> dict:
        aug = self.augment
        assert aug is not None
        obs = item["obs"]
        labels = dict(item["labels"])
        if aug.flip and rng.random() < 0.5:
            obs = obs[:, :, ::-1]
            for h in self.space.categorical_heads:
                labels[h.name] = int(self.flip_perms[h.name][labels[h.name]])
            for p in self.space.pointer_heads:
                if p.source == "obs" and labels[f"{p.name}_x"] >= 0:
                    labels[f"{p.name}_x"] = p.fine_hw[1] - 1 - labels[f"{p.name}_x"]
            if "intent" in item and self.intent_flip is not None:
                item["intent"] = self.intent_flip(item["intent"])
        if aug.shift > 0:
            s = aug.shift
            dx, dy = (int(v) for v in rng.integers(-s, s + 1, size=2))
            padded = np.pad(obs, ((0, 0), (s, s), (s, s)))
            obs = padded[:, s + dy : s + dy + self.h, s + dx : s + dx + self.w]
            for p in self.space.pointer_heads:
                if p.source != "obs" or labels[f"{p.name}_x"] < 0:
                    continue
                fh, fw = p.fine_hw
                labels[f"{p.name}_x"] = int(np.clip(labels[f"{p.name}_x"] - round(dx * fw / self.w), 0, fw - 1))
                labels[f"{p.name}_y"] = int(np.clip(labels[f"{p.name}_y"] - round(dy * fh / self.h), 0, fh - 1))
        if aug.color > 0:
            rgb = 3 * self.stack
            gain = 1.0 + float(rng.uniform(-aug.color, aug.color))
            bias = float(rng.uniform(-aug.color, aug.color)) * 64.0
            obs = obs.astype(np.float32)
            obs[:rgb] = obs[:rgb] * gain + bias
            obs = np.clip(obs, 0, 255).astype(np.uint8)
        item["obs"] = np.ascontiguousarray(obs)
        item["labels"] = labels
        return item

    def __getitem__(self, i: int) -> dict:
        item = self.item_numpy(i)
        out = {
            "obs": torch.from_numpy(np.ascontiguousarray(item["obs"])),
            "labels": {k: torch.tensor(v, dtype=torch.long) for k, v in item["labels"].items()},
            "danger": torch.tensor(item["danger"], dtype=torch.float32),
        }
        if "aux" in item:
            out["aux"] = torch.from_numpy(np.asarray(item["aux"], dtype=np.float32))
        if "minimap" in item:
            out["minimap"] = torch.from_numpy(np.ascontiguousarray(item["minimap"]))
        if "intent" in item:
            out["intent"] = torch.from_numpy(np.asarray(item["intent"], dtype=np.int64))
        return out

    # -- statistics -----------------------------------------------------------------------------
    def label_counts(self, key: str, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for e, t in self.index:
            v = self.labels[e][key][t]
            if v >= 0:
                counts[v] += 1
        return counts


def compute_class_weights(
    dataset: BCDataset, power: float = 0.5, max_weight: float = 10.0
) -> dict[str, torch.Tensor]:
    """Inverse-frequency^power class weights per categorical head (rare bombs get > 1)."""
    out: dict[str, torch.Tensor] = {}
    for h in dataset.space.categorical_heads:
        counts = dataset.label_counts(h.name, h.num_classes).astype(np.float64)
        freq = counts / max(counts.sum(), 1.0)
        w = np.where(counts > 0, (freq + 1e-12) ** (-power), 0.0)
        mean = (w * freq).sum()
        w = w / mean if mean > 0 else np.ones_like(w)
        out[h.name] = torch.tensor(np.clip(w, 0.0, max_weight), dtype=torch.float32)
    return out


class ChunkShuffleSampler(Sampler[int]):
    """Shuffles chunk order, then samples within groups of ``group`` chunks (cheap for compressed storage)."""

    def __init__(self, dataset: BCDataset, group: int = 4, seed: int = 0):
        self.dataset = dataset
        self.group = group
        self.seed = seed
        self.epoch = 0
        by_chunk: dict[tuple[int, int], list[int]] = {}
        for i, (e, t) in enumerate(dataset.index):
            cs = int(dataset.episodes[e].meta.get("chunk_size", 512))
            by_chunk.setdefault((e, t // cs), []).append(i)
        self.chunks = list(by_chunk.values())

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.chunks))
        for g in range(0, len(order), self.group):
            pool = np.concatenate([self.chunks[c] for c in order[g : g + self.group]])
            yield from rng.permutation(pool).tolist()

    def __len__(self) -> int:
        return len(self.dataset)


def worker_seed_init(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


def split_episodes(episodes: Sequence[Episode], val_fraction: float, seed: int = 0) -> tuple[list[Episode], list[Episode]]:
    """Split by episode (never by frame: neighbouring frames are near-duplicates)."""
    eps = list(episodes)
    if len(eps) < 2 or val_fraction <= 0:
        return eps, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eps))
    n_val = max(1, int(math.ceil(len(eps) * val_fraction)))
    val = [eps[i] for i in order[:n_val]]
    train = [eps[i] for i in order[n_val:]]
    return train, val
