"""Turning recorded human input into labels (no manual labelling).

``compute_labels`` rebuilds the full input timeline of an episode (per-tick snapshots + raw events)
and asks the :class:`~hakisa.actions.ActionSpace` what the human did ``lookahead`` ms after each frame.
Because raw events are stored, labels can be recomputed later with different bindings or lookahead
(``hakisa relabel``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hakisa.actions import ActionSpace, InputTimeline
from hakisa.hippocampus.storage import Episode
from hakisa.types import InputSnapshot, Region


def episode_timeline(ep: Episode) -> InputTimeline:
    keys: list[str] = list(ep.meta.get("tracked_keys", []))
    buttons: list[str] = list(ep.meta.get("tracked_buttons", []))
    t = ep.array("t")
    kmask = ep.array("keys") if ep.has("keys") else np.zeros((len(t), 0), dtype=bool)
    bmask = ep.array("buttons") if ep.has("buttons") else np.zeros((len(t), 0), dtype=bool)
    cursor = ep.array("cursor") if ep.has("cursor") else np.zeros((len(t), 2), dtype=np.int64)
    snaps = []
    for i in range(len(t)):
        snaps.append(
            InputSnapshot(
                float(t[i]),
                frozenset(k for j, k in enumerate(keys) if kmask[i, j]),
                frozenset(b for j, b in enumerate(buttons) if bmask[i, j]),
                (int(cursor[i, 0]), int(cursor[i, 1])),
            )
        )
    return InputTimeline(snaps, ep.events())


def compute_labels(ep: Episode, space: ActionSpace) -> dict[str, np.ndarray]:
    tl = episode_timeline(ep)
    region = Region(*ep.meta["obs_region"])
    minimap = Region(*ep.meta["minimap_region"]) if ep.meta.get("minimap_region") else None
    t = ep.array("t")
    out = {k: np.empty(len(t), dtype=np.int64) for k in space.label_keys()}
    for i, ti in enumerate(t):
        lab = space.labels_from_timeline(tl, float(ti), region, minimap)
        for k, v in lab.items():
            out[k][i] = v
    return out


def relabel_episode(path: str | Path, space: ActionSpace) -> dict[str, np.ndarray]:
    ep = Episode(path)
    labels = compute_labels(ep, space)
    np.savez(Path(path) / "labels.npz", **labels)
    return labels


def life_lost_array(ep: Episode) -> np.ndarray:
    """Boolean per tick: a life was lost at this tick (from ``life_lost`` or a drop in ``hud/lives``)."""
    if ep.has("life_lost"):
        return ep.array("life_lost").astype(bool)
    if ep.has("hud/lives"):
        lives = ep.array("hud/lives").astype(np.float64)
        lost = np.zeros(len(lives), dtype=bool)
        lost[1:] = lives[1:] < lives[:-1]
        return lost
    return np.zeros(ep.length, dtype=bool)


def danger_array(ep: Episode, horizon: int) -> np.ndarray:
    """1.0 if a life is lost within the next ``horizon`` ticks (the amygdala's label)."""
    lost = life_lost_array(ep).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(lost)])
    n = len(lost)
    idx = np.arange(n)
    hi = np.minimum(idx + horizon + 1, n)
    return ((csum[hi] - csum[idx + 1]) > 0).astype(np.float32)
