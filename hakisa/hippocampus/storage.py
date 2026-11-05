"""Episode storage ("memories").

Layout of one episode directory::

    meta.json              game, tick rate, regions, tracked keys, source (human/expert/agent/takeover)
    frames_00000.npy       uint8 (n, H, W, 3)   -- .npz when compress=True
    minimap_00000.npy      optional second stream (LoL)
    steps_00000.npz        per-tick arrays for the same chunk (t, cursor, keys, buttons, hud/*, label/*, ...)
    events.jsonl           raw input events (for re-labelling with different bindings / lookahead)
    extras.jsonl           anything else with a timestamp (e.g. LoL live-client snapshots)
    labels.npz             labels computed after recording (``hakisa relabel`` recomputes them)

Chunks are written as soon as they are full, so a crash loses at most one chunk.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hakisa.types import InputEvent


def _event_to_dict(e: InputEvent) -> dict[str, Any]:
    return {"t": e.t, "kind": e.kind, "name": e.name, "pressed": e.pressed, "x": e.x, "y": e.y}


def _event_from_dict(d: dict[str, Any]) -> InputEvent:
    return InputEvent(float(d["t"]), d["kind"], d["name"], bool(d["pressed"]), d.get("x"), d.get("y"))


class EpisodeWriter:
    def __init__(
        self,
        root: str | Path,
        meta: dict[str, Any],
        chunk_size: int = 512,
        compress: bool = False,
        background: bool = True,
        name: str | None = None,
    ):
        root = Path(root)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = root / (name or f"{stamp}_{meta.get('game', 'game')}_{uuid.uuid4().hex[:6]}")
        self.dir.mkdir(parents=True, exist_ok=False)
        self.meta = dict(meta)
        self.meta.setdefault("created", stamp)
        self.meta["complete"] = False
        self.meta["chunk_size"] = chunk_size
        self.meta["compress"] = compress
        self.chunk_size = chunk_size
        self.compress = compress
        self._frames: list[np.ndarray] = []
        self._minimap: list[np.ndarray] = []
        self._steps: dict[str, list[Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._chunk = 0
        self.length = 0
        self._write_meta()
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        if background:
            self._q = queue.Queue(maxsize=8)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # -- public -------------------------------------------------------------------------------
    def add(self, frame: np.ndarray, step: dict[str, Any] | None = None, minimap: np.ndarray | None = None) -> None:
        if self._error is not None:
            raise RuntimeError("episode writer failed") from self._error
        self._frames.append(np.asarray(frame, dtype=np.uint8))
        if minimap is not None:
            self._minimap.append(np.asarray(minimap, dtype=np.uint8))
        step = step or {}
        n_before = len(self._frames) - 1
        for k, v in step.items():
            lst = self._steps.setdefault(k, [])
            if len(lst) < n_before:  # key appeared late: pad with zeros of the same shape
                pad = np.zeros_like(np.asarray(v))
                lst.extend([pad] * (n_before - len(lst)))
            lst.append(v)
        for k, lst in self._steps.items():
            if k not in step and len(lst) < len(self._frames):
                lst.append(np.zeros_like(np.asarray(lst[-1])) if lst else 0)
        self.length += 1
        if len(self._frames) >= self.chunk_size:
            self._flush()

    def add_events(self, events: Iterable[InputEvent]) -> None:
        self._events.extend(_event_to_dict(e) for e in events)

    def add_extra(self, record: dict[str, Any]) -> None:
        with open(self.dir / "extras.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def set_labels(self, labels: dict[str, np.ndarray]) -> None:
        np.savez(self.dir / "labels.npz", **{k: np.asarray(v) for k, v in labels.items()})

    def close(self) -> Path:
        if self._frames:
            self._flush()
        if self._q is not None and self._thread is not None:
            self._q.put(None)
            self._thread.join()
        if self._error is not None:
            raise RuntimeError("episode writer failed") from self._error
        self.meta["complete"] = True
        self.meta["length"] = self.length
        self.meta["chunks"] = self._chunk
        self._write_meta()
        return self.dir

    # -- internals ----------------------------------------------------------------------------
    def _write_meta(self) -> None:
        with open(self.dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2, default=str)

    def _flush(self) -> None:
        job = (
            self._chunk,
            np.stack(self._frames),
            np.stack(self._minimap) if self._minimap else None,
            {k: np.asarray(v) for k, v in self._steps.items()},
            self._events,
        )
        self._frames, self._minimap, self._steps, self._events = [], [], {}, []
        self._chunk += 1
        if self._q is not None:
            self._q.put(job)
        else:
            self._save(*job)

    def _worker(self) -> None:
        assert self._q is not None
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                self._save(*job)
            except BaseException as exc:  # surfaced on the next add()/close()
                self._error = exc

    def _save(self, idx: int, frames: np.ndarray, minimap, steps, events) -> None:
        tag = f"{idx:05d}"
        if self.compress:
            np.savez_compressed(self.dir / f"frames_{tag}.npz", frames=frames)
            if minimap is not None:
                np.savez_compressed(self.dir / f"minimap_{tag}.npz", frames=minimap)
        else:
            np.save(self.dir / f"frames_{tag}.npy", frames)
            if minimap is not None:
                np.save(self.dir / f"minimap_{tag}.npy", minimap)
        np.savez(self.dir / f"steps_{tag}.npz", **steps)
        if events:
            with open(self.dir / "events.jsonl", "a", encoding="utf-8") as fh:
                for e in events:
                    fh.write(json.dumps(e) + "\n")


class Episode:
    """Read-only view of a recorded episode (memory-mapped frames, cached compressed chunks)."""

    def __init__(self, path: str | Path, cache_chunks: int = 8):
        self.path = Path(path)
        with open(self.path / "meta.json", "r", encoding="utf-8") as fh:
            self.meta: dict[str, Any] = json.load(fh)
        self._frame_files = sorted(self.path.glob("frames_*.np[yz]"))
        self._minimap_files = sorted(self.path.glob("minimap_*.np[yz]"))
        step_files = sorted(self.path.glob("steps_*.npz"))
        n = min(len(self._frame_files), len(step_files))  # an interrupted recording may miss the last steps
        self._frame_files = self._frame_files[:n]
        self._minimap_files = self._minimap_files[:n]
        self._chunk_lens: list[int] = []
        arrays: dict[str, list[np.ndarray]] = {}
        for f in step_files[:n]:
            with np.load(f) as z:
                lens = {len(z[k]) for k in z.files}
                length = lens.pop() if len(lens) == 1 else min(lens)
                self._chunk_lens.append(length)
                for k in z.files:
                    arrays.setdefault(k, []).append(z[k][:length])
        self._arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        self._offsets = np.concatenate([[0], np.cumsum(self._chunk_lens)]).astype(np.int64)
        self.length = int(self._offsets[-1])
        self._labels: dict[str, np.ndarray] | None = None
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._cache_chunks = cache_chunks

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, Any]:  # DataLoader workers: never pickle mapped chunks
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    # -- arrays ---------------------------------------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self._arrays

    def array(self, key: str) -> np.ndarray:
        return self._arrays[key]

    def keys(self) -> list[str]:
        return sorted(self._arrays)

    def labels(self) -> dict[str, np.ndarray]:
        if self._labels is None:
            out: dict[str, np.ndarray] = {}
            lab = self.path / "labels.npz"
            if lab.exists():
                with np.load(lab) as z:
                    out = {k: z[k][: self.length] for k in z.files}
            for k, v in self._arrays.items():
                if k.startswith("label/") and k[6:] not in out:
                    out[k[6:]] = v
            self._labels = out
        return self._labels

    # -- frames ---------------------------------------------------------------------------------
    def _chunk(self, kind: str, idx: int) -> np.ndarray:
        key = (kind, idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        files = self._frame_files if kind == "frames" else self._minimap_files
        f = files[idx]
        if f.suffix == ".npy":
            arr = np.load(f, mmap_mode="r")
        else:
            with np.load(f) as z:
                arr = z["frames"]
        self._cache[key] = arr
        if len(self._cache) > self._cache_chunks:
            self._cache.popitem(last=False)
        return arr

    def _gather(self, kind: str, idx: np.ndarray | int) -> np.ndarray:
        scalar = np.isscalar(idx)
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        if (idx < 0).any() or (idx >= self.length).any():
            raise IndexError(f"frame index out of range 0..{self.length - 1}")
        chunk_ids = np.searchsorted(self._offsets, idx, side="right") - 1
        out = None
        for c in np.unique(chunk_ids):
            sel = chunk_ids == c
            data = self._chunk(kind, int(c))[idx[sel] - self._offsets[c]]
            if out is None:
                out = np.empty((len(idx),) + data.shape[1:], dtype=np.uint8)
            out[sel] = data
        assert out is not None
        return out[0] if scalar else out

    def frames(self, idx: np.ndarray | int) -> np.ndarray:
        return self._gather("frames", idx)

    def minimap(self, idx: np.ndarray | int) -> np.ndarray:
        if not self._minimap_files:
            raise KeyError("episode has no minimap stream")
        return self._gather("minimap", idx)

    @property
    def has_minimap(self) -> bool:
        return bool(self._minimap_files)

    # -- raw input ------------------------------------------------------------------------------
    def events(self) -> list[InputEvent]:
        p = self.path / "events.jsonl"
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as fh:
            return [_event_from_dict(json.loads(line)) for line in fh if line.strip()]

    def extras(self) -> list[dict[str, Any]]:
        p = self.path / "extras.jsonl"
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def list_episodes(root: str | Path) -> list[Path]:
    root = Path(root)
    if (root / "meta.json").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/meta.json"))


def load_episodes(roots: Iterable[str | Path]) -> list[Episode]:
    eps: list[Episode] = []
    for r in roots:
        for p in list_episodes(r):
            ep = Episode(p)
            if ep.length > 0:
                eps.append(ep)
    return eps
