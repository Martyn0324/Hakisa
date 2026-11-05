"""Human Teacher mode — record your own gameplay (frames + every key/mouse input + HUD readings).

Hakisa 1.0's recorder blocked until a key was pressed and stored only that key, so "doing nothing"
was never recorded and every frame needed manual reward labels.  Here the full input state is sampled
at every tick (plus the raw event stream), and labels are derived automatically afterwards
(``hippocampus.labeling``), so a play session becomes a training set with zero manual labelling.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hakisa.actions import tracked_buttons, tracked_keys
from hakisa.config import GameConfig
from hakisa.hippocampus.labeling import relabel_episode
from hakisa.hippocampus.storage import EpisodeWriter
from hakisa.thalamus.capture import sub_regions
from hakisa.types import Region
from hakisa.utils import RateKeeper, crop, resize_rgb


class Recorder:
    def __init__(
        self,
        cfg: GameConfig,
        out_root: str | Path,
        capture_region: Region,
        tracker,
        hud=None,
        source: str = "human",
        live_client=None,
        chunk_size: int = 512,
        compress: bool = False,
        extras_every: int = 10,
    ):
        self.cfg = cfg
        self.tracker = tracker
        self.hud = hud
        self.live_client = live_client
        self.extras_every = extras_every
        self.obs_region, self.minimap_region = sub_regions(cfg, capture_region)
        space = cfg.action_space
        self.keys = tracked_keys(space)
        self.buttons = tracked_buttons(space)
        meta = {
            "game": cfg.name,
            "source": source,
            "tick_hz": cfg.tick_hz,
            "lookahead_ms": cfg.lookahead_ms,
            "capture_region": list(capture_region.as_tuple()),
            "obs_region": list(self.obs_region.as_tuple()),
            "minimap_region": list(self.minimap_region.as_tuple()) if self.minimap_region else None,
            "tracked_keys": self.keys,
            "tracked_buttons": self.buttons,
            "obs_hw": [cfg.obs.height, cfg.obs.width],
        }
        self.writer = EpisodeWriter(out_root, meta, chunk_size=chunk_size, compress=compress)
        self.ticks = 0

    def tick(self, screen: np.ndarray) -> dict[str, Any]:
        snap = self.tracker.snapshot()
        events = self.tracker.drain_events()
        frame = resize_rgb(crop(screen, self.cfg.obs.crop), self.cfg.obs.height, self.cfg.obs.width)
        mm = None
        if self.cfg.obs.minimap is not None:
            m = self.cfg.obs.minimap
            mm = resize_rgb(crop(screen, m.region), m.height, m.width)
        step: dict[str, Any] = {
            "t": float(snap.t),
            "keys": np.array([k in snap.keys for k in self.keys], dtype=bool),
            "buttons": np.array([b in snap.buttons for b in self.buttons], dtype=bool),
            "cursor": np.array(snap.cursor, dtype=np.int32),
            "cursor_norm": np.array(self.obs_region.to_norm(*snap.cursor), dtype=np.float32),
        }
        hud_values: dict[str, Any] = {}
        if self.hud is not None:
            hud_values = self.hud.read(screen)
            for k, v in hud_values.items():
                if isinstance(v, tuple):
                    step[f"hud/{k}_x"], step[f"hud/{k}_y"] = float(v[0]), float(v[1])
                elif v is None:
                    step[f"hud/{k}"] = np.nan
                else:
                    step[f"hud/{k}"] = float(v)
        if self.live_client is not None:
            t, stats, raw = self.live_client.latest()
            step["aux"] = stats.vector() if stats is not None else np.zeros(self.cfg.obs.aux_dim, np.float32)
            if stats is not None and self.ticks % self.extras_every == 0:
                self.writer.add_extra({"t": float(snap.t), "poll_t": t, "stats": stats.values, "team": stats.team, "result": stats.result})
        self.writer.add(frame, step, mm)
        self.writer.add_events(events)
        self.ticks += 1
        return hud_values

    def close(self) -> Path:
        path = self.writer.close()
        if self.ticks:
            relabel_episode(path, self.cfg.action_space)
        return path


def record_session(
    cfg: GameConfig,
    out_root: str | Path,
    max_seconds: float | None = None,
    countdown: float = 3.0,
    compress: bool = False,
    log_fn: Callable[[str], None] = print,
) -> list[Path]:  # pragma: no cover - needs a display, OS hooks and a game
    """Interactive recording loop: F7 = start a new episode, F12 = stop. Splits on detected game over."""
    from hakisa.envs.realtime import Safety
    from hakisa.limbic.probes import HudReader, build_probes
    from hakisa.thalamus.capture import make_dpi_aware, make_source, resolve_capture_region
    from hakisa.thalamus.tracker import InputTracker

    make_dpi_aware()
    region = resolve_capture_region(cfg)
    source = make_source(cfg.capture.backend, region)
    tracker = InputTracker().start()
    safety = Safety(cfg.runtime, None, log_fn).attach(tracker)
    base = cfg.source_path.parent if cfg.source_path else None
    probes = build_probes(cfg.probes, base)
    hud = HudReader(probes) if probes else None
    live = None
    if cfg.name == "lol":
        from hakisa.limbic.lol_client import LiveClientPoller

        live = LiveClientPoller(cfg.reward.get("live_client_url", "https://127.0.0.1:2999/liveclientdata/allgamedata"),
                                interval=1.0 / float(cfg.record.get("live_client_hz", 2)))
        live.start()
    log_fn(f"[record] capturing {region} at {cfg.tick_hz} Hz; {safety.keys['kill']} stops, "
           f"{safety.keys['start']} starts a new episode. Starting in {countdown:.0f}s...")
    time.sleep(countdown)
    paths: list[Path] = []
    rec = Recorder(cfg, out_root, region, tracker, hud, live_client=live, compress=compress)
    rate = RateKeeper(cfg.tick_hz)
    t0 = time.perf_counter()
    try:
        while not safety.kill.is_set():
            if max_seconds and time.perf_counter() - t0 > max_seconds:
                break
            hud_values = rec.tick(source.grab())
            if safety.started.is_set() or hud_values.get("game_over"):
                safety.started.clear()
                paths.append(rec.close())
                log_fn(f"[record] saved {paths[-1]} ({rec.ticks} ticks)")
                rec = Recorder(cfg, out_root, region, tracker, hud, live_client=live, compress=compress)
            late = rate.wait()
            if late > 0.5 / cfg.tick_hz:
                log_fn(f"[record] warning: tick {late * 1000:.0f} ms late (lower tick_hz or obs size)")
    finally:
        if rec.ticks:
            paths.append(rec.close())
            log_fn(f"[record] saved {paths[-1]} ({rec.ticks} ticks)")
        tracker.stop()
        source.close()
        if live is not None:
            live.stop()
    return paths
