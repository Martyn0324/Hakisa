"""Game configuration (YAML) loading.

Everything that is game-specific (key bindings, HUD regions, reward weights, model choice)
lives in ``hakisa/games/<name>.yaml`` so the code never hard-codes coordinates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

GAMES_DIR = Path(__file__).resolve().parent / "games"


@dataclass
class CaptureConfig:
    backend: str = "auto"  # auto | dxcam | mss | sim
    window_title: str | None = None  # substring of the window title; None => absolute screen region
    region: tuple[int, int, int, int] | None = None  # effective window (left, top, width, height)
    fps: float | None = None  # defaults to tick_hz


@dataclass
class MinimapConfig:
    region: tuple[int, int, int, int]  # relative to the captured image
    height: int = 64
    width: int = 64


@dataclass
class ObsConfig:
    """Model input.  ``crop`` is the *effective window* inside the captured image (None = everything).

    HUD probes read the full captured image, the policy only sees the crop (resized to height x width).
    """

    height: int
    width: int
    stack: int = 3
    cursor_channel: bool = False
    aux_dim: int = 0
    crop: tuple[int, int, int, int] | None = None
    minimap: MinimapConfig | None = None

    @property
    def channels(self) -> int:
        return 3 * self.stack + (1 if self.cursor_channel else 0)


@dataclass
class GameConfig:
    name: str
    title: str
    tick_hz: float
    capture: CaptureConfig
    obs: ObsConfig
    actions: dict[str, Any]
    probes: list[dict[str, Any]] = field(default_factory=list)
    reward: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    intents: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    source_path: Path | None = None

    @property
    def lookahead_ms(self) -> float:
        return float(self.actions.get("lookahead_ms", 0.0))

    @cached_property
    def action_space(self):  # -> hakisa.actions.ActionSpace (import here to avoid a cycle)
        from hakisa.actions import ActionSpace

        return ActionSpace.from_config(self)

    def section(self, dotted: str, default: Any = None) -> Any:
        """Read a nested value, e.g. ``cfg.section("training.bc.epochs", 10)``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _tuple4(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"expected [left, top, width, height], got {value!r}")
    return tuple(int(v) for v in value)  # type: ignore[return-value]


def parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ValueError(f"override must look like a.b.c=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip().split("."), yaml.safe_load(value)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    data = copy.deepcopy(data)
    for item in overrides or []:
        keys, value = parse_override(item)
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return data


def config_from_dict(data: dict[str, Any], source_path: Path | None = None) -> GameConfig:
    cap = data.get("capture", {}) or {}
    obs = data.get("obs", {}) or {}
    minimap = obs.get("minimap")
    obs_cfg = ObsConfig(
        height=int(obs["height"]),
        width=int(obs["width"]),
        stack=int(obs.get("stack", 3)),
        cursor_channel=bool(obs.get("cursor_channel", False)),
        aux_dim=int(obs.get("aux_dim", 0)),
        crop=_tuple4(obs.get("crop")),
        minimap=(
            MinimapConfig(
                region=_tuple4(minimap["region"]),  # type: ignore[arg-type]
                height=int(minimap.get("height", 64)),
                width=int(minimap.get("width", 64)),
            )
            if minimap
            else None
        ),
    )
    return GameConfig(
        name=str(data["name"]),
        title=str(data.get("title", data["name"])),
        tick_hz=float(data.get("tick_hz", 20)),
        capture=CaptureConfig(
            backend=str(cap.get("backend", "auto")),
            window_title=cap.get("window_title"),
            region=_tuple4(cap.get("region")),
            fps=cap.get("fps"),
        ),
        obs=obs_cfg,
        actions=data.get("actions", {}) or {},
        probes=list(data.get("probes", []) or []),
        reward=data.get("reward", {}) or {},
        model=data.get("model", {}) or {},
        training=data.get("training", {}) or {},
        intents=data.get("intents", {}) or {},
        record=data.get("record", {}) or {},
        runtime=data.get("runtime", {}) or {},
        raw=data,
        source_path=source_path,
    )


def available_games() -> list[str]:
    return sorted(p.stem for p in GAMES_DIR.glob("*.yaml"))


def load_game_config(name_or_path: str | Path, overrides: list[str] | None = None) -> GameConfig:
    """Load ``hakisa/games/<name>.yaml`` or an explicit YAML path, applying ``a.b=value`` overrides."""
    path = Path(name_or_path)
    if not path.suffix:
        path = GAMES_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"game config {name_or_path!r} not found (available: {available_games()})")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data = apply_overrides(data, overrides)
    return config_from_dict(data, source_path=path)
