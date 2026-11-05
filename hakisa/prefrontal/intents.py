"""Intents: the vocabulary System 2 (the LLM) uses to steer System 1 (the reflex policy).

A plan is one line, e.g. ``PLAN: zone=bottom_left focus=on bomb=hold``.  Every slot has ``any`` as
option 0 = "no opinion" (the null intent System 1 is also trained with).

Labels come for free by *hindsight* (no manual labelling): the plan a demonstration "had" at tick t
is read from what the player did next — where the ship was 1.5 s later, whether focus was held,
whether a bomb was pressed.  Rationales are short template explanations built from the same HUD
facts, so the model learns the ``<think> ... </think> PLAN: ...`` format; GRPO then refines the
plans against game outcomes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from hakisa.config import GameConfig

_PLAN_RE = re.compile(r"PLAN:\s*([^\n<]*)")


@dataclass
class IntentSchema:
    slots: dict[str, list[str]]

    @classmethod
    def from_config(cls, cfg: GameConfig) -> "IntentSchema":
        slots = (cfg.intents or {}).get("slots") or {}
        if not slots:
            raise ValueError(f"game {cfg.name!r} defines no intents.slots")

        def as_text(v) -> str:  # YAML 1.1 turns unquoted on/off/yes/no into booleans
            return ("on" if v else "off") if isinstance(v, bool) else str(v)

        clean = {str(k): [as_text(v) for v in opts] for k, opts in slots.items()}
        for name, opts in clean.items():
            if opts[0] != "any":
                raise ValueError(f"intent slot {name!r} must start with 'any' (the null intent)")
        return cls(clean)

    @property
    def names(self) -> list[str]:
        return list(self.slots)

    @property
    def sizes(self) -> list[int]:
        return [len(v) for v in self.slots.values()]

    def null(self) -> dict[str, str]:
        return {k: "any" for k in self.slots}

    def encode(self, plan: dict[str, str] | None) -> np.ndarray:
        plan = plan or {}
        return np.array([opts.index(plan.get(k, "any")) if plan.get(k, "any") in opts else 0
                         for k, opts in self.slots.items()], dtype=np.int64)

    def decode(self, ids) -> dict[str, str]:
        return {k: opts[int(i)] for (k, opts), i in zip(self.slots.items(), ids)}

    def format_plan(self, plan: dict[str, str]) -> str:
        return "PLAN: " + " ".join(f"{k}={plan.get(k, 'any')}" for k in self.slots)

    def parse_plan(self, text: str) -> dict[str, str] | None:
        m = _PLAN_RE.search(text or "")
        if not m:
            return None
        plan = self.null()
        ok = False
        for tok in m.group(1).split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            if k in self.slots and v in self.slots[k]:
                plan[k] = v
                ok = True
        return plan if ok else None

    def flip(self, ids: np.ndarray) -> np.ndarray:
        """Mirror left/right options (for horizontally flipped training images)."""
        out = np.array(ids, copy=True)
        for j, (k, opts) in enumerate(self.slots.items()):
            name = opts[int(out[j])]
            swapped = name.replace("left", "\0").replace("right", "left").replace("\0", "right")
            if swapped in opts:
                out[j] = opts.index(swapped)
        return out


# ---------------------------------------------------------------------------------------------
# Bullet hells
# ---------------------------------------------------------------------------------------------

ZONES = [["top_left", "top", "top_right"], ["left", "center", "right"], ["bottom_left", "bottom", "bottom_right"]]


def zone_of(x: float, y: float) -> str:
    col = min(int(np.clip(x, 0, 0.9999) * 3), 2)
    row = min(int(np.clip(y, 0, 0.9999) * 3), 2)
    return ZONES[row][col]


def player_positions(ep) -> np.ndarray | None:
    """(T, 2) normalised player positions from whatever the recording has (simulator, locate probe, cursor)."""
    if ep.has("info/player_xy"):
        return ep.array("info/player_xy").astype(np.float32)
    if ep.has("hud/player_x") and ep.has("hud/player_y") and ep.meta.get("capture_region"):
        crop = ep.meta.get("obs_region")
        cap = ep.meta["capture_region"]
        x = (ep.array("hud/player_x") + cap[0] - crop[0]) / crop[2]
        y = (ep.array("hud/player_y") + cap[1] - crop[1]) / crop[3]
        return np.stack([x, y], 1).astype(np.float32)
    if ep.has("cursor_norm") and ep.meta.get("game", "").startswith(("bullet_heaven", "dojo_mouse")):
        return ep.array("cursor_norm").astype(np.float32)  # the ship follows the cursor
    return None


def hindsight_bullet_hell(ep, schema: IntentSchema, horizon: int, focus_key: str) -> np.ndarray:
    """(T, S) intent ids for every tick of a bullet-hell episode."""
    labels = ep.labels()
    T = ep.length
    out = np.zeros((T, len(schema.slots)), dtype=np.int64)
    pos = player_positions(ep)
    focus = labels.get(focus_key)
    bomb = labels.get("bomb")
    for t in range(T):
        plan = schema.null()
        end = min(t + horizon, T - 1)
        if "zone" in schema.slots and pos is not None and np.isfinite(pos[end]).all():
            plan["zone"] = zone_of(*pos[end])
        if "focus" in schema.slots and focus is not None:
            plan["focus"] = "on" if focus[t : end + 1].mean() > 0.5 else "off"
        if "bomb" in schema.slots and bomb is not None:
            plan["bomb"] = "now" if bomb[t : t + max(horizon // 3, 1)].any() else "hold"
        out[t] = schema.encode(plan)
    return out


def bullet_hell_hud(ep, t: int) -> dict[str, Any]:
    hud: dict[str, Any] = {}
    for k in ("lives", "bombs", "score"):
        if ep.has(f"hud/{k}"):
            v = ep.array(f"hud/{k}")[t]
            hud[k] = None if not np.isfinite(v) else int(v)
    if ep.has("info/boss_hp"):
        hud["boss_hp"] = float(ep.array("info/boss_hp")[t])
    if ep.has("info/danger"):
        hud["clearance"] = float(ep.array("info/danger")[t])
    pos = player_positions(ep)
    if pos is not None and np.isfinite(pos[t]).all():
        hud["zone"] = zone_of(*pos[t])
    return hud


def danger_word(hud: dict[str, Any]) -> str:
    if "danger_p" in hud:  # System 1's amygdala at play time
        p = hud["danger_p"]
        return "high" if p > 0.5 else ("medium" if p > 0.15 else "low")
    c = hud.get("clearance")
    if c is None:
        return "unknown"
    return "high" if c < 4 else ("medium" if c < 12 else "low")


def bullet_hell_state_text(hud: dict[str, Any]) -> str:
    parts = []
    for k in ("lives", "bombs", "score"):
        if hud.get(k) is not None:
            parts.append(f"{k} {hud[k]}")
    if hud.get("boss_hp"):
        parts.append(f"boss hp {int(round(100 * hud['boss_hp']))}%")
    parts.append(f"danger {danger_word(hud)}")
    if hud.get("zone"):
        parts.append(f"I am at {hud['zone'].replace('_', ' ')}")
    return ", ".join(parts)


def bullet_hell_rationale(hud: dict[str, Any], plan: dict[str, str]) -> str:
    danger = danger_word(hud)
    bombs = hud.get("bombs")
    s = []
    if plan.get("bomb") == "now":
        s.append(f"Danger is {danger} and I still have {bombs} bombs, so I bomb now.")
    elif danger == "high":
        s.append("Bullets are very close but I can still dodge, so I keep my bombs.")
    zone = plan.get("zone", "any")
    if zone != "any":
        here = hud.get("zone")
        if here == zone:
            s.append(f"I stay at the {zone.replace('_', ' ')}.")
        else:
            s.append(f"I move to the {zone.replace('_', ' ')} where the gaps are.")
    if plan.get("focus") == "on":
        s.append("I focus to slip through the dense pattern.")
    elif plan.get("focus") == "off" and danger == "low":
        s.append("It is calm, so I move fast without focus.")
    return " ".join(s) or "Nothing special, I keep dodging and shooting."


# ---------------------------------------------------------------------------------------------
# League of Legends (coarse hindsight from labels + live-client extras)
# ---------------------------------------------------------------------------------------------


def _minimap_lane(fx: int, fy: int, fine_hw: tuple[int, int]) -> str:
    """Summoner's Rift minimap (blue base bottom-left): rough lane/region from a minimap click."""
    h, w = fine_hw
    u, v = (fx + 0.5) / w, (fy + 0.5) / h
    if (u < 0.2 and v > 0.8) or (u > 0.8 and v < 0.2):
        return "base"
    if v < 0.3 or u < 0.2:
        return "top"
    if u > 0.7 or v > 0.8:
        return "bot"
    if abs(u - (1 - v)) < 0.15:
        return "mid"
    return "jungle"


def hindsight_lol(ep, schema: IntentSchema, horizon: int, space) -> np.ndarray:
    labels = ep.labels()
    T = ep.length
    out = np.zeros((T, len(schema.slots)), dtype=np.int64)
    at = space.head("action_type")
    names = [c.name for c in at.choices]
    act = labels.get("action_type")
    mm_x, mm_y = labels.get("minimap_pointer_x"), labels.get("minimap_pointer_y")
    aux = ep.array("aux") if ep.has("aux") else None  # normalised LOL_FEATURES
    for t in range(T):
        plan = schema.null()
        end = min(t + horizon, T)
        window = act[t:end] if act is not None else np.array([], dtype=np.int64)
        chosen = [names[i] for i in window if i >= 0]
        if "recall" in chosen:
            plan["objective"] = "recall"
        elif aux is not None and end > t + 1:
            d = aux[end - 1] - aux[t]
            if d[6] > 0 or d[8] > 0:  # kills / assists went up
                plan["objective"] = "fight"
            elif d[9] > 0:  # cs went up
                plan["objective"] = "farm"
            elif d[16] > 0:  # turret taken
                plan["objective"] = "push"
        if mm_x is not None and "lane" in schema.slots:
            idx = [i for i in range(t, end) if mm_x[i] >= 0]
            if idx:
                plan["lane"] = _minimap_lane(int(mm_x[idx[-1]]), int(mm_y[idx[-1]]), space.head("minimap_pointer").fine_hw)
        if aux is not None and "risk" in schema.slots:
            hp = aux[t][3]
            died = end > t + 1 and aux[end - 1][7] > aux[t][7]
            plan["risk"] = "high" if died or hp < 0.25 else ("medium" if hp < 0.6 else "low")
        out[t] = schema.encode(plan)
    return out


def lol_rationale(state_text: str, plan: dict[str, str]) -> str:
    obj, lane, risk = plan.get("objective", "any"), plan.get("lane", "any"), plan.get("risk", "any")
    s = []
    if obj == "recall":
        s.append("My resources are low, I should recall and buy.")
    elif obj == "fight":
        s.append("There is a fight I can win, I join it.")
    elif obj == "farm":
        s.append("I farm minions to get gold.")
    elif obj == "push":
        s.append("The wave is strong, I push the tower.")
    if lane != "any":
        s.append(f"I go {lane}.")
    if risk == "high":
        s.append("This is risky, I must be careful.")
    return " ".join(s) or "I keep playing safe."


def hindsight_for_episode(ep, cfg: GameConfig, schema: IntentSchema) -> np.ndarray:
    horizon = int(round(float(cfg.intents.get("horizon_s", 1.5)) * cfg.tick_hz))
    if cfg.name == "lol":
        return hindsight_lol(ep, schema, horizon, cfg.action_space)
    focus_key = "focus" if "focus" in cfg.action_space.label_keys() else "slow"
    return hindsight_bullet_hell(ep, schema, horizon, focus_key)
