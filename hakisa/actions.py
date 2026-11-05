"""Action spaces ("motor programs").

Two directions are handled here:

* **Recording → labels**: raw human input (held keys/buttons, press events, cursor) is turned into
  one integer label per action head.  This is what replaced the old manual labelling.
* **Model output → inputs**: integer actions are turned into a :class:`~hakisa.types.DesiredInput`
  that the :class:`~hakisa.thalamus.control.Controller` makes true on the machine.

Head kinds
----------
``categorical`` with ``mode: hold``
    The class whose key/button set is currently held (largest match wins), e.g. 8-way movement.
``categorical`` with ``mode: tap``
    An edge-triggered command, e.g. bomb or LoL abilities.  The best matching *press event* inside the
    label window wins (priority, then specificity, so ``ctrl+q`` beats ``q``).
``binary``
    Shorthand for a categorical head with choices ``off``/``on``.
``pointer``
    A mouse target expressed as two integers ``<name>_x`` and ``<name>_y`` on a *fine grid*
    (``grid * sub`` cells).  This is Hakisa 1.0's "X layer + Y layer" decomposition; the model computes
    it as P(x) · P(y | x) from a spatial heatmap (see ``basal_ganglia.heads``).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from hakisa.types import DesiredInput, InputEvent, InputSnapshot, Region, Tap

if TYPE_CHECKING:  # pragma: no cover
    from hakisa.config import GameConfig

MODIFIERS = frozenset({"ctrl", "shift", "alt"})


@dataclass(frozen=True)
class Choice:
    name: str
    keys: tuple[str, ...] = ()
    button: str | None = None
    pointer: str | None = None  # pointer head that provides this command's target
    priority: int = 0
    hold_ok: bool = False  # tap heads: holding the binding also counts (e.g. right button held = move)
    where: str = "any"  # any | main | minimap: where a button press must happen to match

    @property
    def is_null(self) -> bool:
        return not self.keys and self.button is None

    @property
    def specificity(self) -> int:
        return len(self.keys) + (1 if self.button else 0)


@dataclass
class HeadSpec:
    name: str
    kind: str  # "categorical" | "pointer"
    mode: str = "hold"
    choices: tuple[Choice, ...] = ()
    grid: tuple[int, int] = (0, 0)  # pointer: (rows, cols) of the coarse grid
    sub: tuple[int, int] = (1, 1)  # pointer: sub-cell bins (rows, cols)
    source: str = "obs"  # pointer: "obs" | "minimap"
    conditioned_on: str | None = None  # pointer: categorical head whose choice conditions it (LoL)
    weight: float = 1.0

    @property
    def num_classes(self) -> int:
        return len(self.choices)

    @property
    def fine_hw(self) -> tuple[int, int]:
        return self.grid[0] * self.sub[0], self.grid[1] * self.sub[1]

    @property
    def label_keys(self) -> list[str]:
        if self.kind == "pointer":
            return [f"{self.name}_x", f"{self.name}_y"]
        return [self.name]

    def index(self, choice_name: str) -> int:
        for i, c in enumerate(self.choices):
            if c.name == choice_name:
                return i
        raise KeyError(f"head {self.name!r} has no choice {choice_name!r}")

    @property
    def null_index(self) -> int:
        for i, c in enumerate(self.choices):
            if c.is_null:
                return i
        return 0


# ---------------------------------------------------------------------------------------------
# Timeline reconstruction (state at an arbitrary time from snapshots + events)
# ---------------------------------------------------------------------------------------------


@dataclass
class _State:
    keys: frozenset[str]
    buttons: frozenset[str]
    cursor: tuple[int, int]


class InputTimeline:
    """Replays snapshots + events so that the input state at *any* time can be queried.

    Snapshots (taken at every capture tick) are authoritative and re-synchronise the state, so a
    dropped OS event cannot corrupt the rest of a recording.
    """

    def __init__(self, snapshots: Iterable[InputSnapshot], events: Iterable[InputEvent]):
        items: list[tuple[float, int, Any]] = []
        for s in snapshots:
            items.append((s.t, 0, s))
        for e in events:
            items.append((e.t, 1, e))
        items.sort(key=lambda it: (it[0], it[1]))

        state = _State(frozenset(), frozenset(), (0, 0))
        self._times: list[float] = []
        self._states: list[_State] = []
        self._events: list[InputEvent] = []
        self._event_times: list[float] = []
        for t, kind, obj in items:
            if kind == 0:
                state = _State(obj.keys, obj.buttons, obj.cursor)
            else:
                ev: InputEvent = obj
                mods = frozenset(state.keys & MODIFIERS)
                cursor = state.cursor
                if ev.x is not None and ev.y is not None:
                    cursor = (int(ev.x), int(ev.y))
                keys, buttons = state.keys, state.buttons
                if ev.kind == "key":
                    keys = keys | {ev.name} if ev.pressed else keys - {ev.name}
                elif ev.kind == "button":
                    buttons = buttons | {ev.name} if ev.pressed else buttons - {ev.name}
                state = _State(keys, buttons, cursor)
                full = InputEvent(ev.t, ev.kind, ev.name, ev.pressed, cursor[0], cursor[1], mods)
                self._events.append(full)
                self._event_times.append(ev.t)
            self._times.append(t)
            self._states.append(state)

    def state_at(self, t: float) -> InputSnapshot:
        i = bisect.bisect_right(self._times, t) - 1
        if i < 0:  # before anything was observed: nothing held, cursor where it was first seen
            cursor = self._states[0].cursor if self._states else (0, 0)
            return InputSnapshot(t, frozenset(), frozenset(), cursor)
        s = self._states[i]
        return InputSnapshot(t, s.keys, s.buttons, s.cursor)

    def events_between(self, t0: float, t1: float, presses_only: bool = True) -> list[InputEvent]:
        lo = bisect.bisect_left(self._event_times, t0)
        hi = bisect.bisect_left(self._event_times, t1)
        evs = self._events[lo:hi]
        if presses_only:
            evs = [e for e in evs if e.pressed and e.kind in ("key", "button")]
        return evs


# ---------------------------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------------------------


def _parse_choice(name: str, spec: Any) -> Choice:
    if spec is None:
        return Choice(name)
    if isinstance(spec, (list, tuple)):
        return Choice(name, keys=tuple(str(k).lower() for k in spec))
    if isinstance(spec, dict):
        return Choice(
            name,
            keys=tuple(str(k).lower() for k in spec.get("keys", []) or []),
            button=spec.get("button"),
            pointer=spec.get("pointer"),
            priority=int(spec.get("priority", 0)),
            hold_ok=bool(spec.get("hold_ok", False)),
            where=str(spec.get("where", "any")),
        )
    raise ValueError(f"bad choice spec for {name!r}: {spec!r}")


def _swap_lr(name: str) -> str:
    return name.replace("left", "\0").replace("right", "left").replace("\0", "right")


class ActionSpace:
    def __init__(self, heads: list[HeadSpec], lookahead_s: float = 0.0, tick_s: float = 0.05):
        names = [h.name for h in heads]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate head names: {names}")
        self.heads = heads
        self.lookahead_s = float(lookahead_s)
        self.tick_s = float(tick_s)
        self._by_name = {h.name: h for h in heads}
        # pointer -> list of (categorical head name, [choice indices that use this pointer])
        self.pointer_users: dict[str, list[tuple[str, list[int]]]] = {}
        for p in self.pointer_heads:
            users = []
            for h in self.categorical_heads:
                idx = [i for i, c in enumerate(h.choices) if c.pointer == p.name]
                if idx:
                    users.append((h.name, idx))
            self.pointer_users[p.name] = users
        for h in self.categorical_heads:
            for c in h.choices:
                if c.pointer is not None and c.pointer not in self._by_name:
                    raise ValueError(f"choice {h.name}.{c.name} refers to unknown pointer {c.pointer!r}")

    # -- construction -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: "GameConfig") -> "ActionSpace":
        heads: list[HeadSpec] = []
        for spec in cfg.actions.get("heads", []):
            kind = spec["type"]
            name = spec["name"]
            weight = float(spec.get("weight", 1.0))
            if kind == "binary":
                on = {"keys": spec.get("keys", []), "button": spec.get("button")}
                heads.append(
                    HeadSpec(
                        name,
                        "categorical",
                        mode=spec.get("mode", "hold"),
                        choices=(Choice("off"), _parse_choice("on", on)),
                        weight=weight,
                    )
                )
            elif kind == "categorical":
                raw = spec["choices"]
                if isinstance(raw, dict):
                    choices = tuple(_parse_choice(k, v) for k, v in raw.items())
                else:
                    choices = tuple(_parse_choice(c["name"], c) for c in raw)
                heads.append(HeadSpec(name, "categorical", mode=spec.get("mode", "hold"), choices=choices, weight=weight))
            elif kind == "pointer":
                source = spec.get("source", "obs")
                stride = int(spec.get("stride", 4))
                if source == "minimap":
                    if cfg.obs.minimap is None:
                        raise ValueError(f"pointer {name!r} uses the minimap but obs.minimap is not configured")
                    h, w = cfg.obs.minimap.height, cfg.obs.minimap.width
                else:
                    h, w = cfg.obs.height, cfg.obs.width
                grid = tuple(spec.get("grid", (h // stride, w // stride)))
                sub = tuple(spec.get("sub", (stride, stride)))
                heads.append(
                    HeadSpec(
                        name,
                        "pointer",
                        grid=(int(grid[0]), int(grid[1])),
                        sub=(int(sub[0]), int(sub[1])),
                        source=source,
                        conditioned_on=spec.get("conditioned_on"),
                        weight=weight,
                    )
                )
            else:
                raise ValueError(f"unknown head type {kind!r}")
        return cls(heads, lookahead_s=cfg.lookahead_ms / 1000.0, tick_s=1.0 / cfg.tick_hz)

    # -- queries ------------------------------------------------------------------------------
    @property
    def categorical_heads(self) -> list[HeadSpec]:
        return [h for h in self.heads if h.kind == "categorical"]

    @property
    def pointer_heads(self) -> list[HeadSpec]:
        return [h for h in self.heads if h.kind == "pointer"]

    def head(self, name: str) -> HeadSpec:
        return self._by_name[name]

    def label_keys(self) -> list[str]:
        out: list[str] = []
        for h in self.heads:
            out.extend(h.label_keys)
        return out

    def pointer_always_active(self, name: str) -> bool:
        return not self.pointer_users.get(name)

    def noop_action(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for h in self.heads:
            if h.kind == "categorical":
                out[h.name] = h.null_index
            else:
                fh, fw = h.fine_hw
                out[f"{h.name}_x"] = fw // 2
                out[f"{h.name}_y"] = fh // 2
        return out

    def flip_permutation(self, head: HeadSpec) -> list[int]:
        """Class permutation for a horizontal mirror (left <-> right in choice names)."""
        names = [c.name for c in head.choices]
        return [names.index(_swap_lr(n)) if _swap_lr(n) in names else i for i, n in enumerate(names)]

    # -- geometry -----------------------------------------------------------------------------
    def _region_for(self, head: HeadSpec, region: Region, minimap_region: Region | None) -> Region:
        if head.source == "minimap":
            if minimap_region is None:
                raise ValueError(f"pointer {head.name!r} needs a minimap region")
            return minimap_region
        return region

    def pixel_to_pointer(
        self, head: HeadSpec | str, x: float, y: float, region: Region, minimap_region: Region | None = None
    ) -> tuple[int, int]:
        head = self.head(head) if isinstance(head, str) else head
        reg = self._region_for(head, region, minimap_region)
        u, v = reg.to_norm(x, y)
        fh, fw = head.fine_hw
        return min(int(u * fw), fw - 1), min(int(v * fh), fh - 1)

    def pointer_to_pixel(
        self, head: HeadSpec | str, fx: int, fy: int, region: Region, minimap_region: Region | None = None
    ) -> tuple[int, int]:
        head = self.head(head) if isinstance(head, str) else head
        reg = self._region_for(head, region, minimap_region)
        fh, fw = head.fine_hw
        return reg.from_norm((int(fx) + 0.5) / fw, (int(fy) + 0.5) / fh)

    # -- recording -> labels --------------------------------------------------------------------
    @staticmethod
    def _hold_match(head: HeadSpec, snap: InputSnapshot) -> int:
        best, best_spec = head.null_index, -1
        for i, c in enumerate(head.choices):
            if set(c.keys) <= snap.keys and (c.button is None or c.button in snap.buttons):
                if c.specificity > best_spec:
                    best, best_spec = i, c.specificity
        return best

    @staticmethod
    def _where_ok(c: Choice, x: int | None, y: int | None, minimap_region: Region | None) -> bool:
        if c.where == "any" or x is None or y is None:
            return True
        inside = minimap_region is not None and minimap_region.contains(x, y)
        return inside if c.where == "minimap" else not inside

    def _tap_match(
        self, head: HeadSpec, events: list[InputEvent], snap: InputSnapshot, minimap_region: Region | None
    ) -> tuple[int, tuple[int, int] | None]:
        best: tuple[int, int] | None = None  # (priority, specificity)
        best_idx, best_pos = head.null_index, None
        for ev in events:
            for i, c in enumerate(head.choices):
                if c.is_null:
                    continue
                if c.button is not None:
                    ok = ev.kind == "button" and ev.name == c.button and set(c.keys) <= ev.mods
                else:
                    ok = ev.kind == "key" and ev.name == c.keys[-1] and set(c.keys[:-1]) <= ev.mods
                if not ok or not self._where_ok(c, ev.x, ev.y, minimap_region):
                    continue
                score = (c.priority, c.specificity)
                if best is None or score > best:
                    best, best_idx, best_pos = score, i, (ev.x, ev.y)
        if best is not None:
            return best_idx, best_pos  # type: ignore[return-value]
        # nothing pressed in the window: held bindings that count as repeated commands
        for i, c in sorted(enumerate(head.choices), key=lambda ic: -ic[1].priority):
            if not c.hold_ok or c.is_null:
                continue
            held = set(c.keys) <= snap.keys and (c.button is None or c.button in snap.buttons)
            if held and self._where_ok(c, snap.cursor[0], snap.cursor[1], minimap_region):
                return i, snap.cursor
        return head.null_index, None

    def labels_from_timeline(
        self, timeline: InputTimeline, t: float, region: Region, minimap_region: Region | None = None
    ) -> dict[str, int]:
        """Labels for the frame captured at time ``t`` (uses ``t + lookahead``)."""
        tau = t + self.lookahead_s
        snap = timeline.state_at(tau)
        window = timeline.events_between(tau, tau + self.tick_s)
        labels: dict[str, int] = {}
        target_pos: dict[str, tuple[int, int] | None] = {}
        for h in self.categorical_heads:
            if h.mode == "hold":
                idx, pos = self._hold_match(h, snap), snap.cursor
            else:
                idx, pos = self._tap_match(h, window, snap, minimap_region)
            labels[h.name] = idx
            choice = h.choices[idx]
            if choice.pointer is not None:
                target_pos[choice.pointer] = pos if pos is not None else snap.cursor
        for p in self.pointer_heads:
            if self.pointer_always_active(p.name):
                pos = snap.cursor
            else:
                pos = target_pos.get(p.name)
            if pos is None:
                labels[f"{p.name}_x"] = -1
                labels[f"{p.name}_y"] = -1
            else:
                fx, fy = self.pixel_to_pointer(p, pos[0], pos[1], region, minimap_region)
                labels[f"{p.name}_x"] = fx
                labels[f"{p.name}_y"] = fy
        return labels

    # -- model output -> inputs -----------------------------------------------------------------
    def to_desired(
        self, action: dict[str, Any], region: Region, minimap_region: Region | None = None
    ) -> DesiredInput:
        keys: set[str] = set()
        buttons: set[str] = set()
        taps: list[Tap] = []
        cursor: tuple[int, int] | None = None
        pixels: dict[str, tuple[int, int]] = {}
        for p in self.pointer_heads:
            fx, fy = int(action.get(f"{p.name}_x", -1)), int(action.get(f"{p.name}_y", -1))
            if fx >= 0 and fy >= 0:
                pixels[p.name] = self.pointer_to_pixel(p, fx, fy, region, minimap_region)
        for h in self.categorical_heads:
            c = h.choices[int(action[h.name])]
            if h.mode == "hold":
                keys.update(c.keys)
                if c.button:
                    buttons.add(c.button)
            elif not c.is_null:
                taps.append(Tap(keys=c.keys, button=c.button, cursor=pixels.get(c.pointer) if c.pointer else None))
        for p in self.pointer_heads:
            if self.pointer_always_active(p.name) and p.name in pixels:
                cursor = pixels[p.name]
        return DesiredInput(frozenset(keys), frozenset(buttons), cursor, taps)

    def describe(self) -> str:
        lines = []
        for h in self.heads:
            if h.kind == "pointer":
                lines.append(f"{h.name}: pointer grid={h.grid} sub={h.sub} fine={h.fine_hw} source={h.source}")
            else:
                lines.append(f"{h.name}: {h.mode} {[c.name for c in h.choices]}")
        return "\n".join(lines)


def tracked_keys(space: ActionSpace) -> list[str]:
    """All keyboard keys that appear in any binding (recorded as a bitmask per tick)."""
    out: list[str] = []
    for h in space.categorical_heads:
        for c in h.choices:
            for k in c.keys:
                if k not in out:
                    out.append(k)
    return out


def tracked_buttons(space: ActionSpace) -> list[str]:
    out: list[str] = []
    for h in space.categorical_heads:
        for c in h.choices:
            if c.button and c.button not in out:
                out.append(c.button)
    return out


__all__ = [
    "MODIFIERS",
    "Choice",
    "HeadSpec",
    "ActionSpace",
    "InputTimeline",
    "tracked_keys",
    "tracked_buttons",
]
