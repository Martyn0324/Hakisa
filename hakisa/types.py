"""Small shared value types used across Hakisa.

Kept dependency-free (no torch / numpy) so every module can import them cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Region:
    """A rectangle in *physical* screen pixels (call ``make_dpi_aware()`` on Windows first)."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_norm(self, x: float, y: float) -> tuple[float, float]:
        """Screen pixel -> normalised [0, 1) coordinates inside the region (clamped)."""
        u = (x - self.left) / max(self.width, 1)
        v = (y - self.top) / max(self.height, 1)
        eps = 1e-6
        return min(max(u, 0.0), 1.0 - eps), min(max(v, 0.0), 1.0 - eps)

    def from_norm(self, u: float, v: float) -> tuple[int, int]:
        """Normalised [0, 1] coordinates -> screen pixel inside the region."""
        x = self.left + min(max(u, 0.0), 1.0) * self.width
        y = self.top + min(max(v, 0.0), 1.0) * self.height
        return int(min(round(x), self.right - 1)), int(min(round(y), self.bottom - 1))

    def offset(self, dx: int, dy: int) -> "Region":
        return Region(self.left + dx, self.top + dy, self.width, self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


@dataclass(frozen=True)
class InputEvent:
    """A raw input event as seen by the OS hooks (or synthesised by tests / the simulator)."""

    t: float
    kind: str  # "key" | "button" | "move" | "scroll"
    name: str  # normalised key name ("w", "shift", "f12"), button ("left"/"right"/"middle"), "" for moves
    pressed: bool = True
    x: int | None = None
    y: int | None = None
    mods: frozenset[str] = frozenset()  # modifiers held at event time (filled by InputTimeline)


@dataclass(frozen=True)
class InputSnapshot:
    """Held keys / buttons and cursor position at time ``t``."""

    t: float
    keys: frozenset[str] = frozenset()
    buttons: frozenset[str] = frozenset()
    cursor: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class Tap:
    """A one-shot action: move cursor (optional) -> press keys in order -> click button -> release keys."""

    keys: tuple[str, ...] = ()
    button: str | None = None
    cursor: tuple[int, int] | None = None


@dataclass
class DesiredInput:
    """What the controller should make true for the current tick."""

    keys: frozenset[str] = frozenset()
    buttons: frozenset[str] = frozenset()
    cursor: tuple[int, int] | None = None
    taps: list[Tap] = field(default_factory=list)
