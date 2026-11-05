"""Reward functions for bullet hells (shared by Jigoku Kisetsukan, Bullet Heaven 2 and DanmakuDojo).

Design (cf. Alex Irpan's advice quoted in Hakisa 1.0: "if the reward has to be shaped, it should at
least be rich"):

* ``survival``  small reward every tick alive — dense signal that dodging is good
* ``life_lost`` big penalty when the lives counter drops — the event we really care about
* ``bomb_used`` moderate cost — bombs are a resource, spend them to *avoid* a death (cost < life)
* ``score``     scaled & clipped score delta — kills and grazing, never dominating survival
* ``game_over`` / ``stage_clear`` episode outcome
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class BulletHellRewardConfig:
    survival: float = 0.01
    life_lost: float = -1.0
    bomb_used: float = -0.3
    score_scale: float = 1e-4
    score_clip: float = 1.0
    game_over: float = -1.0
    stage_clear: float = 2.0
    extend: float = 0.5  # extra life gained

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "BulletHellRewardConfig":
        d = d or {}
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})


class BulletHellReward:
    def __init__(self, cfg: BulletHellRewardConfig | dict[str, Any] | None = None):
        self.cfg = cfg if isinstance(cfg, BulletHellRewardConfig) else BulletHellRewardConfig.from_dict(cfg)
        self._prev: dict[str, Any] = {}

    def reset(self, hud: dict[str, Any] | None = None) -> None:
        self._prev = {k: v for k, v in (hud or {}).items() if v is not None}

    def __call__(self, hud: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        c = self.cfg
        prev = self._prev
        r = c.survival
        info: dict[str, Any] = {"life_lost": False, "bomb_used": False, "game_over": False, "stage_clear": False,
                                "extend": False, "score_delta": 0.0}
        lives, bombs, score = hud.get("lives"), hud.get("bombs"), hud.get("score")
        if lives is not None and prev.get("lives") is not None:
            if lives < prev["lives"]:
                info["life_lost"] = True
                r += c.life_lost * (prev["lives"] - lives)
            elif lives > prev["lives"]:
                info["extend"] = True
                r += c.extend
        if bombs is not None and prev.get("bombs") is not None and not info["life_lost"]:
            if bombs < prev["bombs"]:
                info["bomb_used"] = True
                r += c.bomb_used * (prev["bombs"] - bombs)
        if score is not None and prev.get("score") is not None and score > prev["score"]:
            delta = float(score - prev["score"])
            info["score_delta"] = delta
            r += min(delta * c.score_scale, c.score_clip)
        if hud.get("game_over") and not prev.get("game_over"):
            info["game_over"] = True
            r += c.game_over
        if hud.get("stage_clear") and not prev.get("stage_clear"):
            info["stage_clear"] = True
            r += c.stage_clear
        for k, v in hud.items():
            if v is None:
                continue
            if k == "score" and prev.get("score") is not None and v < prev["score"]:
                continue  # scores only go up within an episode: a lower read is a misread
            prev[k] = v
        return float(r), info
