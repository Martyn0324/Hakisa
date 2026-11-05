"""Shaped reward for League of Legends from Live Client stats (OpenAI-Five-style rich shaping).

Every term is a *difference* between consecutive snapshots; weights live in ``games/lol.yaml``.
Gold is measured as net worth (current gold + item value) so buying items is not punished.
"""

from __future__ import annotations

from typing import Any

from hakisa.limbic.lol_client import LoLStats

DEFAULT_WEIGHTS = {
    "gold": 0.002, "cs": 0.15, "kill": 1.0, "assist": 0.5, "death": -1.5, "level": 0.3, "hp_loss": -0.5,
    "hp_gain": 0.1, "turret": 0.8, "turret_lost": -0.8, "dragon": 0.8, "baron": 1.5, "inhibitor": 1.0,
    "win": 10.0, "lose": -10.0,
}


class LoLReward:
    def __init__(self, weights: dict[str, Any] | None = None):
        self.w = dict(DEFAULT_WEIGHTS)
        for k, v in (weights or {}).items():
            if k in self.w:
                self.w[k] = float(v)
        self._prev: LoLStats | None = None

    def reset(self, stats: LoLStats | None = None) -> None:
        self._prev = stats

    @staticmethod
    def _net_worth(s: LoLStats) -> float:
        return s["gold"] + s["items_value"]

    def __call__(self, stats: LoLStats | None) -> tuple[float, dict[str, Any]]:
        info: dict[str, Any] = {"terminal": False, "result": None}
        if stats is None:
            return 0.0, info
        prev, self._prev = self._prev, stats
        if prev is None:
            return 0.0, info
        w = self.w

        def d(key: str) -> float:
            return stats[key] - prev[key]

        r = 0.0
        r += w["gold"] * max(0.0, self._net_worth(stats) - self._net_worth(prev))
        r += w["cs"] * max(0.0, d("cs"))
        r += w["kill"] * max(0.0, d("kills"))
        r += w["assist"] * max(0.0, d("assists"))
        r += w["death"] * max(0.0, d("deaths"))
        r += w["level"] * max(0.0, d("level"))
        if not stats["is_dead"] and not prev["is_dead"]:  # respawn heals to full: not a real heal
            dhp = d("hp_frac")
            r += w["hp_loss"] * (-dhp) if dhp < 0 else w["hp_gain"] * dhp
        r += w["turret"] * max(0.0, d("turrets_taken")) + w["turret_lost"] * max(0.0, d("turrets_lost"))
        r += w["dragon"] * (max(0.0, d("dragons_team")) - max(0.0, d("dragons_enemy")))
        r += w["baron"] * (max(0.0, d("barons_team")) - max(0.0, d("barons_enemy")))
        r += w["inhibitor"] * (max(0.0, d("inhibs_taken")) - max(0.0, d("inhibs_lost")))
        if stats.result is not None and prev.result is None:
            info["terminal"] = True
            info["result"] = stats.result
            r += w["win"] if str(stats.result).lower().startswith("win") else w["lose"]
        info["deaths"] = max(0.0, d("deaths"))
        info["kills"] = max(0.0, d("kills"))
        info["life_lost"] = info["deaths"] > 0
        return float(r), info
