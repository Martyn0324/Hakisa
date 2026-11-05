"""League of Legends Live Client Data API (local, read-only, provided by the game client).

Endpoint: ``https://127.0.0.1:2999/liveclientdata/allgamedata`` (self-signed Riot certificate).
Top-level keys: ``activePlayer``, ``allPlayers``, ``events`` (``{"Events": [...]}``), ``gameData``.

The schema is only community-documented (Overwolf docs, open-source clients) and changes over time
(e.g. ``summonerName`` -> ``riotId``), so every access here is defensive: a missing field becomes 0
and an unusable payload returns ``None`` instead of raising.
"""

from __future__ import annotations

import json
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOL_FEATURES = [
    "game_time", "level", "gold", "hp_frac", "max_hp", "resource_frac", "kills", "deaths", "assists", "cs",
    "ward_score", "is_dead", "respawn_timer", "team_kills", "enemy_kills", "kill_diff", "turrets_taken",
    "turrets_lost", "dragons_team", "dragons_enemy", "barons_team", "barons_enemy", "inhibs_taken",
    "inhibs_lost", "items_count", "items_value", "ability_q", "ability_w", "ability_e", "ability_r",
    "attack_damage", "armor",
]
_SCALE = {
    "game_time": 1800, "level": 18, "gold": 3000, "hp_frac": 1, "max_hp": 3000, "resource_frac": 1, "kills": 10,
    "deaths": 10, "assists": 15, "cs": 200, "ward_score": 30, "is_dead": 1, "respawn_timer": 60, "team_kills": 30,
    "enemy_kills": 30, "kill_diff": 10, "turrets_taken": 11, "turrets_lost": 11, "dragons_team": 4,
    "dragons_enemy": 4, "barons_team": 2, "barons_enemy": 2, "inhibs_taken": 3, "inhibs_lost": 3,
    "items_count": 7, "items_value": 20000, "ability_q": 5, "ability_w": 5, "ability_e": 5, "ability_r": 3,
    "attack_damage": 300, "armor": 200,
}


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


@dataclass
class LoLStats:
    values: dict[str, float]
    team: str | None = None
    result: str | None = None  # "Win" / "Lose" once the game ended
    events: list[dict[str, Any]] = field(default_factory=list)

    def __getitem__(self, key: str) -> float:
        return self.values.get(key, 0.0)

    def vector(self) -> np.ndarray:
        return np.array([self.values.get(k, 0.0) / _SCALE[k] for k in LOL_FEATURES], dtype=np.float32)

    def text(self) -> str:
        v = self.values
        hp = int(round(100 * v.get("hp_frac", 0)))
        return (
            f"time {int(v.get('game_time', 0)) // 60}:{int(v.get('game_time', 0)) % 60:02d}, level {int(v.get('level', 0))}, "
            f"hp {hp}%, gold {int(v.get('gold', 0))}, KDA {int(v.get('kills', 0))}/{int(v.get('deaths', 0))}/"
            f"{int(v.get('assists', 0))}, cs {int(v.get('cs', 0))}, team kills {int(v.get('team_kills', 0))}-"
            f"{int(v.get('enemy_kills', 0))}, turrets {int(v.get('turrets_taken', 0))}-{int(v.get('turrets_lost', 0))}"
            + (", DEAD" if v.get("is_dead") else "")
        )


def _player_names(p: dict[str, Any]) -> set[str]:
    names = set()
    for k in ("riotId", "summonerName", "riotIdGameName"):
        if p.get(k):
            names.add(str(p[k]))
            names.add(str(p[k]).split("#")[0])
    return names


def _structure_team(name: str) -> str | None:
    """Turret/inhibitor names: ``Turret_T1_...`` belongs to ORDER (blue), ``T2`` to CHAOS (red)."""
    if "_T1_" in name or name.startswith("Turret_T1") or "Barracks_T1" in name:
        return "ORDER"
    if "_T2_" in name or name.startswith("Turret_T2") or "Barracks_T2" in name:
        return "CHAOS"
    return None


def parse_allgamedata(data: Any, me: str | None = None) -> LoLStats | None:
    if not isinstance(data, dict):
        return None
    ap = data.get("activePlayer")
    players = data.get("allPlayers")
    gd = data.get("gameData") or {}
    if not isinstance(ap, dict) or not isinstance(players, list) or not players:
        return None
    my_names = _player_names(ap)
    if me:
        my_names.add(me)
    mine = next((p for p in players if isinstance(p, dict) and _player_names(p) & my_names), None)
    if mine is None:
        return None
    team = mine.get("team")
    team_of: dict[str, str] = {}
    for p in players:
        if isinstance(p, dict):
            for n in _player_names(p):
                team_of[n] = p.get("team")
    cs = ap.get("championStats") or {}
    scores = mine.get("scores") or {}
    items = [i for i in (mine.get("items") or []) if isinstance(i, dict)]
    abilities = ap.get("abilities") or {}
    max_hp = _num(cs.get("maxHealth"), 1.0) or 1.0
    res_max = _num(cs.get("resourceMax"), 0.0)
    v: dict[str, float] = {
        "game_time": _num(gd.get("gameTime")),
        "level": _num(ap.get("level")),
        "gold": _num(ap.get("currentGold")),
        "hp_frac": _num(cs.get("currentHealth")) / max_hp,
        "max_hp": max_hp,
        "resource_frac": (_num(cs.get("resourceValue")) / res_max) if res_max > 0 else 0.0,
        "kills": _num(scores.get("kills")),
        "deaths": _num(scores.get("deaths")),
        "assists": _num(scores.get("assists")),
        "cs": _num(scores.get("creepScore")),
        "ward_score": _num(scores.get("wardScore")),
        "is_dead": 1.0 if mine.get("isDead") else 0.0,
        "respawn_timer": _num(mine.get("respawnTimer")),
        "items_count": float(len(items)),
        "items_value": sum(_num(i.get("price")) * max(_num(i.get("count"), 1.0), 1.0) for i in items),
        "attack_damage": _num(cs.get("attackDamage")),
        "armor": _num(cs.get("armor")),
    }
    for key, slot in (("ability_q", "Q"), ("ability_w", "W"), ("ability_e", "E"), ("ability_r", "R")):
        v[key] = _num((abilities.get(slot) or {}).get("abilityLevel"))
    team_kills = enemy_kills = 0.0
    for p in players:
        if not isinstance(p, dict):
            continue
        k = _num((p.get("scores") or {}).get("kills"))
        if p.get("team") == team:
            team_kills += k
        else:
            enemy_kills += k
    v["team_kills"], v["enemy_kills"], v["kill_diff"] = team_kills, enemy_kills, team_kills - enemy_kills
    counts = {k: 0.0 for k in ("turrets_taken", "turrets_lost", "inhibs_taken", "inhibs_lost",
                               "dragons_team", "dragons_enemy", "barons_team", "barons_enemy")}
    result = None
    events = ((data.get("events") or {}).get("Events")) or []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        name = ev.get("EventName")
        if name in ("TurretKilled", "InhibKilled"):
            owner = _structure_team(str(ev.get("TurretKilled") or ev.get("InhibKilled") or ""))
            kind = "turrets" if name == "TurretKilled" else "inhibs"
            if owner is not None and team is not None:
                counts[f"{kind}_lost" if owner == team else f"{kind}_taken"] += 1
        elif name in ("DragonKill", "BaronKill"):
            killer_team = team_of.get(str(ev.get("KillerName", "")))
            kind = "dragons" if name == "DragonKill" else "barons"
            if killer_team is not None:
                counts[f"{kind}_team" if killer_team == team else f"{kind}_enemy"] += 1
        elif name == "GameEnd":
            result = ev.get("Result")
    v.update(counts)
    return LoLStats(values=v, team=team, result=result, events=[e for e in events if isinstance(e, dict)])


class LiveClientPoller(threading.Thread):
    """Background poller; ``latest()`` returns (timestamp, stats or None, raw json or None)."""

    def __init__(self, url: str = "https://127.0.0.1:2999/liveclientdata/allgamedata", interval: float = 0.5,
                 timeout: float = 1.0, clock=time.perf_counter):
        super().__init__(daemon=True)
        self.url = url
        self.interval = interval
        self.timeout = timeout
        self.clock = clock
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[float, LoLStats | None, Any] = (0.0, None, None)
        self.errors = 0
        host = urllib.parse.urlparse(url).hostname or ""
        handlers: list[Any] = [urllib.request.ProxyHandler({})]  # never send local traffic to a proxy
        if url.startswith("https"):
            ctx = ssl.create_default_context()
            if host in ("127.0.0.1", "localhost"):  # Riot's self-signed certificate on the local machine only
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=ctx))
        self._opener = urllib.request.build_opener(*handlers)

    def poll_once(self) -> tuple[LoLStats | None, Any]:
        try:
            with self._opener.open(self.url, timeout=self.timeout) as r:
                raw = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            self.errors += 1
            return None, None
        return parse_allgamedata(raw), raw

    def run(self) -> None:
        while not self._stop.is_set():
            stats, raw = self.poll_once()
            with self._lock:
                self._latest = (self.clock(), stats, raw)
            self._stop.wait(self.interval)

    def latest(self) -> tuple[float, LoLStats | None, Any]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
