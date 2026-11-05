import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import torch

from hakisa.config import load_game_config
from hakisa.envs.lol_env import LoLEnv
from hakisa.envs.realtime import Safety
from hakisa.limbic.lol_client import LOL_FEATURES, LiveClientPoller, parse_allgamedata
from hakisa.limbic.lol_reward import LoLReward
from hakisa.models import build_policy
from hakisa.thalamus.capture import CallableSource
from hakisa.thalamus.control import Controller, LoggingBackend
from hakisa.types import Region


def payload(gold=500.0, cs=0, kills=0, deaths=0, hp=600.0, events=(), dead=False, items=()):
    me = {
        "riotId": "Hakisa#BR1", "summonerName": "Hakisa", "team": "ORDER", "isDead": dead, "respawnTimer": 0.0,
        "level": 3, "scores": {"kills": kills, "deaths": deaths, "assists": 1, "creepScore": cs, "wardScore": 2.0},
        "items": [{"itemID": i, "price": p, "count": 1} for i, p in items],
    }
    ally = {"riotId": "Duo#BR1", "team": "ORDER", "scores": {"kills": 2, "deaths": 1, "assists": 0, "creepScore": 10}}
    enemy = {"riotId": "Enemy#NA1", "team": "CHAOS", "scores": {"kills": 1, "deaths": 2, "assists": 0, "creepScore": 30}}
    return {
        "activePlayer": {
            "riotId": "Hakisa#BR1", "summonerName": "Hakisa", "level": 3, "currentGold": gold,
            "championStats": {"currentHealth": hp, "maxHealth": 600.0, "resourceValue": 100.0, "resourceMax": 300.0,
                              "attackDamage": 60.0, "armor": 30.0},
            "abilities": {"Q": {"abilityLevel": 2}, "W": {"abilityLevel": 1}, "E": {"abilityLevel": 0}, "R": {"abilityLevel": 0}},
        },
        "allPlayers": [me, ally, enemy],
        "events": {"Events": [{"EventID": 0, "EventName": "GameStart", "EventTime": 0.0}, *events]},
        "gameData": {"gameMode": "ARAM", "gameTime": 123.0},
    }


def test_parse_allgamedata():
    data = payload(events=[
        {"EventName": "TurretKilled", "TurretKilled": "Turret_T2_C_05_A", "KillerName": "Hakisa"},
        {"EventName": "TurretKilled", "TurretKilled": "Turret_T1_C_05_A", "KillerName": "Minion_T200"},
        {"EventName": "DragonKill", "DragonType": "Fire", "KillerName": "Duo#BR1"},
        {"EventName": "BaronKill", "KillerName": "Enemy#NA1"},
        {"EventName": "GameEnd", "Result": "Win"},
    ], items=[(1055, 450), (2003, 50)])
    s = parse_allgamedata(data)
    assert s is not None and s.team == "ORDER" and s.result == "Win"
    assert s["turrets_taken"] == 1 and s["turrets_lost"] == 1
    assert s["dragons_team"] == 1 and s["barons_enemy"] == 1
    assert s["team_kills"] == 2 and s["enemy_kills"] == 1 and s["kill_diff"] == 1
    assert s["items_value"] == 500 and s["ability_q"] == 2
    v = s.vector()
    assert v.shape == (len(LOL_FEATURES),) and np.isfinite(v).all()
    assert "KDA 0/0/1" in s.text()


@pytest.mark.parametrize("bad", [None, [], {}, {"activePlayer": {}}, {"activePlayer": {"riotId": "x"}, "allPlayers": []}, "oops"])
def test_parse_garbage_returns_none(bad):
    assert parse_allgamedata(bad) is None


def test_lol_reward_terms():
    rf = LoLReward({"gold": 0.01, "cs": 0.1, "kill": 1.0, "death": -2.0, "hp_loss": -1.0, "win": 5.0})
    rf.reset(parse_allgamedata(payload()))
    r, _ = rf(parse_allgamedata(payload(gold=600.0, cs=2)))
    assert np.isclose(r, 0.01 * 100 + 0.1 * 2)
    r, _ = rf(parse_allgamedata(payload(gold=150.0, cs=2, items=[(1055, 450)])))  # bought an item: no penalty
    assert np.isclose(r, 0.0)
    r, info = rf(parse_allgamedata(payload(gold=150.0, cs=2, items=[(1055, 450)], hp=300.0)))
    assert np.isclose(r, -0.5)  # lost half the HP
    r, info = rf(parse_allgamedata(payload(gold=150.0, cs=2, items=[(1055, 450)], hp=300.0, deaths=1, dead=True)))
    assert info["life_lost"] and np.isclose(r, -2.0)
    r, info = rf(parse_allgamedata(payload(gold=150.0, cs=2, items=[(1055, 450)], hp=600.0, deaths=1,
                                           events=[{"EventName": "GameEnd", "Result": "Win"}])))
    assert info["terminal"] and np.isclose(r, 5.0)  # respawn heal is not rewarded
    assert rf(None)[0] == 0.0


class _Server:
    """Mock Live Client: serves a scripted sequence of payloads (404 first, then a truncated body)."""

    def __init__(self, sequence):
        self.sequence = list(sequence)  # served in order; the last item stays (set ``current`` to change it)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                item = outer.sequence[0] if len(outer.sequence) == 1 else outer.sequence.pop(0)
                if len(outer.sequence) == 1 and outer.current is not None:
                    item = outer.current
                if item == 404:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = item if isinstance(item, bytes) else json.dumps(item).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.current = None
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/liveclientdata/allgamedata"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_poller_handles_errors():
    srv = _Server([404, b'{"activePlayer": {"lev', payload()])
    try:
        poller = LiveClientPoller(srv.url, interval=0.01)
        assert poller.poll_once() == (None, None)  # 404 (loading screen)
        assert poller.poll_once() == (None, None)  # truncated JSON
        stats, raw = poller.poll_once()
        assert stats is not None and raw["gameData"]["gameMode"] == "ARAM"
        assert poller.errors == 2
    finally:
        srv.close()


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.0)


def test_lol_env_end_to_end():
    cfg = load_game_config("lol")
    srv = _Server([payload()])
    poller = LiveClientPoller(srv.url, interval=0.01)
    poller.start()
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    source = CallableSource(lambda: frame, Region(0, 0, 1920, 1080))
    backend = LoggingBackend()
    controller = Controller(backend)
    clock = FakeClock()
    safety = Safety(cfg.runtime, controller, log_fn=lambda m: None)
    env = LoLEnv(cfg, source, controller, Region(0, 0, 1920, 1080), safety=safety, poller=poller,
                 clock=clock, sleep=clock.sleep, log_fn=lambda m: None, wait_timeout_s=5)
    try:
        import time

        deadline = time.time() + 5
        while poller.latest()[1] is None and time.time() < deadline:
            time.sleep(0.01)
        obs, info = env.reset()
        assert obs["obs"].shape == (1, 6, 144, 256) and obs["minimap"].shape == (1, 6, 64, 64)
        assert obs["aux"].shape == (1, 32)
        policy = build_policy(cfg).eval()
        with torch.no_grad():
            actions, *_ = policy.act(*(torch.from_numpy(obs[k]) for k in ("obs", "aux", "minimap")))
        at = cfg.action_space.head("action_type")
        a = {k: v.numpy() for k, v in actions.items()}
        a["action_type"] = np.array([at.index("q")])
        a["pointer_x"], a["pointer_y"] = np.array([128]), np.array([72])
        srv.current = payload(gold=700.0, cs=3, kills=1)
        while poller.latest()[1] is None or poller.latest()[1]["kills"] < 1:
            time.sleep(0.01)
        obs, r, term, trunc, info = env.step(a)
        assert ("key_down", "q") in backend.calls
        moves = [c for c in backend.calls if c[0] == "move_to"]
        assert moves and abs(moves[-1][1][0] - 960) < 10 and abs(moves[-1][1][1] - 540) < 10  # quick cast at the target
        assert r[0] > 0  # gold + CS + kill
        srv.current = payload(gold=700.0, cs=3, kills=1, events=[{"EventName": "GameEnd", "Result": "Win"}])
        while poller.latest()[1].result is None:
            time.sleep(0.01)
        env.wait_timeout_s = 0.5
        with pytest.raises(TimeoutError):  # GameEnd -> auto-reset waits for the next game
            env.step({**a, "action_type": np.array([0])})
    finally:
        poller.stop()
        srv.close()


def test_stats_vector_roundtrip_and_s2_samples(tmp_path):
    from hakisa.hippocampus.storage import EpisodeWriter, load_episodes
    from hakisa.limbic.lol_client import stats_from_vector
    from hakisa.prefrontal.intents import IntentSchema
    from hakisa.prefrontal.train import build_s2_samples

    s = parse_allgamedata(payload(gold=1234.0, cs=42, kills=3))
    back = stats_from_vector(s.vector())
    assert abs(back["gold"] - 1234.0) < 0.01 and abs(back["cs"] - 42) < 1e-3 and abs(back["kills"] - 3) < 1e-4
    assert "cs 42" in back.text()  # float32 round-trip must not display 41
    cfg = load_game_config("lol")
    space = cfg.action_space
    w = EpisodeWriter(tmp_path, {"game": "lol", "obs_region": [0, 0, 256, 144], "minimap_region": [0, 0, 64, 64]})
    for t in range(12):
        step = {f"label/{k}": (space.head("action_type").index("recall") if (k == "action_type" and t == 5) else 0)
                for k in space.label_keys()}
        step["label/minimap_pointer_x"] = step["label/minimap_pointer_y"] = -1
        step["aux"] = s.vector()
        w.add(np.zeros((144, 256, 3), np.uint8), step, minimap=np.zeros((64, 64, 3), np.uint8))
    w.close()
    eps = load_episodes([tmp_path])
    samples = build_s2_samples(eps, cfg, build_policy(cfg), IntentSchema.from_config(cfg), stride=4)
    assert len(samples) == 3 and samples[0].vis.shape[0] == 18  # 16 + global + minimap token
    assert "KDA 3/" in samples[0].prompt[1] and "objective=recall" in samples[0].completion
