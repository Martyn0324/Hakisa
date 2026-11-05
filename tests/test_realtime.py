import numpy as np
import pytest
import torch

from hakisa.agent import Agent
from hakisa.config import load_game_config
from hakisa.envs.dojo_window import DojoVirtualWindow
from hakisa.envs.realtime import HakisaStop, RealtimeGameEnv, Safety
from hakisa.hippocampus.storage import Episode
from hakisa.limbic.probes import HudReader, build_probes
from hakisa.models import build_policy
from hakisa.record import Recorder
from hakisa.thalamus.control import Controller
from hakisa.thalamus.tracker import InputTracker
from hakisa.types import InputEvent


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.0)


def _make_env(scheme="keyboard", origin=(100, 50)):
    cfg = load_game_config("dojo_keyboard" if scheme == "keyboard" else "dojo_mouse")
    window = DojoVirtualWindow(scheme, seed=4, origin=origin)
    controller = Controller(window)
    clock = FakeClock()
    hud = HudReader(build_probes(cfg.probes), stable_frames=1)
    safety = Safety(cfg.runtime, controller, log_fn=lambda m: None)
    safety.started.set()
    env = RealtimeGameEnv(cfg, window, controller, window.region, hud=hud, safety=safety, clock=clock,
                          sleep=clock.sleep, log_fn=lambda m: None)
    return cfg, window, controller, env


def test_realtime_env_drives_the_virtual_game():
    cfg, window, controller, env = _make_env("keyboard")
    obs, info = env.reset()
    assert obs.shape == (1, 9, 112, 96)
    assert info["hud"]["lives"] == 3
    move = cfg.action_space.head("move")
    x0 = window.env.px[0]
    for _ in range(5):
        a = {"move": np.array([move.index("left")]), "focus": np.array([0]), "shoot": np.array([1]), "bomb": np.array([0])}
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r).all()
    assert window.env.px[0] < x0  # the injected "a" key really moved the ship left
    assert controller.held_keys == {"a", "z"}
    hud = env.last_hud
    assert hud["lives"] == window.env.lives[0] and hud["score"] == window.env.score[0]
    env.step({"move": np.array([0]), "focus": np.array([0]), "shoot": np.array([1]), "bomb": np.array([1])})
    assert window.env.bombs[0] == 1  # the bomb tap reached the game
    env.close()
    assert not controller.held_keys


def test_mouse_scheme_pointer_moves_the_cursor():
    cfg, window, controller, env = _make_env("mouse", origin=(10, 20))
    obs, _ = env.reset()
    assert obs.shape == (1, 10, 112, 96)
    p = cfg.action_space.head("pointer")
    fh, fw = p.fine_hw
    a = {"pointer_x": np.array([5]), "pointer_y": np.array([fh - 6]), "fire": np.array([1]), "bomb": np.array([0]), "slow": np.array([0])}
    for _ in range(25):  # 4.5 units/tick -> ~20 ticks to cover the distance
        env.step(a)
    # ship flew to the bottom-left target
    assert window.env.px[0] < 30 and window.env.py[0] > 180
    assert controller.held_buttons == {"left"}


def test_kill_switch_releases_and_stops():
    cfg, window, controller, env = _make_env("keyboard")
    env.reset()
    env.step({"move": np.array([1]), "focus": np.array([1]), "shoot": np.array([1]), "bomb": np.array([0])})
    assert controller.held_keys
    env.safety.trigger_kill()
    assert not controller.held_keys
    with pytest.raises(HakisaStop):
        env.step({"move": np.array([1]), "focus": np.array([0]), "shoot": np.array([1]), "bomb": np.array([0])})


def test_agent_runs_and_takeover_records(tmp_path):
    cfg, window, controller, env = _make_env("keyboard")
    policy = build_policy(cfg)
    tracker = InputTracker(clock=env.clock)
    recs = []

    def factory():
        rec = Recorder(cfg, tmp_path, window.region, tracker, source="takeover")
        recs.append(rec)
        return rec

    agent = Agent(policy, env, torch.device("cpu"), recorder_factory=factory, log_fn=lambda m: None)
    stats = agent.run(max_steps=10)
    assert stats["steps"] == 10 and stats["decide_ms_median"] > 0
    # human takes over for a few ticks, holding "d" -> recorded as a takeover episode with labels
    env.safety.takeover.set()
    tracker.feed(InputEvent(env.clock(), "key", "d", True))
    calls = {"n": 0}
    orig_grab = env.grab

    def grab_and_stop():
        calls["n"] += 1
        if calls["n"] >= 6:
            env.safety.takeover.clear()
        return orig_grab()

    env.grab = grab_and_stop
    agent._takeover()
    ep = Episode(agent.saved[0])
    assert ep.meta["source"] == "takeover" and ep.length >= 5
    labels = ep.labels()
    right = cfg.action_space.head("move").index("right")
    assert (labels["move"] == right).all()
