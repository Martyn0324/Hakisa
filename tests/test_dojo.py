import numpy as np

from hakisa.config import load_game_config
from hakisa.envs.dojo import DanmakuDojo
from hakisa.envs.dojo_expert import ScriptedExpert
from hakisa.limbic.probes import HudReader, build_probes


def _run(env, policy, steps):
    total, deaths = np.zeros(env.N), np.zeros(env.N)
    for _ in range(steps):
        _, r, term, trunc, info = env.step(policy())
        total += r
        deaths += info["life_lost"]
    return total, deaths


def test_shapes_and_determinism():
    for scheme, channels in (("keyboard", 9), ("mouse", 10)):
        env_a = DanmakuDojo(num_envs=3, scheme=scheme, seed=5)
        env_b = DanmakuDojo(num_envs=3, scheme=scheme, seed=5)
        obs, info = env_a.reset(5)
        env_b.reset(5)
        assert obs.shape == (3, channels, 112, 96) and obs.dtype == np.uint8
        rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
        for _ in range(120):
            oa, ra, *_ = env_a.step(env_a.random_actions(rng_a))
            ob, rb, *_ = env_b.step(env_b.random_actions(rng_b))
        assert np.array_equal(oa, ob) and np.allclose(ra, rb)


def test_snapshot_restore_and_replicate():
    env = DanmakuDojo(num_envs=2, scheme="keyboard", seed=3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(env.random_actions(rng))
    snap = env.snapshot()
    acts = [env.random_actions(rng) for _ in range(30)]
    first = [env.step(a)[0] for a in acts][-1]
    env.restore(snap)
    second = [env.step(a)[0] for a in acts][-1]
    assert np.array_equal(first, second)
    clone = env.replicate(1, 4)
    a = {k: np.repeat(v[1:2], 4) for k, v in acts[0].items()}
    obs, *_ = clone.step(a)
    assert all(np.array_equal(obs[0], obs[j]) for j in range(1, 4))  # common random numbers


def test_bomb_clears_bullets():
    env = DanmakuDojo(num_envs=1, scheme="keyboard", seed=1)
    idle = {"move": np.array([0]), "focus": np.array([0]), "shoot": np.array([1]), "bomb": np.array([0])}
    for _ in range(400):
        env.step(idle)
        if env.balive.sum() > 20 and env.bomb_cd[0] == 0 and env.invuln[0] == 0:
            break
    before = int(env.balive.sum())
    bombs_before = int(env.bombs[0])
    assert before > 20 and bombs_before > 0
    _, r, _, _, info = env.step(dict(idle, bomb=np.array([1])))
    assert info["bomb_used"][0] and env.bombs[0] == bombs_before - 1
    assert env.balive[0].sum() < before  # cleared (only bullets fired after the bomb remain)
    assert env.invuln[0] > 0
    assert r[0] < 0  # bombs cost reward


def test_expert_beats_random():
    for scheme in ("keyboard", "mouse"):
        env_r = DanmakuDojo(num_envs=8, scheme=scheme, seed=11)
        env_e = DanmakuDojo(num_envs=8, scheme=scheme, seed=11)
        rng = np.random.default_rng(0)
        expert = ScriptedExpert(env_e, seed=0)
        ret_r, deaths_r = _run(env_r, lambda: env_r.random_actions(rng), 500)
        ret_e, deaths_e = _run(env_e, expert.act, 500)
        assert deaths_e.sum() < deaths_r.sum(), (scheme, deaths_e.sum(), deaths_r.sum())
        assert ret_e.mean() > ret_r.mean()


def test_screen_hud_matches_internal_state():
    cfg = load_game_config("dojo_keyboard")
    env = DanmakuDojo(num_envs=1, scheme="keyboard", seed=2, cfg=cfg)
    hud = HudReader(build_probes(cfg.probes), stable_frames=1)
    rng = np.random.default_rng(3)
    for step in range(400):
        env.step(env.random_actions(rng))
        if step % 25 == 0:
            screen = env.render_screen(0)
            assert screen.shape == (448, 544, 3)
            read = hud.read(screen)
            assert read["lives"] == env.lives[0], (read, env.lives[0])
            assert read["bombs"] == env.bombs[0]
            assert read["score"] == env.score[0]
