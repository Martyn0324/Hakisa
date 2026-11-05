import numpy as np
import torch

from hakisa.config import load_game_config
from hakisa.consolidation import PPOConfig, PPOTrainer, compute_gae, evaluate_policy
from hakisa.dojo_tools import record_dojo_demos
from hakisa.hippocampus.dataset import BCDataset
from hakisa.hippocampus.storage import load_episodes
from hakisa.models import build_policy
from hakisa.study import StudyConfig, bc_loss, evaluate_bc, train_bc

torch.set_num_threads(2)


def test_compute_gae_by_hand():
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last = torch.tensor([0.2])
    g, lam = 0.9, 0.8
    adv, ret = compute_gae(rewards, values, dones, last, g, lam)
    d2 = 2.0 + g * 0.2 - 0.3
    d1 = 0.0 + 0.0 - 0.4  # episode ended after step 1: no bootstrap
    d0 = 1.0 + g * 0.4 - 0.5
    a2, a1 = d2, d1
    a0 = d0 + g * lam * a1
    assert torch.allclose(adv[:, 0], torch.tensor([a0, a1, a2]), atol=1e-6)
    assert torch.allclose(ret, adv + values)


class BanditEnv:
    """1-step episodes: reward 1 if move == 'right' else 0 (tiny images)."""

    def __init__(self, cfg, n=8):
        self.cfg, self.N = cfg, n
        self.right = cfg.action_space.head("move").index("right")
        self.obs = np.zeros((n, cfg.obs.channels, cfg.obs.height, cfg.obs.width), np.uint8)

    def reset(self, seed=None):
        return self.obs, {}

    def step(self, a):
        r = (a["move"] == self.right).astype(np.float32)
        done = np.ones(self.N, bool)
        info = {"episode_return": r.astype(np.float64), "episode_length": np.ones(self.N)}
        return self.obs, r, done, np.zeros(self.N, bool), info


def test_ppo_learns_a_bandit():
    cfg = load_game_config("dojo_keyboard", overrides=["obs.height=16", "obs.width=16"])
    torch.manual_seed(0)
    policy = build_policy(cfg)
    pcfg = PPOConfig(rollout_steps=16, epochs=4, minibatch_size=64, lr=3e-3, kl_coef=0.0, ent_coef=0.0,
                     critic_warmup_iters=1, reward_scale=False, target_kl=None)
    trainer = PPOTrainer(policy, pcfg, torch.device("cpu"))
    env = BanditEnv(cfg)
    obs, _ = env.reset()
    right = env.right
    with torch.no_grad():
        p0 = torch.softmax(policy(torch.from_numpy(obs)).logits["move"], -1)[0, right].item()
    trainer.total_iterations = 12
    actor_before = {k: v.clone() for k, v in policy.state_dict().items() if not k.startswith("value_head")}
    roll, obs = trainer.collect(env, obs, pcfg.rollout_steps)
    stats = trainer.update(roll)  # critic warm-up iteration
    assert stats["warmup"] == 1.0
    for k, v in policy.state_dict().items():  # only the value head may change during warm-up
        if k in actor_before:
            assert torch.equal(v, actor_before[k]), k
    for _ in range(11):
        roll, obs = trainer.collect(env, obs, pcfg.rollout_steps)
        stats = trainer.update(roll)
    with torch.no_grad():
        p1 = torch.softmax(policy(torch.from_numpy(obs)).logits["move"], -1)[0, right].item()
    assert stats["kl_ref"] > 0  # moved away from the reference ...
    assert p1 > max(0.5, 2 * p0)  # ... towards the rewarded action


def test_bc_learns_from_dojo_demos(tmp_path):
    record_dojo_demos(tmp_path, "keyboard", total_steps=640, num_envs=8, log_fn=lambda m: None)
    cfg = load_game_config("dojo_keyboard")
    eps = load_episodes([tmp_path])
    ds = BCDataset(eps, cfg)
    assert len(ds) == 640
    policy = build_policy(cfg)
    batch = {k: v for k, v in torch.utils.data.default_collate([ds[i] for i in range(32)]).items()}
    loss0, _ = bc_loss(policy, batch, None, 0.0, 0.5)
    hist = train_bc(policy, ds, ds, StudyConfig(epochs=3, batch_size=32, lr=3e-3, amp=False, log_every=1000, ema_decay=0.9),
                    tmp_path / "run", torch.device("cpu"), log_fn=lambda m: None)
    metrics = evaluate_bc(hist["ema"], ds, torch.device("cpu"))
    assert metrics["nll/total"] < float(loss0.detach())
    assert (tmp_path / "run" / "best.pt").exists()
    assert metrics["acc/shoot"] > 0.95  # the demonstrator always shoots


def test_evaluate_policy_runs_full_episodes():
    from hakisa.envs.dojo import DanmakuDojo

    cfg = load_game_config("dojo_keyboard")
    policy = build_policy(cfg)
    res = evaluate_policy(policy, lambda s: DanmakuDojo(num_envs=4, scheme="keyboard", seed=s, max_steps=60), episodes=4, seed=3)
    assert res["episodes"] == 4 and res["episode_length"] <= 60


def test_dagger_records_expert_labels_on_learner_states(tmp_path):
    from hakisa.dojo_tools import record_dojo_dagger
    from hakisa.hippocampus.storage import load_episodes

    calls = {"n": 0}

    def learner(obs):  # always "stand still, no shooting": very different from the expert
        calls["n"] += 1
        n = obs.shape[0]
        return {"move": np.zeros(n, np.int64), "focus": np.zeros(n, np.int64), "shoot": np.zeros(n, np.int64),
                "bomb": np.zeros(n, np.int64)}

    record_dojo_dagger(tmp_path, "keyboard", learner, total_steps=256, num_envs=4, beta=0.0, log_fn=lambda m: None)
    eps = load_episodes([tmp_path])
    assert calls["n"] == 64 and sum(e.length for e in eps) == 256
    assert all(e.meta["source"] == "dagger" for e in eps)
    shoot = np.concatenate([e.labels()["shoot"] for e in eps])
    assert shoot.mean() > 0.99  # labels are the expert's (always shoots), not the learner's


def test_ema_warmup_tracks_short_runs():
    from hakisa.study import EMA

    m = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(m.weight)
    ema = EMA(m, decay=0.999)
    with torch.no_grad():
        m.weight.fill_(1.0)
    for _ in range(500):
        ema.update(m)
    assert ema.module.weight.mean() > 0.9  # without warm-up it would be ~0.39 (1 - 0.999**500)
