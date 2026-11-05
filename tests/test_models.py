import numpy as np
import pytest
import torch

from hakisa.basal_ganglia.heads import PointerLogits
from hakisa.config import load_game_config
from hakisa.models import BulletHeavenNet, JigokuNet, LoLNet, build_policy, load_policy, save_policy

torch.set_num_threads(2)


def _batch(cfg, b=2):
    obs = torch.randint(0, 255, (b, cfg.obs.channels, cfg.obs.height, cfg.obs.width), dtype=torch.uint8)
    aux = torch.randn(b, cfg.obs.aux_dim) if cfg.obs.aux_dim else None
    mm = None
    if cfg.obs.minimap is not None:
        mm = torch.randint(0, 255, (b, 3 * cfg.obs.stack, cfg.obs.minimap.height, cfg.obs.minimap.width), dtype=torch.uint8)
    return obs, aux, mm


@pytest.mark.parametrize("game,cls", [("jigoku", JigokuNet), ("bullet_heaven2", BulletHeavenNet), ("lol", LoLNet),
                                      ("dojo_keyboard", JigokuNet), ("dojo_mouse", BulletHeavenNet)])
def test_game_nets_forward_and_act(game, cls):
    cfg = load_game_config(game)
    policy = build_policy(cfg)
    assert isinstance(policy, cls)
    assert policy.num_parameters() < 5_000_000
    obs, aux, mm = _batch(cfg)
    policy.eval()
    actions, logp, value, out = policy.act(obs, aux, mm)
    space = cfg.action_space
    assert set(actions) == set(space.label_keys())
    assert torch.isfinite(logp).all() and value.shape == (2,)
    for h in space.categorical_heads:
        assert out.logits[h.name].shape == (2, h.num_classes)
    for p in space.pointer_heads:
        gh, gw = p.grid
        assert out.pointer[p.name].cell.shape == (2, gh * gw)
        fh, fw = p.fine_hw
        assert (actions[f"{p.name}_x"] < fw).all() and (actions[f"{p.name}_y"] < fh).all()
    lp2, ent, v2, dist = policy.evaluate_actions(obs, actions, aux, mm)
    assert torch.allclose(lp2, logp, atol=1e-4)
    assert torch.allclose(dist.kl(dist, actions), torch.zeros(2), atol=1e-5)
    labels = {k: v.clone() for k, v in actions.items()}
    losses = dist.bc_losses(labels)
    total = sum(losses.values())
    policy.train()
    out = policy(obs, aux, mm, condition=labels)
    loss = sum(policy.distribution(out).bc_losses(labels).values()) + out.danger.sum() * 0
    loss.backward()
    assert torch.isfinite(total)


def test_pointer_x_then_y_matches_joint():
    torch.manual_seed(0)
    gh, gw = 3, 4
    cell = torch.randn(1, gh * gw) * 1.5
    offset = torch.zeros(1, 1, gh * gw)
    p = PointerLogits(cell.expand(20000, -1), offset.expand(20000, -1, -1), (gh, gw), (1, 1))
    fx, fy = p.sample()
    counts = np.bincount((fy * gw + fx).numpy(), minlength=gh * gw) / 20000
    joint = torch.softmax(cell[0], -1).numpy()
    assert np.abs(counts - joint).max() < 0.015
    # log_prob of fine coordinates = log joint
    lp = p.log_prob(torch.tensor([1]).expand(20000), torch.tensor([2]).expand(20000))[0]
    assert torch.isclose(lp, torch.log_softmax(cell[0], -1)[2 * gw + 1], atol=1e-5)


def test_pointer_entropy_and_kl_consistency():
    torch.manual_seed(1)
    a = PointerLogits(torch.randn(2, 6), torch.randn(2, 4, 6), (2, 3), (2, 2))
    b = PointerLogits(torch.randn(2, 6), torch.randn(2, 4, 6), (2, 3), (2, 2))
    # brute-force over all fine positions
    fh, fw = 4, 6
    xs, ys = torch.meshgrid(torch.arange(fw), torch.arange(fh), indexing="xy")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    for i in range(2):
        la = torch.stack([a.log_prob(x.view(1).expand(2), y.view(1).expand(2))[i] for x, y in zip(xs, ys)])
        lb = torch.stack([b.log_prob(x.view(1).expand(2), y.view(1).expand(2))[i] for x, y in zip(xs, ys)])
        assert torch.isclose(la.exp().sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(-(la.exp() * la).sum(), a.entropy()[i], atol=1e-4)
        assert torch.isclose((la.exp() * (la - lb)).sum(), a.kl(b)[i], atol=1e-4)


def test_inactive_pointer_does_not_count():
    cfg = load_game_config("lol")
    policy = build_policy(cfg).eval()
    obs, aux, mm = _batch(cfg)
    actions, *_ = policy.act(obs, aux, mm)
    at = cfg.action_space.head("action_type")
    actions["action_type"] = torch.tensor([at.index("recall"), at.index("recall")])
    lp_a, *_ = policy.evaluate_actions(obs, actions, aux, mm)
    actions["pointer_x"] = (actions["pointer_x"] + 7) % 256
    lp_b, *_ = policy.evaluate_actions(obs, actions, aux, mm)
    assert torch.allclose(lp_a, lp_b)  # recall has no target: the pointer is irrelevant


def test_null_intent_equals_none():
    cfg = load_game_config("dojo_keyboard")
    policy = build_policy(cfg).eval()
    obs, _, _ = _batch(cfg)
    n_slots = len(policy.intent_sizes)
    out_a = policy(obs)
    out_b = policy(obs, intent=torch.zeros(2, n_slots, dtype=torch.long))
    for k in out_a.logits:
        assert torch.allclose(out_a.logits[k], out_b.logits[k])


def test_save_load_roundtrip(tmp_path):
    cfg = load_game_config("dojo_mouse")
    policy = build_policy(cfg).eval()
    save_policy(policy, tmp_path / "p.pt", note="x")
    loaded = load_policy(tmp_path / "p.pt").eval()
    obs, _, _ = _batch(cfg)
    a = policy(obs)
    b = loaded(obs)
    assert torch.allclose(a.logits["fire"], b.logits["fire"])


def test_temperature_sharpens_sampling():
    torch.manual_seed(0)
    p = PointerLogits(torch.randn(1, 12).expand(4000, -1), torch.zeros(4000, 1, 12), (3, 4), (1, 1))
    fx1, _ = p.sample()
    fx2, _ = p.tempered(0.3).sample()
    top = int(torch.softmax(p.cell[0], -1).view(3, 4).sum(0).argmax())
    assert (fx2 == top).float().mean() > (fx1 == top).float().mean()
    cfg = load_game_config("dojo_keyboard")
    policy = build_policy(cfg).eval()
    obs, _, _ = _batch(cfg)
    a, lp, _, _ = policy.act(obs, temperature=0.5)
    lp2, *_ = policy.evaluate_actions(obs, a)
    assert torch.allclose(lp, lp2, atol=1e-5)  # log-prob reported at T=1


def test_intent_pathway_learns_quickly():
    """Regression: with std=0.02 intent embeddings this stayed at ~0.5/0.5 after 150 steps."""
    torch.manual_seed(0)
    cfg = load_game_config("dojo_keyboard", overrides=["obs.height=32", "obs.width=32"])
    pol = build_policy(cfg)
    opt = torch.optim.AdamW(pol.parameters(), lr=3e-4)
    for _ in range(120):
        obs = torch.randint(0, 255, (32, 9, 32, 32), dtype=torch.uint8)
        intent = torch.zeros(32, 3, dtype=torch.long)
        intent[:, 1] = torch.randint(1, 3, (32,))
        labels = {"move": torch.zeros(32, dtype=torch.long), "focus": (intent[:, 1] == 2).long(),
                  "shoot": torch.ones(32, dtype=torch.long), "bomb": torch.zeros(32, dtype=torch.long)}
        loss = sum(pol.distribution(pol(obs, intent=intent, condition=labels)).bc_losses(labels).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
    pol.eval()
    obs = torch.randint(0, 255, (16, 9, 32, 32), dtype=torch.uint8)
    on = torch.zeros(16, 3, dtype=torch.long)
    on[:, 1] = 2
    off = torch.zeros(16, 3, dtype=torch.long)
    off[:, 1] = 1
    with torch.no_grad():
        p_on = torch.softmax(pol(obs, intent=on).logits["focus"], -1)[:, 1].mean()
        p_off = torch.softmax(pol(obs, intent=off).logits["focus"], -1)[:, 1].mean()
    assert p_on > 0.8 and p_off < 0.2
