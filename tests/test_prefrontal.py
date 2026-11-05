import time

import numpy as np
import torch

from hakisa.config import load_game_config
from hakisa.dojo_tools import record_dojo_demos
from hakisa.envs.dojo import DanmakuDojo
from hakisa.hippocampus.storage import load_episodes
from hakisa.models import build_policy
from hakisa.prefrontal.intents import IntentSchema, hindsight_for_episode, zone_of
from hakisa.prefrontal.llm import LoRALinear, load_llm
from hakisa.prefrontal.reasoner import Prefrontal, make_completion, make_prompt, s1_visual_tokens
from hakisa.prefrontal.runtime import AsyncPlanner
from hakisa.prefrontal.train import build_s2_samples, grpo_train, play_with_planner, train_s2_sl

torch.set_num_threads(2)


def _schema():
    return IntentSchema.from_config(load_game_config("dojo_keyboard"))


def test_plan_format_and_parse():
    s = _schema()
    plan = {"zone": "bottom_left", "focus": "on", "bomb": "hold"}
    text = "<think>\nblah\n</think>\n\n" + s.format_plan(plan)
    assert s.parse_plan(text) == plan
    assert s.parse_plan("PLAN: zone=nowhere focus=on") == {"zone": "any", "focus": "on", "bomb": "any"}
    assert s.parse_plan("no plan here") is None
    assert s.parse_plan("PLAN: garbage") is None
    ids = s.encode(plan)
    assert s.decode(ids) == plan
    assert s.decode(s.flip(ids))["zone"] == "bottom_right"
    assert zone_of(0.1, 0.95) == "bottom_left" and zone_of(0.5, 0.5) == "center"


def test_hindsight_from_demos(tmp_path):
    record_dojo_demos(tmp_path, "keyboard", total_steps=400, num_envs=4, log_fn=lambda m: None)
    cfg = load_game_config("dojo_keyboard")
    s = IntentSchema.from_config(cfg)
    ep = load_episodes([tmp_path])[0]
    ids = hindsight_for_episode(ep, cfg, s)
    assert ids.shape == (ep.length, 3)
    assert (ids[:, 0] > 0).all()  # zone always known in the simulator
    pos = ep.array("info/player_xy")
    assert s.decode(ids[0])["zone"] == zone_of(*pos[30])  # 1.5 s at 20 Hz = 30 ticks later


def _tiny_pf(mode="full"):
    torch.manual_seed(0)
    llm, tok = load_llm("tiny-qwen3")
    return Prefrontal(llm, tok, vis_dim=64, schema=_schema(), title="DanmakuDojo", spec="tiny-qwen3", train_mode=mode)


def test_prefrontal_loss_generate_and_lora():
    pf = _tiny_pf("lora")
    lora = [m for m in pf.llm.modules() if isinstance(m, LoRALinear)]
    assert lora and all(not m.base.weight.requires_grad for m in lora)
    trainable = {n for n, p in pf.named_parameters() if p.requires_grad}
    assert all(("lora_" in n) or n.startswith(("projector", "vis_pos")) for n in trainable)
    vis = torch.randn(2, 17, 64)
    prompts = [make_prompt("X", "lives 3", None, None)] * 2
    comps = [make_completion("PLAN: zone=top focus=on bomb=hold", "Calm.")] * 2
    loss = pf.loss(vis, prompts, comps)
    loss.backward()
    assert torch.isfinite(loss) and pf.vis_pos.grad is not None
    texts = pf.generate(vis, prompts, max_new_tokens=8)
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)


def test_sl_overfits_and_roundtrip(tmp_path):
    record_dojo_demos(tmp_path / "d", "keyboard", total_steps=160, num_envs=4, log_fn=lambda m: None)
    cfg = load_game_config("dojo_keyboard")
    s1 = build_policy(cfg)
    samples = build_s2_samples(load_episodes([tmp_path / "d"]), cfg, s1, IntentSchema.from_config(cfg), stride=20, think=False)
    assert samples and samples[0].vis.shape == (17, 64)
    pf = _tiny_pf("full")
    hist = train_s2_sl(pf, samples[:4], None, epochs=40, batch_size=4, lr=3e-3, lr_projector=3e-3, log_fn=lambda m: None)
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    pf.save(tmp_path / "s2")
    pf2 = Prefrontal.load(tmp_path / "s2")
    vis = torch.stack([samples[0].vis])
    assert torch.allclose(pf.loss(vis, [samples[0].prompt], [samples[0].completion]),
                          pf2.loss(vis, [samples[0].prompt], [samples[0].completion]), atol=1e-5)


def test_grpo_and_planner_play_run():
    cfg = load_game_config("dojo_keyboard")
    s1 = build_policy(cfg).eval()
    pf = _tiny_pf("full")
    before = {k: v.clone() for k, v in pf.adapter_state().items()}
    hist = grpo_train(pf, s1, pf.schema, lambda seed: DanmakuDojo(num_envs=2, scheme="keyboard", seed=seed),
                      steps=2, group=2, horizon=3, advance=2, log_fn=lambda m: None)
    assert len(hist) == 2 and np.isfinite(hist[-1]["kl"])
    after = pf.adapter_state()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    res = play_with_planner(s1, pf, lambda seed: DanmakuDojo(num_envs=2, scheme="keyboard", seed=seed, max_steps=20),
                            episodes=2, schema=pf.schema, replan_every=10)
    assert res["episodes"] == 2 and "invalid_plan_rate" in res


class _StubPF:
    title = "X"
    device = torch.device("cpu")

    def __init__(self, schema):
        self.schema = schema

    def eval(self):
        return self

    def plan(self, vis, prompts, **kw):
        return [("PLAN: zone=top focus=on bomb=hold", self.schema.parse_plan("PLAN: zone=top focus=on bomb=hold"))]


def test_async_planner_publishes_and_expires():
    cfg = load_game_config("dojo_keyboard")
    s1 = build_policy(cfg).eval()
    planner = AsyncPlanner(_StubPF(_schema()), hz=50, max_age_s=0.3, log_fn=lambda m: None)
    try:
        assert planner.latest_intent("cpu") is None
        obs = torch.zeros(1, cfg.obs.channels, cfg.obs.height, cfg.obs.width, dtype=torch.uint8)
        planner.observe(s1, obs, None, None, {"lives": 3})
        deadline = time.time() + 5
        intent = None
        while intent is None and time.time() < deadline:
            time.sleep(0.02)
            intent = planner.latest_intent("cpu")
        assert intent is not None and _schema().decode(intent[0].numpy())["zone"] == "top"
        assert s1_visual_tokens(s1, obs).shape == (1, 17, 64)
    finally:
        planner.stop()
    time.sleep(0.4)
    assert planner.latest_intent("cpu") is None  # stale plans are dropped


def test_save_load_non_default_tiny_size(tmp_path):
    from hakisa.prefrontal.llm import ByteTokenizer, tiny_qwen3

    torch.manual_seed(0)
    pf = Prefrontal(tiny_qwen3(ByteTokenizer().vocab_size, hidden=96, layers=3, heads=6), ByteTokenizer(), vis_dim=64,
                    schema=_schema(), title="X", spec="tiny-qwen3", train_mode="full")
    pf.save(tmp_path / "s2")
    pf2 = Prefrontal.load(tmp_path / "s2")
    vis = torch.randn(1, 17, 64)
    prompt = [make_prompt("X", "lives 3", None, None)]
    comp = ["PLAN: zone=top focus=on bomb=hold"]
    assert torch.allclose(pf.loss(vis, prompt, comp), pf2.loss(vis, prompt, comp), atol=1e-5)
