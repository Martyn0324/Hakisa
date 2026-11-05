# System 1 + System 2 on DanmakuDojo (keyboard, difficulty 2.0)

LLM: `tiny-qwen3` (1.10 M parameters, full training). **This is a tiny randomly initialised Qwen3-architecture model, NOT the pre-trained Qwen3-0.6B** (the Hugging Face Hub was unreachable from the container). It validates the pipeline only.

Plan accuracy of the supervised planner vs. hindsight plans (validation): {"parse_rate": 1.0, "exact": 0.48828125, "slot_acc/zone": 0.71875, "slot_acc/focus": 0.71484375, "slot_acc/bomb": 0.9921875}

Does System 1 obey intents? (bomb probability / focus rate / mean x position under a fixed intent): {"null/p_bomb": 0.0, "null/focus_rate": 0.126, "null/x": 0.423, "bomb_now/p_bomb": 0.0, "bomb_now/focus_rate": 0.149, "bomb_now/x": 0.547, "focus_on/p_bomb": 0.0, "focus_on/focus_rate": 0.169, "focus_on/x": 0.519, "focus_off/p_bomb": 0.0, "focus_off/focus_rate": 0.06, "focus_off/x": 0.485, "zone_bottom_left/p_bomb": 0.0, "zone_bottom_left/focus_rate": 0.101, "zone_bottom_left/x": 0.324, "zone_bottom_right/p_bomb": 0.0, "zone_bottom_right/focus_rate": 0.106, "zone_bottom_right/x": 0.548}

| Agent | Return | Deaths / episode | Survival (s) | Score | Bombs | Invalid plans |
|---|---|---|---|---|---|---|
| System 1 alone (intent-conditioned policy, null intent) | -0.37 ± 0.32 | 3.00 ± 0.00 | 16.2 | 3961 | 0.06 | — |
| System 1 + System 2 (supervised planner) | -0.30 ± 0.29 | 3.00 ± 0.00 | 16.6 | 4214 | 0.12 | 0.0% |
| System 1 + System 2 (after GRPO) | -0.02 ± 0.45 | 3.00 ± 0.00 | 17.5 | 5555 | 0.25 | 0.0% |
