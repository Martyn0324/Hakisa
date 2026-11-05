# System 1 + System 2 on DanmakuDojo (keyboard, difficulty 2.0)

LLM: `tiny-qwen3` (1.10 M parameters, full training). **This is a tiny randomly initialised Qwen3-architecture model, NOT the pre-trained Qwen3-0.6B** (the Hugging Face Hub was unreachable from the container). It validates the pipeline only.

Plan accuracy of the supervised planner vs. hindsight plans (validation): {"parse_rate": 0.9765625, "exact": 0.43359375, "slot_acc/zone": 0.6953125, "slot_acc/focus": 0.671875, "slot_acc/bomb": 0.8984375}

| Agent | Return | Deaths / episode | Survival (s) | Score | Bombs | Invalid plans |
|---|---|---|---|---|---|---|
| System 1 alone (intent-conditioned policy, null intent) | 0.30 ± 0.49 | 3.00 ± 0.00 | 18.9 | 6172 | 0.34 | — |
| System 1 + System 2 (supervised planner) | 0.26 ± 0.50 | 3.00 ± 0.00 | 18.7 | 5637 | 0.19 | 14.0% |
| System 1 + System 2 (after GRPO) | 0.17 ± 0.35 | 3.00 ± 0.00 | 18.4 | 5581 | 0.25 | 0.0% |
