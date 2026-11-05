# System 1 + System 2 on DanmakuDojo (keyboard, difficulty 2.0)

LLM: `tiny-qwen3` (1.10 M parameters, full training). **This is a tiny randomly initialised Qwen3-architecture model, NOT the pre-trained Qwen3-0.6B** (the Hugging Face Hub was unreachable from the container). It validates the pipeline only.

Plan accuracy of the supervised planner vs. hindsight plans (validation): {"parse_rate": 1.0, "exact": 0.48828125, "slot_acc/zone": 0.71875, "slot_acc/focus": 0.71484375, "slot_acc/bomb": 0.9921875}

Does System 1 obey intents? (bomb probability / focus rate / mean x position under a fixed intent): {"null/p_bomb": 0.0, "null/focus_rate": 0.126, "null/x": 0.423, "bomb_now/p_bomb": 0.0, "bomb_now/focus_rate": 0.149, "bomb_now/x": 0.547, "focus_on/p_bomb": 0.0, "focus_on/focus_rate": 0.169, "focus_on/x": 0.519, "focus_off/p_bomb": 0.0, "focus_off/focus_rate": 0.06, "focus_off/x": 0.485, "zone_bottom_left/p_bomb": 0.0, "zone_bottom_left/focus_rate": 0.101, "zone_bottom_left/x": 0.324, "zone_bottom_right/p_bomb": 0.0, "zone_bottom_right/focus_rate": 0.106, "zone_bottom_right/x": 0.548}

| Agent | Return | Deaths / episode | Survival (s) | Score | Bombs | Invalid plans |
|---|---|---|---|---|---|---|
| System 1 alone (intent-conditioned policy, null intent) | -0.37 ± 0.32 | 3.00 ± 0.00 | 16.2 | 3961 | 0.06 | — |
| System 1 + System 2 (supervised planner) | -0.30 ± 0.29 | 3.00 ± 0.00 | 16.6 | 4214 | 0.12 | 0.0% |
| System 1 + System 2 (after GRPO) | -0.02 ± 0.45 | 3.00 ± 0.00 | 17.5 | 5555 | 0.25 | 0.0% |


## Larger evaluation (96 episodes per arm, fresh seeds — `scripts/s2_eval_more.py`)

| Agent | Return | Survival (s) | Score | Bombs | Invalid plans |
|---|---|---|---|---|---|
| System 1 alone (null intent) | -0.44 ± 0.14 | 15.9 ± 0.6 | 4001 ± 412 | 0.05 | — |
| System 1 + System 2 (supervised planner) | -0.17 ± 0.20 | 16.9 ± 0.8 | 4856 ± 617 | 0.16 | 0.0% |
| System 1 + System 2 (after GRPO) | -0.18 ± 0.19 | 16.9 ± 0.7 | 5008 ± 716 | 0.22 | 0.0% |

Difference vs. System 1 alone: supervised planner +0.27 return (Welch t = 2.20), GRPO planner +0.26 (t = 2.18).
Reading: System 2 gives a modest but consistent gain (more survival and score), mostly by steering System 1's
zone/focus; 150 GRPO steps did not add a measurable gain over the supervised planner. Action sampling is not
seeded, so repeated evaluations of the same agent vary by about ±0.15 return (an earlier interrupted run measured
System 1 alone at -0.28) — treat the effect as modest.

### Intent following (System 1 obeys System 2?)

After fixing the intent-embedding initialisation and the EMA warm-up (see the design spec, section 8):
zone *bottom_left* → mean x 0.32, *bottom_right* → 0.55; *focus on* → 16.9 % focus vs *off* → 6.0 %;
*bomb now* raises the bomb probability 8× but it stays tiny (the demonstrator bombs only at the last instant).
The first run (archived as `README_attempt1.md`) had no intent following at all and no System 2 gain.
