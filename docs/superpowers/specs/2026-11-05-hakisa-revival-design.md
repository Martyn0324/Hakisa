# Hakisa Revival — Design Spec (2026-11-05)

> Status: written autonomously at the owner's request ("you're operating autonomously ... I'll only
> evaluate the results once ... Delivered"). Interrupt at any time to redirect.

## 1. Intent (what the owner said vs. what I assumed)

**Said (requirements):**
- Hakisa = an "image-classifier-like" game-playing agent: each frame (state) is classified into an action.
  Keep it simple and efficient, in the spirit of AlphaGo / AlphaStar / PPO-on-Atari, *not* an LLM computer-use agent.
- Ladder of four tasks:
  1. Rebuild the repo: human-gameplay dataset creation (the owner's own gameplay), SL baseline, RL fine-tuning.
  2. Bullet-hell framework: *Jigoku Kisetsukan* (WASD move, Shift focus, Z shoot, X bomb; Steam) and
     *Bullet Heaven 2* (mouse; left click hold = shoot w/ autoshoot, right click = bomb). One architecture per game,
     reward functions, plus a third game I can actually test with.
  3. Evolve Hakisa into an LLM-with-reasoning player (advised: Qwen3-0.6B, must run on a GTX 1650 Ti 4 GB).
  4. League of Legends framework: QWER, 1–9, mouse; skill = key + mouse position (quick cast); easy to adjust later.
- Rules: only pre-trained **text/embedding** models allowed; pre-trained image classifiers/feature extractors are
  **forbidden**. No human-in-the-loop evaluation mid-training. Prefer efficient models. Avoid running on the owner's PC.

**Assumed (owner may correct):**
- The owner runs Windows 10/11 with a 1080p display (possibly with DPI scaling), Python 3.11–3.13, CUDA GPU (GTX 1650 Ti).
- Exact game menus / HUD positions / restart sequences are unknown to me → they are **config** (YAML) with a
  calibration tool, not hard-coded.
- "Success" for delivery = a working, tested, documented pipeline + quantitative evidence on a simulator I can run,
  not a trained model for the real games (I cannot run Windows/Steam/LoL from the cloud container).

## 2. Hard facts about the build environment

- Cloud container: 4 CPU cores, 15 GB RAM, **no GPU**; `huggingface.co` blocked by network policy (as of writing);
  Colab cannot be driven from here. → All experiments are CPU-scaled; GPU-scale runs ship as ready-to-run
  scripts/notebooks; Qwen3 code is validated with a tiny randomly initialised Qwen3-architecture model and the real
  Qwen3-0.6B path is provided but not executed here.

## 3. Research summary (2025–2026) that shaped the design

| Finding | Consequence for Hakisa |
|---|---|
| VPT (OpenAI, 2022) and NitroGen (NVIDIA, Jan 2026, 40k h / 1000+ games) learn keyboard/mouse/gamepad policies by **behaviour cloning on gameplay with frame-aligned action labels**; NitroGen recovers labels from on-screen input overlays. | Record the owner's play with *synchronised input logs*: labels come for free (this kills the old labelling pain). |
| Counter-Strike BC (Pearce & Zhu, 2022) plays a modern game from pixels with BC on human data and a discretised mouse. | Pure-pixel BC works for real-time games when actions are *held-key states* sampled at a fixed rate. |
| Helix (Figure, 2025): System 2 VLM at 7–9 Hz + System 1 visuomotor policy at 200 Hz. GR00T N1 (NVIDIA, 2025): VLM + fast action module, trained jointly. Hi Robot (Physical Intelligence, 2025): high-level VLM issues language commands to a low-level policy. | A reasoning LLM is too slow for frame-level control, so Hakisa becomes **dual-system**: fast CNN reflex policy (S1, 20–30 Hz) + Qwen3 "prefrontal" planner (S2, ~1 Hz) that emits *intents* consumed by S1. |
| Lumine (2025, Qwen2-VL-7B, Genshin Impact) and Game-TARS (ByteDance, 2025) use **sparse / adaptive thinking**: reason only when needed. | S2 runs asynchronously and only occasionally; S1 never waits for it. |
| Impoola (RLC 2025): replacing the flatten→Linear of IMPALA-CNN with global average pooling improves image-based deep RL; the flatten layer held ~84 % of IMPALA-CNN's parameters. | No giant flatten→Linear "bottleneck" (the old Hakisa had a 44 M-parameter one). Use pooling/attention pooling. |
| MobileNetV4 (2024): Universal Inverted Bottleneck (UIB) = IB / ConvNeXt / FFN family of depthwise blocks; EfficientNetV2: Fused-MBConv is faster in early high-resolution stages. | From-scratch encoder: Fused-MBConv early, UIB/MBConv (depthwise) later, one cheap attention stage at 1/16 resolution. |
| PIRLNav (CVPR 2023): BC→RL fine-tuning works, but BC gives an actor without a critic; naive RL init "often leads to poor outcomes". AlphaStar regularised RL with KL to the supervised policy. DAPG (2018) keeps a decaying demo term during RL. | RL fine-tune = PPO with (a) critic warm-up while the actor is frozen, (b) KL(π‖π_BC) anchor, (c) decaying BC loss on human demos. |
| GRPO (DeepSeekMath, 2024) — group-relative policy optimisation without a value network, the standard RL recipe for reasoning LLMs. | S2 (Qwen3) is RL-tuned with GRPO on intents, with rewards from game outcomes (simulator) or S1's critic (real games). |

## 4. Architecture (neuro-inspired names kept from the original project)

```
            ┌─────────────────────────── hakisa/ ───────────────────────────┐
 screen ──► │ thalamus/   capture (dxcam|mss|sim) · input control (SendInput)│
 keys/mouse │             input tracker (recording) · window/DPI helpers     │
            │ hippocampus/ episode storage (chunked npz) · BC dataset        │
            │ cortex/     visual encoder (from scratch)                      │
            │ basal_ganglia/ action heads: categorical · binary · pointer    │
            │ superior colliculus = pointer head (X then Y|X) for the mouse  │
            │ amygdala    = "danger" auxiliary head (life lost in next ~1 s)  │
            │ limbic/     HUD probes · reward functions · LoL live client     │
            │ envs/       real-time game env · DanmakuDojo simulator          │
            │ study.py    supervised learning (behaviour cloning)             │
            │ consolidation.py RL fine-tuning (PPO + KL-to-BC, async)         │
            │ prefrontal/ System 2: Qwen3 + vision projector + intents        │
            │ agent.py    runtime loop (S1 + optional S2), kill switch, takeover│
            └────────────────────────────────────────────────────────────────┘
```

### 4.1 Observation
- Capture the **effective window** (the owner's trick) as `[left, top, width, height]` or by window title + crop.
- Resize to the model input (e.g. Jigoku 192×160, Bullet Heaven 160×208, LoL 144×256 main + 64×64 minimap).
- Stack the last `k` frames (default 3, RGB) as channels → motion of bullets is visible.
- Normalisation happens inside the model (uint8 travels from capture to GPU → cheap).
- Optional `aux` vector (cursor position, HUD readings, LoL live-client stats).

### 4.2 Actions = held-input **states** sampled every tick (not single events)
- Jigoku: `move` ∈ {none, 8 directions} (9-way), `focus` (Shift), `shoot` (Z), `bomb` (X, edge/tap).
- Bullet Heaven 2: `pointer` (mouse target in the effective window), `fire` (left button), `bomb` (right click tap),
  `slow` (Shift).
- LoL: `action_type` (noop, move-click, attack-move, Q, W, E, R, D, F, item 1–9, recall, level-up Q/W/E/R, stop)
  → `pointer` **conditioned on** the action type (AlphaStar-style autoregressive head); quick cast = move cursor, tap key.
  Extra `minimap_pointer` for long-distance moves.
- **Pointer head = the owner's X/Y decomposition, made exact.** A convolutional decoder produces a score map
  `s(x, y)` on a coarse grid (stride 4) plus sub-cell offsets. Sampling is done as `x ~ P(x)` (marginal over rows)
  then `y ~ P(y | x)` — this *is* the X-layer / Y-layer idea, but without the independence assumption that made
  `(x1, y2)` combos likely, and with ~1 k parameters instead of `Linear(256 → 1920)`.
- Labels from recordings: `label(t) = input_state(t + lookahead)`; `lookahead_ms` (default 100 ms) compensates human
  reaction time so the model learns the *response to frame t*. Tap events (bomb, LoL keys) are edge-detected in the
  window `[t + lookahead, t + lookahead + tick)`.

### 4.3 Models (all trained from scratch)
- **Cortex encoder**: Conv-stem (s2) → Fused-MBConv (s2) → UIB×2 (s2, k5, ECA) → UIB×2 (s2) → 2× Transformer blocks
  on the 1/16 grid with learned 2-D position embeddings (global context: where are the safe gaps?). CoordConv
  channels. Width presets `nano | small | base`.
- **Heads (basal ganglia)**: attention-pooled global vector (+ aux MLP + intent embedding) → MLP → one linear per
  discrete head. Pointer decoder: FPN-lite upsampling from 1/16 to 1/4 with skip connections (a tiny U-Net — the
  old Hakisa used a U-Net) → score map + offsets; FiLM-conditioned on the action type for LoL.
- **Value head** (for RL) and **danger head** (auxiliary for bullet hells: "will I lose a life within N ticks?";
  label-free because lives are read from the HUD).
- **Intent conditioning**: FiLM on encoder tokens + concat to the global vector; trained with intent dropout so S1
  also works without S2.
- Game nets: `JigokuNet` (keyboard heads, vertical playfield), `BulletHeavenNet` (pointer + buttons), `LoLNet`
  (main view + minimap encoders + game-state vector, autoregressive type→pointer), `DojoNet` (simulator).

### 4.4 Rewards (limbic system)
- HUD **probes**, no manual labelling and no pre-trained vision: `PixelCountProbe` (count icon pixels of a colour in a
  region → lives/bombs), `DigitTemplateProbe` (read the score with 10 digit templates cropped once by the user),
  `TemplateProbe` (detect "game over"/"stage clear" screens), `RegionChangeProbe`.
- `BulletHellReward`: survival bonus per tick, life lost penalty, bomb cost, score delta (scaled, clipped),
  game-over penalty, stage-clear bonus — all YAML-configurable. Shared by Jigoku, Bullet Heaven and the simulator.
- `LoLReward`: from Riot's local Live Client Data API (`https://127.0.0.1:2999/liveclientdata/allgamedata`): Δgold,
  ΔCS, kills, assists, deaths, level-ups, HP change, team objectives (turrets, dragons, baron), win/loss.
  Field access is defensive because the schema is only community-documented.

### 4.5 Training
- **Study (SL/BC)**: multi-head cross-entropy (label smoothing, class weights for rare events like bombs),
  pointer CE (cell + offsets), danger BCE; AdamW + cosine schedule + warm-up, AMP on CUDA, EMA weights,
  augmentations (small shifts with consistent pointer shift, colour jitter, optional horizontal flip with left/right
  label swap). Metrics: per-head accuracy, NLL, pointer distance error, bomb F1.
- **Consolidation (RL)**: PPO (clipped) + GAE + value/reward normalisation + entropy bonus + exact per-head
  KL(π‖π_BC) + decaying BC loss on human demos; critic warm-up iterations with the actor frozen.
  For real games: **asynchronous actor-learner** (actor plays at a fixed Hz with behaviour log-probs; learner updates
  in parallel; PPO's ratio uses behaviour log-probs) or **pause mode** (press the game's pause key during updates).
- **Human takeover (HG-DAgger style)**: a hotkey hands control back to the human during `play`; while the human
  plays, frames + inputs are recorded as corrective demonstrations.

### 4.6 System 2 — prefrontal cortex (Task 3)
- Qwen3-0.6B (text-only, Apache-2.0; hidden size 1024 per its config) — *its language weights are allowed by the rules;
  Hakisa's own from-scratch encoder provides the "eyes"*. Note: Qwen3.5 small models (Mar 2026) ship with native
  vision, so they are **not** used here to respect the "no pre-trained vision" rule.
- Inputs: 16 visual tokens (S1 encoder tokens, pooled 4×4) + 1 global token → MLP projector → LLM embedding space
  (LLaVA-style), plus text: game state from probes/live client, partner instruction (e.g. "save bombs", "go bot"),
  previous plan.
- Output: optional `<think>` rationale, then a constrained line `PLAN: key=value ...` (slots per game, e.g.
  `zone`, `focus`, `bomb` for bullet hells; `objective`, `lane`, `risk`, `recall` for LoL) → parsed → intent ids → S1.
- Training: (1) SL on **hindsight intents** computed automatically from what the player did next (no manual
  labels) + template rationales built from probe readings; LoRA on attention/MLP projections + projector.
  (2) GRPO: sample a group of plans per state, score them by S1 rollouts from cloned simulator states (simulator) or
  by S1's intent-conditioned critic (real games), group-normalised advantages, KL to the SL reference.
- Runtime: S2 runs in its own thread at ~1 Hz; S1 uses the freshest intent (null if stale).
- Memory budget on 4 GB: Qwen3-0.6B bf16/fp16 ≈ 1.2 GB weights + small KV cache; S1 < 50 MB.

### 4.7 League of Legends (Task 4)
- Same recorder + a Live Client poller (2 Hz) stored with the episode.
- `LoLNet`, `LoLReward`, action mapping (quick cast), camera lock assumed, minimap region configurable.
- Duo features: hotkeys → partner instructions for S2 ("follow me", "go top/mid/bot", "retreat", "all-in").
- **Risk notice**: automating inputs in League of Legends violates Riot's Terms of Service; Riot Vanguard may detect
  injected input and the account can be banned. The framework does **not** include any anti-detection/evasion
  measures. Recommended for Practice Tool / custom games against bots only, at the owner's own risk.

### 4.8 The third game: DanmakuDojo (original simulator)
- Vectorised NumPy bullet hell (aimed fans, rings, spirals, rain, walls; fairies + a 3-phase boss; lives, bombs,
  focus, grazing, score). Two control schemes mirroring the real targets: keyboard (Jigoku-like) and mouse
  (Bullet-Heaven-like). Fast render at model resolution for training; "screen" render with a real HUD (lives/bombs
  icons, score digits) to test probes end-to-end; optional pygame window so the owner can play it and test the
  real capture/record/inject stack locally.
- Scripted demonstrator with human-like imperfections (reaction lapses, noise) = stand-in for the owner's demos.

## 5. Verification plan (what "done" means)
1. Unit/integration tests (`pytest`) for every module (actions, storage, dataset alignment, models, heads, probes,
   rewards, simulator determinism/cloning, PPO math, LoL poller with a mock server, S2 with a tiny Qwen3).
2. Simulator experiments (CPU): random vs demonstrator vs BC vs BC+RL (keyboard and mouse schemes): return,
   survival, deaths/min, stage clears, with fixed evaluation seeds. Results + plots in `docs/results/`.
3. S2 experiment (CPU, tiny Qwen3-architecture LLM — **not** the pre-trained 0.6B): S1-only vs S1+S2 vs S1+oracle
   intents.
4. Latency benchmark of each game net (CPU measured; GPU numbers only if measured — never invented).

## 6. Approaches considered
- *Policy representation*: (A) factorised multi-head classifier [chosen — "image classifier" framing, cheap];
  (B) action chunking with flow matching (NitroGen/GR00T) — more expressive but heavier and harder to RL-tune;
  (C) LLM emitting action tokens every frame (Lumine/Game-TARS) — too slow for bullet hells on a 1650 Ti.
- *RL algorithm*: PPO+KL anchor [chosen: stable, simple, works with factorised discrete heads, asynchronous variant
  fits single-instance real-time games]; DQN/DQfD with action branching (sample-efficient but awkward with pointer
  heads and BC-initialised policies); DreamerV3-style world models (most sample-efficient but complex/heavy).
- *Reasoning integration*: dual-system intents [chosen]; LLM-only (too slow); no LLM (loses partner instructions and
  long-horizon planning).

## 7. Sources
- VPT: https://arxiv.org/abs/2206.11795 · NitroGen: https://arxiv.org/abs/2601.02427 , https://nitrogen.minedojo.org/
- Counter-Strike BC: https://arxiv.org/abs/2104.04258
- Helix: https://www.figure.ai/news/helix · GR00T N1: https://arxiv.org/abs/2503.14734 · Hi Robot: https://arxiv.org/abs/2502.19417
- Lumine: https://arxiv.org/abs/2511.08892 · Game-TARS: https://arxiv.org/abs/2510.23691
- Impoola: https://arxiv.org/abs/2503.05546 · MobileNetV4: https://arxiv.org/abs/2404.10518
- PIRLNav: https://arxiv.org/abs/2301.07302 · AlphaStar: https://www.nature.com/articles/s41586-019-1724-z
- DAPG: https://arxiv.org/abs/1709.10087 · PPO: https://arxiv.org/abs/1707.06347 · GAE: https://arxiv.org/abs/1506.02438
- PPO implementation details: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300 · LoRA: https://arxiv.org/abs/2106.09685 · LLaVA: https://arxiv.org/abs/2304.08485
- FiLM: https://arxiv.org/abs/1709.07871 · CoordConv: https://arxiv.org/abs/1807.03247 · ECA-Net: https://arxiv.org/abs/1910.03151
- HG-DAgger: https://arxiv.org/abs/1810.02890
- Qwen3 small models / Qwen3.5 small (Mar 2026): https://artificialanalysis.ai/articles/qwen3-5-small-models
- LoL Live Client Data (community docs): https://dev.overwolf.com/ow-native/live-game-data-gep/supported-games/league-of-legends ,
  https://docs.rs/crate/lol-game-client-api/latest/source/src/model.rs
- Riot third-party applications stance: https://support-leagueoflegends.riotgames.com/hc/en-us/articles/225266848
- DXcam: https://github.com/ra1nty/DXcam

## 8. Experiment notes (added during implementation)

Two pitfalls surfaced by the first DanmakuDojo run, both relevant to the real games:

1. **Label smoothing on rare edge-triggered commands is harmful.** With smoothing 0.05 the bomb head learned
   a ~2.5 % bomb probability *at every tick* — at 20 Hz a bomb every ~2 s; the BC agent used all of its bombs
   (5.4 per episode vs. 1.7 for the demonstrator). Fix: smoothing only on "hold" heads, never on "tap" heads.
   For the same reason inverse-frequency class weights are off by default (they miscalibrate sampling).
2. **Behaviour cloning alone suffers compounding errors.** Validation accuracy was reasonable (move 58 %,
   pointer within 2 cells 68 %), but the clone survived only ~30 s vs. ~89 s for the demonstrator: the
   demonstrations never show recovery from the clone's own mistakes. Fix: a **"play & teach" stage** —
   DAgger in the simulator (the expert labels the states the learner visits) and, on real games, the F8
   human takeover (HG-DAgger), before RL consolidation.
