# Hakisa 2.0

**An AI duo partner that sees the screen and presses the keys** — trained on *your* gameplay, fast
enough for bullet hells, small enough for a GTX 1650 Ti, with an optional Qwen3 "System 2" that
thinks while the reflexes play.

Hakisa started (2021–2023) as an alternative to SerpentAI: a model looking at screenshots and choosing
inputs, trained first by imitating the owner (Study) and then by reinforcement learning
(Consolidation). It stalled on two walls: **labelling** (every frame needed manual action/reward
labels) and **speed** (200×200 convolution kernels on 1080p frames — seconds per decision).
Hakisa 2.0 removes both walls and adds what the field learned since. The original notes are kept in
[`docs/hakisa1_README.md`](docs/hakisa1_README.md) and the original code in [`legacy/`](legacy/).

| | Hakisa 1.0 | Hakisa 2.0 |
|---|---|---|
| Labels | manual (actions *and* rewards per frame) | **automatic**: inputs recorded with every frame; rewards read from the HUD by calibrate-once probes |
| Recorder | blocked until a key was pressed, kept only that key | samples the full held-input state at 20 Hz + raw events (relabel any time) |
| Vision | 200×200 kernels, 44 M-parameter flatten bottleneck | from-scratch depthwise/attention encoder, **0.2–0.7 M parameters, 4–11 ms per decision on CPU** |
| Mouse | X layer + Y layer (independent) | **X then Y\|X** from a convolutional target map (exact, ~1 k parameters) |
| RL | DQN / PPO2 from scratch | PPO fine-tuning **anchored to the studied policy** (KL), critic warm-up, demo replay, async actor-learner |
| Reasoning | — | **System 2**: Qwen3 reads Hakisa's own visual tokens + game state + your instructions and steers the reflex policy |
| Games | Bullet Heaven, Street Fighter (gym-retro) | Jigoku Kisetsukan, Bullet Heaven 2, League of Legends, DanmakuDojo (simulator) |

> **Rules followed:** no pre-trained vision model anywhere (the encoder is trained from scratch);
> the only pre-trained network is the text LLM (Qwen3-0.6B) used by System 2.

## How it works

```
 screen ─► thalamus (capture, crop "effective window", resize, stack 3 frames)
            │
            ▼
         cortex (from-scratch encoder: Fused-MBConv → UIB+ECA → attention on the 1/16 grid)
            │                                   ▲ FiLM(intent)
            ├─► basal ganglia: one classifier head per input (move / focus / shoot / bomb / action type)
            ├─► superior colliculus: target map for the mouse (X, then Y|X)
            ├─► critic (value) and amygdala (danger: "life lost within 1 s?")
            │
 keys/mouse ◄─ thalamus (diff-based SendInput with scan codes; F12 releases everything)

 System 2 (prefrontal, ~1 Hz, async): visual tokens + "lives 2, bombs 1, danger high" + partner text
           → Qwen3 → "<think>…</think> PLAN: zone=bottom_left focus=on bomb=hold" → intent → FiLM above
```

Training ladder (the same for every game):

1. **Record** your own play (`hakisa record`) — frames + inputs + HUD values, no manual labels.
2. **Study** (`hakisa train-bc`) — behaviour cloning: each frame is *classified* into what you did.
   Two lessons from the simulator are built in: rare "tap" commands (bombs) are never label-smoothed, and
   the *copycat* problem (held keys make "repeat the visible motion" a shortcut) is countered with
   history dropout and keyframe oversampling.
3. **Play & teach** (`hakisa play`) — F8 takes over; your corrections are recorded exactly where Hakisa
   fails (HG-DAgger) and go into the next Study run. In the simulator this is DAgger.
4. **Consolidation** (`hakisa train-rl`) — PPO with a (decaying) KL anchor to the studied policy, critic
   warm-up and demo replay; rewards from the HUD.
5. **System 2** (`scripts/s2_experiment.py` / Colab notebook) — Qwen3 planner trained on hindsight plans, then GRPO.

Design spec with the 2025–2026 research it builds on:
[`docs/superpowers/specs/2026-11-05-hakisa-revival-design.md`](docs/superpowers/specs/2026-11-05-hakisa-revival-design.md).
Why the modules have brain names: [`docs/neuro_notes.md`](docs/neuro_notes.md).

## Install

```bash
git clone https://github.com/Martyn0324/Hakisa && cd Hakisa
pip install -e .[windows,cv,viz]        # Windows + real games (dxcam, pynput, mss, OpenCV)
pip install -e .[llm]                   # System 2 (transformers)
pip install -e .[dev]                   # tests
```

Install PyTorch for your GPU first (https://pytorch.org/get-started/locally/). Python ≥ 3.10.

## Quick start

```bash
hakisa games -v                                    # the configured games and their action spaces
python -m pytest                                   # the test-suite (CPU, no game needed)
python scripts/benchmark_latency.py --device cuda --half   # decision latency on YOUR GPU
python scripts/dojo_experiments.py --scheme keyboard       # full BC -> RL experiment on the simulator
```

Per-game guides: [Jigoku Kisetsukan](docs/games/jigoku.md) ·
[Bullet Heaven 2](docs/games/bullet_heaven2.md) · [League of Legends](docs/games/lol.md) ·
[DanmakuDojo](docs/games/dojo.md). Colab (GPU) notebook: [`notebooks/hakisa_colab.ipynb`](notebooks/hakisa_colab.ipynb).

## Results

All numbers below were produced in a 4-core CPU cloud container (no GPU) by the scripts in
`scripts/`; nothing was run on the real games (they need Windows/Steam/Riot clients), and the
Qwen3-0.6B weights could not be downloaded there. See [`docs/results/`](docs/results/).

RESULTS_PLACEHOLDER

## Safety

* **F12** kill switch releases every key/button and stops; **F9** pause; **F8** human takeover; the
  controller also releases everything on exit or crash.
* League of Legends: automating inputs violates Riot's Terms of Service and Vanguard may detect it.
  No evasion features exist in this repository. Practice Tool / custom games only, at your own risk.

## Repository map

```
hakisa/            the package (see module docstrings)
  games/*.yaml     everything game-specific: bindings, HUD probes, rewards, model, hyper-parameters
scripts/           benchmark, simulator experiments, System 2 experiment
notebooks/         Colab notebook (GPU training, real Qwen3-0.6B)
tests/             pytest suite (runs on CPU, no game)
docs/              design spec, game guides, results, Hakisa 1.0 notes
legacy/            Hakisa 1.0 code and notebooks
```

## References

Behaviour cloning on gameplay: VPT (Baker et al., 2022) · Counter-Strike BC (Pearce & Zhu, 2022) ·
NitroGen (NVIDIA, 2026). RL from a supervised prior: AlphaStar (Vinyals et al., 2019) ·
PIRLNav (Ramrakhya et al., 2023) · DAPG (Rajeswaran et al., 2018) · PPO / GAE (Schulman et al.) ·
"The 37 implementation details of PPO" (Huang et al., 2022). Dual-system agents: Helix (Figure, 2025) ·
GR00T N1 (NVIDIA, 2025) · Hi Robot (Physical Intelligence, 2025) · Lumine (2025) · Game-TARS (2025).
Architecture: MobileNetV4 (2024) · EfficientNetV2 (2021) · ECA-Net (2020) · Impoola (2025) ·
CoordConv (2018) · FiLM (2018). LLM: Qwen3 · LoRA · GRPO (DeepSeekMath, 2024) · LLaVA.
Interactive imitation: HG-DAgger (Kelly et al., 2019). Full links in the design spec.
