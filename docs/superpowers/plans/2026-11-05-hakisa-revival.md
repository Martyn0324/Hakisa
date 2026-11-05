# Hakisa Revival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Execution method for this session: **native** (the owner asked for autonomous work and this session does not
> spawn subagents unless asked). Each task: write tests → implement → run `pytest` → commit.

**Goal:** Rebuild Hakisa as a tested Python package that records human gameplay, trains an efficient from-scratch
visual policy by behaviour cloning, fine-tunes it with RL, plays Jigoku Kisetsukan / Bullet Heaven 2 / League of
Legends through screen capture + input injection, and optionally adds a Qwen3 "System 2" planner.

**Architecture:** Neuro-named modules (thalamus I/O, hippocampus data, cortex encoder, basal-ganglia heads, limbic
rewards, prefrontal LLM). One generic `HakisaPolicy` specialised per game (`JigokuNet`, `BulletHeavenNet`, `LoLNet`).
An original vectorised simulator (DanmakuDojo) provides the testable third game and the experiments.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.4, NumPy, PyYAML, OpenCV (headless ok), pytest; Windows-only extras:
`dxcam`, `pynput`; LLM extras: `transformers` (v5 line), `safetensors`; optional `pygame`.

**Spec:** `docs/superpowers/specs/2026-11-05-hakisa-revival-design.md`

## Global Constraints

- No pre-trained vision weights anywhere; only text/embedding models (Qwen3-0.6B default) may be pre-trained.
- Everything game-specific lives in `hakisa/games/*.yaml`; code never hard-codes HUD coordinates or key bindings.
- Actions are held-input states per tick; tap events (bomb, LoL keys) are edge-triggered.
- Pointer actions are stored as fine-grid integers `<head>_x`, `<head>_y` (the owner's X/Y decomposition).
- All input injection goes through `Controller`, which must release every held key/button on stop/kill (F12).
- No anti-cheat evasion features. LoL docs must carry the ToS/Vanguard risk notice.
- Frames travel as `uint8`; normalisation happens inside models.
- Tests run on Linux/CPU without Windows-only packages (backends are imported lazily).
- Reported numbers come from runs actually executed; GPU latency is never estimated as if measured.

## Review Focus

1. **DPI scaling on Windows** — capture region and mouse coordinates disagree when the display is scaled; expected: process is made DPI-aware before any coordinate is read (`make_dpi_aware()` called by `record`/`play`), unit-tested via region/pointer round-trips.
2. **Agent crash while keys are held** — expected: `Controller.release_all()` runs in `finally` and on kill switch; tested with a backend that raises mid-loop.
3. **Episode boundaries in frame stacks / lookahead** — expected: stacks never mix episodes and lookahead never reads past the end; tested in dataset tests.
4. **Rare events (bombs) drowned by class imbalance** — expected: class weights computed from data; tested that weight > 1 for the rare class.
5. **Live Client API unavailable (loading screen, game not started, schema drift)** — expected: poller returns `None`, reward is 0 and nothing crashes; tested with a mock server returning 404 and partial JSON.

---

### Task 1: Package skeleton, config, action space

**Files:**
- Create: `pyproject.toml`, `hakisa/__init__.py`, `hakisa/config.py`, `hakisa/actions.py`, `hakisa/utils.py`,
  `hakisa/games/{jigoku,bullet_heaven2,lol,dojo_keyboard,dojo_mouse}.yaml`
- Move: `Optimized/`, `Preprocessing/`, `HakisaGym*.ipynb` → `legacy/` (git mv, history kept)
- Test: `tests/test_config.py`, `tests/test_actions.py`

**Interfaces (produces):**
- `load_game_config(name_or_path: str) -> GameConfig` with sections `capture`, `obs`, `actions`, `bindings`,
  `reward`, `probes`, `model`, `training`, `intents`, `record`, `runtime`.
- `ActionSpace.from_config(cfg) ; .heads: list[HeadSpec] ; .label_keys() -> list[str]`
- `HeadSpec(name, kind: "categorical"|"pointer", choices, mode: "hold"|"tap", grid:(gh,gw), sub:(sh,sw),
  source:"obs"|"minimap", active_when: dict|None, conditioned_on: str|None)`
- `ActionSpace.labels_from_inputs(snap_at: InputSnapshot, events: list[InputEvent], region: Region) -> dict[str,int]`
  (inactive heads → `-1`)
- `ActionSpace.to_desired(action: dict[str,int], region: Region, minimap_region: Region|None) -> DesiredInput`
- `DesiredInput(keys: frozenset[str], buttons: frozenset[str], cursor: tuple[int,int]|None, taps: list[Tap])`

- [ ] Tests: `test_load_all_game_configs` (every YAML loads; head names unique), `test_move_label_from_wasd`
  (held {w,a} → `up_left`; {w,s} → `none`), `test_tap_label_edge_only` (held X without press event → 0),
  `test_pointer_roundtrip` (pixel → (x,y) fine → pixel within half a fine cell), `test_lol_priority`
  (Q press + right-click same tick → `q`, pointer from Q event cursor), `test_to_desired_jigoku`.
- [ ] Implement, run `pytest tests/test_config.py tests/test_actions.py`, commit.

### Task 2: Thalamus (capture, control, tracker) + hippocampus (storage, dataset)

**Files:** `hakisa/thalamus/{__init__,keys,capture,control,tracker}.py`, `hakisa/hippocampus/{__init__,storage,dataset}.py`,
`tests/test_thalamus.py`, `tests/test_storage.py`, `tests/test_dataset.py`

**Interfaces (produces):**
- `Region(left, top, width, height)`; `FrameSource.grab() -> np.ndarray[H,W,3] uint8 RGB`; `MssSource`,
  `DxcamSource`, `CallableSource`; `make_dpi_aware()`; `find_window_region(title) -> Region`; `resolve_region(cfg)`.
- `Controller(backend, tap_hold_s=0.03).apply(desired) / .release_all()`; backends `WindowsSendInputBackend`,
  `PynputBackend`, `LoggingBackend`.
- `InputTracker.start()/stop()/snapshot()/drain_events()/feed(event)`; `InputEvent`, `InputSnapshot`.
- `EpisodeWriter(root, meta, chunk_size=512, compress=False).add(frame, step: dict, minimap=None)`,
  `.add_events(list)`, `.add_extra(dict)`, `.close() -> Path`; `Episode(path)` with `.length`, `.frames(idx)`,
  `.minimap(idx)`, `.array(key)`, `.events()`, `.extras()`; `list_episodes(root)`.
- `BCDataset(episodes, space, stack, augment=None, intent_schema=None)` items:
  `{"obs": uint8[C,H,W], "aux": float[A], "labels": {k: long}, "danger": float, "intent": long[S] (opt)}`;
  `ChunkShuffleSampler`; `compute_class_weights(dataset, space) -> dict[str, Tensor]`.

- [ ] Tests: controller diff (press only new keys, release removed keys, taps press+release),
  `release_all` after backend error, tracker key-name normalisation (`Key.shift_r`→`shift`, `'W'`→`w`),
  storage round trip (frames, arrays, events), dataset stacks repeat first frame at episode start and never cross
  episodes, lookahead clamps at episode end, flip augmentation swaps left/right labels and mirrors pointer x,
  class weight for rare bomb class > 1.
- [ ] Implement, run tests, commit.

### Task 3: Cortex encoder, basal-ganglia heads, game nets

**Files:** `hakisa/cortex/{__init__,blocks,encoder}.py`, `hakisa/basal_ganglia/{__init__,heads}.py`,
`hakisa/models/{__init__,policy,games}.py`, `scripts/benchmark_latency.py`, `tests/test_models.py`

**Interfaces (produces):**
- `VisualEncoder(in_channels, input_hw, preset in {"nano","small","base"}, attn_layers, coord=True)` →
  `EncoderOutput(tokens[B,N,C], fmap[B,C,h,w], skips[list], pooled[B,C])`, attrs `out_dim`, `grid_hw`.
- `PolicyOutput(logits: dict[str,Tensor], pointer: dict[str,PointerLogits], value: Tensor|None, danger: Tensor|None)`
- `ActionDistribution(space, out)` with `.sample(deterministic)`, `.log_prob(actions)`, `.entropy()`,
  `.kl(other)`, `.bc_losses(labels, class_weights, label_smoothing) -> dict[str,Tensor]`.
- `HakisaPolicy.forward(obs, aux=None, minimap=None, intent=None, condition=None) -> PolicyOutput`,
  `.act(obs, aux=None, minimap=None, intent=None, deterministic=False) -> (actions, log_prob, value, out)`,
  `.encode(obs, ...) -> EncoderOutput`.
- `build_policy(cfg: GameConfig) -> HakisaPolicy` dispatching `model.arch` ∈ {`jigoku`, `bullet_heaven`, `lol`}.
  Jigoku: joint `move_focus` 18-way head option + shoot/bomb + danger. BulletHeaven: cursor-heatmap input channel,
  pointer decoder with skips. LoL: main + minimap encoders, game-vector MLP, type-conditioned pointer (FiLM).

- [ ] Tests: output shapes per game net; `log_prob` of sampled actions finite; pointer two-stage sampling (x then
  y|x) matches joint distribution (chi-square on 20k samples of a fixed map); KL(p,p)=0; inactive pointer adds 0 to
  log-prob; intent=None equals null-intent output; param count < 5 M for `small`.
- [ ] Implement, run tests and `python scripts/benchmark_latency.py --device cpu`, commit.

### Task 4: Limbic (probes + rewards + LoL client)

**Files:** `hakisa/limbic/{__init__,probes,rewards,lol_client,lol_reward}.py`, `tests/test_limbic.py`, `tests/test_lol_client.py`

**Interfaces (produces):**
- `PixelCountProbe`, `DigitTemplateProbe`, `TemplateProbe`, `RegionChangeProbe` with `.read(screen) -> value|None`;
  `build_probes(cfg_list, base_dir)`; `HudReader(probes, stable_frames=3).read(screen) -> dict`.
- `BulletHellReward(cfg).reset(hud) / __call__(hud) -> (float, info)`; defaults survival 0.01, life −1.0, bomb −0.3,
  score_scale 1e-4, score_clip 1.0, game_over −1.0, stage_clear +2.0.
- `parse_allgamedata(data, me=None) -> LoLStats|None`; `LiveClientPoller(url, interval=0.5)`;
  `LoLReward(weights).reset(stats) / __call__(stats) -> (float, info)`.

- [ ] Tests: pixel probe counts icons; digit probe reads a rendered number; HUD debounce ignores 1-frame flicker;
  reward signs (life lost negative, bomb negative, score positive, clipping); LoL parse of a sample payload; reward on
  kill/death/CS/gold; poller with a local mock HTTP server returning valid JSON, 404 and truncated JSON.
- [ ] Implement, run tests, commit.

### Task 5: DanmakuDojo simulator + scripted expert

**Files:** `hakisa/envs/{__init__,dojo,dojo_render,dojo_expert}.py`, `tests/test_dojo.py`

**Interfaces (produces):**
- `DanmakuDojo(num_envs, scheme, obs_hw=(112,96), stack=3, difficulty=1.0, seed=0, max_steps=1800,
  render_screen=False)`; `.reset(seed=None) -> (obs uint8[N,C,H,W], info)`;
  `.step(actions: dict[str, np.ndarray]) -> (obs, reward[N], terminated[N], truncated[N], info)`;
  `.render_screen(i) -> uint8[448,544,3]`; `.snapshot()/.restore(snap)`; `.subset(idx)`; `.action_space`
  (an `ActionSpace` from `dojo_keyboard.yaml`/`dojo_mouse.yaml`); `info` keys: `lives, bombs, score, life_lost,
  bomb_used, stage_clear, player_xy, boss_hp, danger, episode_return, episode_length`.
- `ScriptedExpert(env, skill=0.85, horizon=10, lapse_prob=0.03, seed=0).act() -> dict[str,np.ndarray]`.

- [ ] Tests: deterministic under seed; snapshot/restore reproduces identical next obs; expert survives longer than
  random over 8 envs × 600 steps; bomb clears bullets; screen HUD read by probes equals internal lives/bombs/score.
- [ ] Implement, run tests, commit.

### Task 6: Study (BC), Consolidation (PPO), runtime env, agent, recorder, CLI

**Files:** `hakisa/study.py`, `hakisa/consolidation.py`, `hakisa/envs/realtime.py`, `hakisa/agent.py`,
`hakisa/record.py`, `hakisa/cli.py`, `tests/test_study.py`, `tests/test_consolidation.py`, `tests/test_realtime.py`

**Interfaces (produces):**
- `StudyConfig` dataclass; `train_bc(policy, train_ds, val_ds, cfg, out_dir, device) -> dict`;
  `evaluate_bc(policy, ds, device) -> dict`.
- `compute_gae(rewards, values, dones, last_value, gamma, lam) -> (adv, ret)`; `PPOConfig`;
  `PPOTrainer(policy, ref_policy, cfg, device, demo_ds=None).train(env, total_steps, eval_fn=None, log_fn=None)`;
  `AsyncActorLearner(trainer, env)` for real-time games.
- `RealtimeGameEnv(cfg, source, controller, hud_reader, reward_fn)`; `Agent(...).run()`;
  `record_session(cfg, out_root, ...)`.
- CLI `hakisa {record,calibrate,train-bc,train-rl,play,dojo-demos,dojo-eval,bench,train-s2}`.

- [ ] Tests: `compute_gae` vs. a hand-computed 3-step example; PPO on a toy 1-step bandit env raises reward;
  BC on a tiny synthetic dataset lowers loss; KL term is 0 at the start of PPO; realtime env with `CallableSource` +
  `LoggingBackend` runs 5 steps and releases keys on close.
- [ ] Implement, run tests, commit.

### Task 7: Simulator experiments (evidence)

**Files:** `scripts/dojo_experiments.py`, `docs/results/dojo/*.json|png|md`

- [ ] Generate expert demos (keyboard + mouse schemes), train BC (`JigokuNet`/`BulletHeavenNet`, nano preset),
  PPO fine-tune with KL anchor + critic warm-up, evaluate random / expert / BC / BC+RL on fixed seeds; write
  `docs/results/dojo/README.md` with the table and plots. Commit.

### Task 8: Prefrontal System 2

**Files:** `hakisa/prefrontal/{__init__,intents,llm,reasoner,train,runtime}.py`, `scripts/s2_experiment.py`,
`tests/test_prefrontal.py`, `notebooks/hakisa_colab.ipynb`

**Interfaces (produces):**
- `IntentSchema(slots)`, `.format_plan(d) -> "PLAN: k=v ..."`, `.parse_plan(text) -> dict|None`, `.encode(d)`;
  `hindsight_intent(game, info_arrays, t, horizon, schema)`; `state_text(game, hud)`; `rationale(game, hud, intent)`.
- `load_llm(spec: str, device, dtype)` where spec is a HF id (e.g. `Qwen/Qwen3-0.6B`) or `tiny-qwen3`;
  `inject_lora(model, r, alpha, targets)`; `CharTokenizer`.
- `Prefrontal(llm, tokenizer, vis_dim, schema)`: `.loss(vis, prompts, targets)`, `.plan(vis, prompt, ...)`,
  `.sequence_logprob(vis, prompts, completions)`.
- `AsyncPlanner(prefrontal, schema, hz=1.0).submit(...) / .latest(max_age_s)`.

- [ ] Tests: plan format/parse round trip + garbage → None; hindsight zone from a trajectory; tiny Qwen3 forward
  with visual embeddings; LoRA only trains adapters; SL loss decreases over 30 steps; GRPO step runs and changes
  adapter weights. Experiment: S1-only vs S1+S2 vs S1+oracle intents (tiny LLM, labelled as such). Commit.

### Task 9: League of Legends framework

**Files:** `hakisa/envs/lol_env.py`, `hakisa/games/lol.yaml` (from Task 1), `docs/games/lol.md`, `tests/test_lol.py`

- [ ] Tests: LoL action → desired input (Q at target = cursor move then tap `q`; level-up = ctrl+key; item 1–9);
  `LoLNet` forward/act with minimap + game vector; env step with mock live client + fake frames; reward on GameEnd.
- [ ] Implement, run tests, commit.

### Task 10: Docs, notebook, final verification

- [ ] README rewrite (history preserved under "Hakisa 1.0"), `docs/games/*.md`, `docs/neuro_notes.md`, Colab
  notebook, `pytest -q` all green, push to `claude/hakisa-revival-mpe21d`.

---

## Status / resume notes (2026-11-05)

Done (all committed on `claude/hakisa-revival-mpe21d`; `pytest` green):
- Tasks 1–6, 8, 9 implemented and tested (core, models, limbic, DanmakuDojo, training/runtime/CLI,
  System 2, LoL). Extras added during implementation: DAgger "play & teach" stage, anti-copycat
  training (history dropout + keyframe oversampling), no smoothing on tap heads, play temperature.
- Simulator experiments (Task 7): attempt 1 archived in `docs/results/dojo_*/results_attempt1.json`
  (diagnosed: bomb spam from label smoothing; copycat BC). Attempt 2 = `scripts/dojo_experiments.py
  --stage bc,dagger,rl` for both schemes; System 2 = `scripts/s2_experiment.py`.

To resume if interrupted:
1. `pip install -e .[dev,llm]`; data are regenerated by `scripts/dojo_experiments.py --stage demos`
   (100k frames per scheme, deterministic seeds) — `data/` and `runs/` are not in git.
2. Re-run the missing stages, e.g. `--stage dagger,rl` (needs `runs/<game>/bc/best.pt`).
3. `python scripts/summarize_results.py` prints the tables for the README "Results" section.

Blocked from the cloud container (owner action needed): GitHub push (403, Claude GitHub App access),
`huggingface.co` (network policy) — the real Qwen3-0.6B run is in `notebooks/hakisa_colab.ipynb`.
