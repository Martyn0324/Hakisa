# Jigoku Kisetsukan: Sense of the Seasons (keyboard)

Config: [`hakisa/games/jigoku.yaml`](../../hakisa/games/jigoku.yaml) · Network: **JigokuNet**
(from-scratch encoder + foveal attention head + move/focus/shoot/bomb heads + danger head).

Controls Hakisa uses (as you described them): **WASD** move · **Shift** focus · **Z** shoot · **X** bomb.
If your in-game bindings differ, edit `actions.heads` in the YAML (the recorder and the controller
both read it).

## 0. Install (Windows)

```bash
pip install -e .[windows,cv,viz]
```

Use the PyTorch build matching your GPU (https://pytorch.org/get-started/locally/).

## 1. Calibrate (once per resolution)

Run the game windowed and note the window title (default substring `Jigoku`). Then:

```bash
hakisa calibrate --game jigoku --digits
```

You will be asked to click: the **playfield** (the "effective window" — exclude the side HUD),
the **lives** and **bombs** icon areas (plus one pixel of an icon and how many are visible),
the **score** area (type the number you see; repeat with other screenshots until 0–9 templates
exist) and the **game over** text. Paste the printed YAML into `jigoku.yaml` (`obs.crop`, `probes`).
Probes stay disabled until `calibrated: true`.

Optional: set `runtime.reset_macro` to the keys that restart a stage after game over, e.g.
`[{key: z, delay: 1.0}, {key: z, delay: 2.0}]`. Without it, press **F7** whenever a stage starts.

## 2. Record your play (Human Teacher)

```bash
hakisa record --game jigoku --out data/jigoku
```

Play normally. Every tick (20 Hz) the frame, all held keys, the raw key events and the HUD values are
stored; **F7** starts a new episode, **F12** stops. Labels are computed automatically
(`lookahead_ms: 100` = what you pressed 100 ms after seeing the frame). Aim for 1–3 hours to start.
If the game runs as administrator, run the recorder as administrator too (Windows input hooks).

## 3. Study (behaviour cloning)

```bash
hakisa train-bc --game jigoku --data data/jigoku --out runs/jigoku/bc
```

Watch `runs/jigoku/bc/study_log.jsonl`: `val/acc/move`, `val/f1/focus`, `val/nll/total`.
`best.pt` is the checkpoint with the lowest validation NLL (validation is split by episode).

## 4. Play (and teach)

```bash
hakisa play --game jigoku --ckpt runs/jigoku/bc/best.pt --takeover-dir data/jigoku_takeover
```

**F8** = take over (your corrections are recorded into `data/jigoku_takeover`), **F9** = pause,
**F12** = kill switch (releases every key). Add the takeover data to the next `train-bc` run
(`--data data/jigoku data/jigoku_takeover`) — this is the HG-DAgger loop.

## 5. Consolidation (RL fine-tuning on the real game)

Needs calibrated probes (the reward comes from lives/bombs/score).

```bash
hakisa train-rl --game jigoku --init runs/jigoku/bc/best.pt --demos data/jigoku --out runs/jigoku/rl --iterations 200
```

PPO + KL anchor to the BC policy + critic warm-up + demo replay; the game keeps running while the
learner updates (`update_mode: async`). If the game can be paused, set `update_mode: pause` and
`runtime.game_pause_key`.

## Tuning knobs

| What | Where |
|---|---|
| Decision rate | `tick_hz` (20) |
| Model size | `model.preset`: nano / small / base |
| Reaction-time alignment | `actions.lookahead_ms` |
| Reward weights | `reward:` (survival, life_lost, bomb_used, score_scale, ...) |
| BC / RL hyper-parameters | `training.bc`, `training.rl` |
