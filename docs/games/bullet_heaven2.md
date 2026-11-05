# Bullet Heaven 2 (mouse)

Config: [`hakisa/games/bullet_heaven2.yaml`](../../hakisa/games/bullet_heaven2.yaml) · Network: **BulletHeavenNet**
(cursor-position input channel + from-scratch encoder + **superior-colliculus pointer decoder** +
fire/bomb/slow heads + danger head).

Controls: the ship follows the **mouse**; **left button** held = shoot (auto-shoot enabled);
**right click** = bomb; **Shift** = slow movement.

## The pointer (Hakisa 1.0's X/Y idea, made exact)

Hakisa 1.0 picked the mouse X with one layer and Y with another, independently. Two independent heads
can mix the coordinates of two different targets. BulletHeavenNet computes a score map over the
playfield (a U-Net-lite decoder back to 1/4 resolution: 36×48 cells × 4×4 sub-positions = 144×192
targets inside the effective window) and samples **X from P(x)**, then **Y from P(y | x)** — your
two-step decomposition, but exact, with ~1 k parameters instead of `Linear(256 → 1920)`.

## Workflow

Identical to [Jigoku Kisetsukan](jigoku.md) with `--game bullet_heaven2`:

```bash
hakisa calibrate --game bullet_heaven2 --digits
hakisa record    --game bullet_heaven2 --out data/bh2
hakisa train-bc  --game bullet_heaven2 --data data/bh2 --out runs/bh2/bc
hakisa play      --game bullet_heaven2 --ckpt runs/bh2/bc/best.pt --takeover-dir data/bh2_takeover
hakisa train-rl  --game bullet_heaven2 --init runs/bh2/bc/best.pt --demos data/bh2 --out runs/bh2/rl
```

Notes:
* Calibrate `obs.crop` to the playfield; the pointer is expressed relative to it.
* The Steam version and the Flash version (Ruffle) work the same as long as the window title matches
  `capture.window_title`.
* Mouse moves are absolute (`SendInput` with `MOUSEEVENTF_ABSOLUTE`); Hakisa makes the process
  DPI-aware so scaled displays work.
