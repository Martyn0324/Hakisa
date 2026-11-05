# DanmakuDojo (the third game: Hakisa's own simulator)

An original, vectorised NumPy bullet hell written for Hakisa so that every part of the pipeline can be
run and measured anywhere (the real games need Windows + Steam). Two control schemes mirror the real
targets: `dojo_keyboard` (like Jigoku Kisetsukan) and `dojo_mouse` (like Bullet Heaven 2).

* fairies in waves (aimed fans, rings, rain), a 3-phase boss (spirals, rings, walls with gaps, rain)
* lives, bombs (clear the screen + invulnerability), focus/slow, grazing, score
* `render_screen()` draws a real HUD (lives/bombs icons, score digits) that Hakisa's probes read
* deterministic and clonable with common random numbers (used by System 2's GRPO)
* a scripted demonstrator with human-like lapses stands in for your own play

```bash
hakisa dojo-demos --scheme keyboard --steps 100000 --out data/dojo_keyboard
hakisa train-bc   --game dojo_keyboard --data data/dojo_keyboard --out runs/dojo_keyboard/bc
hakisa dojo-eval  --scheme keyboard --ckpt runs/dojo_keyboard/bc/best.pt
hakisa dojo-play  --scheme keyboard --record data/my_dojo     # play it yourself (pip install pygame)
python scripts/dojo_experiments.py --scheme keyboard           # the full BC -> RL experiment
```

`DojoVirtualWindow` exposes the simulator as a fake *game window* (it receives Hakisa's key/mouse
events and returns screenshots), which is how the real-time stack is tested end-to-end.
