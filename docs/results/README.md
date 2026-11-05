# Results

Produced by the scripts in `scripts/` inside a 4-core CPU cloud container (no GPU). Each folder holds the
raw `results.json` (every number, including standard errors and per-iteration training curves), a
`README.md` table and, where available, `learning_curves.png`.

| Folder | What | Script |
|---|---|---|
| [`dojo_keyboard/`](dojo_keyboard/) | DanmakuDojo, keyboard scheme (Jigoku-like), JigokuNet nano | `scripts/dojo_experiments.py --scheme keyboard` |
| [`dojo_mouse/`](dojo_mouse/) | DanmakuDojo, mouse scheme (Bullet-Heaven-like), BulletHeavenNet nano | `scripts/dojo_experiments.py --scheme mouse` |
| [`system2/`](system2/) | System 1 + System 2 (tiny Qwen3-architecture stand-in, not the pre-trained 0.6B) | `scripts/s2_experiment.py` |
| [`latency_cpu.jsonl`](latency_cpu.jsonl) | decision latency of every game network (CPU) | `scripts/benchmark_latency.py` |

What these results are *not*: they are not measurements on Jigoku Kisetsukan, Bullet Heaven 2 or League of
Legends (those need your Windows machine), and the System 2 numbers do not use the pre-trained Qwen3-0.6B
(the Hugging Face Hub was blocked in the container; use the Colab notebook for that).
