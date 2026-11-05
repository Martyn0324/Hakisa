# dojo_mouse — difficulty 2.0

Evaluation: 32 full episodes per agent, fixed seeds, ±95% CI. Survival is in seconds (20 ticks/s, max 90 s).
RL was stopped early at 286720 environment steps (iteration 140) because of the CPU budget; the last saved checkpoint is evaluated.

| Agent | Return | Deaths / episode | Survival (s) | Score | Stage clear | Bombs |
|---|---|---|---|---|---|---|
| Random | -1.13 ± 0.29 | 3.00 ± 0.00 | 13.3 | 5834 | 0% | 1.22 |
| Demonstrator (scripted 'human') | 18.85 ± 0.72 | 0.69 ± 0.33 | 88.8 | 19547 | 3% | 0.50 |
| Study / BC (sampled) | -0.80 ± 0.18 | 3.00 ± 0.00 | 14.6 | 2965 | 0% | 0.03 |
| Study / BC (greedy) | -1.25 ± 0.13 | 3.00 ± 0.00 | 13.1 | 1203 | 0% | 0.00 |
| + Play & teach (DAgger round 1, sampled) | -0.51 ± 0.17 | 3.00 ± 0.00 | 15.7 | 3559 | 0% | 0.03 |
| + Play & teach (DAgger round 1, greedy) | -1.08 ± 0.11 | 3.00 ± 0.00 | 13.8 | 1587 | 0% | 0.00 |
| + Play & teach (DAgger round 2, sampled) | -0.46 ± 0.27 | 3.00 ± 0.00 | 15.8 | 4057 | 0% | 0.06 |
| + Play & teach (DAgger round 2, greedy) | -0.74 ± 0.12 | 3.00 ± 0.00 | 15.1 | 2480 | 0% | 0.00 |
| BC + Consolidation / RL (sampled) | 5.64 ± 0.46 | 3.00 ± 0.00 | 41.4 | 31310 | 0% | 5.91 |
| BC + Consolidation / RL (greedy) | 3.74 ± 0.34 | 3.00 ± 0.00 | 32.8 | 28822 | 0% | 5.72 |
| BC + Consolidation / RL (temperature 0.5) | 5.60 ± 0.41 | 3.00 ± 0.00 | 41.2 | 31197 | 0% | 5.88 |
