# dojo_keyboard — difficulty 2.0

Evaluation: 32 full episodes per agent, fixed seeds, ±95% CI. Survival is in seconds (20 ticks/s, max 90 s).
RL was stopped early at 409600 environment steps (iteration 200) because of the CPU budget; the last saved checkpoint is evaluated.

| Agent | Return | Deaths / episode | Survival (s) | Score | Stage clear | Bombs |
|---|---|---|---|---|---|---|
| Random | 0.63 ± 0.50 | 3.00 ± 0.00 | 20.6 | 12289 | 0% | 2.44 |
| Demonstrator (scripted 'human') | 19.34 ± 0.25 | 0.59 ± 0.21 | 88.6 | 24655 | 12% | 1.69 |
| Study / BC (sampled) | -0.14 ± 0.26 | 3.00 ± 0.00 | 17.2 | 4302 | 0% | 0.06 |
| Study / BC (greedy) | 0.24 ± 0.22 | 3.00 ± 0.00 | 18.9 | 4612 | 0% | 0.00 |
| + Play & teach (DAgger round 1, sampled) | 0.09 ± 0.36 | 3.00 ± 0.00 | 18.1 | 5440 | 0% | 0.25 |
| + Play & teach (DAgger round 1, greedy) | 0.51 ± 0.40 | 3.00 ± 0.00 | 20.5 | 4071 | 0% | 0.00 |
| + Play & teach (DAgger round 2, sampled) | 0.91 ± 0.53 | 3.00 ± 0.00 | 21.1 | 8929 | 0% | 0.66 |
| + Play & teach (DAgger round 2, greedy) | 0.56 ± 0.30 | 3.00 ± 0.00 | 20.4 | 4719 | 0% | 0.00 |
| BC + Consolidation / RL (sampled) | 8.69 ± 0.76 | 3.00 ± 0.00 | 55.4 | 33509 | 0% | 5.78 |
| BC + Consolidation / RL (greedy) | 9.68 ± 0.92 | 2.97 ± 0.06 | 60.2 | 33097 | 0% | 5.78 |
| BC + Consolidation / RL (temperature 0.5) | 10.95 ± 0.95 | 2.94 ± 0.09 | 65.8 | 34773 | 0% | 6.00 |
