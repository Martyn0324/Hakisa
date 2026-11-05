"""Print the markdown result tables from docs/results/*/results.json (used for the README)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(name: str, r: dict | None) -> str | None:
    if not r:
        return None
    ci = lambda k: 1.96 * r.get(k + "_se", 0.0)  # noqa: E731
    extra = f" {100 * r['invalid_plan_rate']:.1f}% |" if "invalid_plan_rate" in r else ""
    return (f"| {name} | {r['episode_return']:.2f} ± {ci('episode_return'):.2f} | {r['episode_deaths']:.2f} ± {ci('episode_deaths'):.2f} | "
            f"{r['episode_length'] / 20:.1f} | {r['episode_score']:.0f} | {100 * r.get('episode_clear', 0):.0f}% | {r.get('episode_bombs', 0):.2f} |{extra}")


def dojo_table(path: Path) -> str:
    res = json.loads(path.read_text())
    s = res.get("setup", {})
    lines = [f"**{s.get('game')}** (difficulty {s.get('difficulty')}, {s.get('eval_episodes')} evaluation episodes per agent, ±95% CI)", "",
             "| Agent | Return | Deaths / episode | Survival (s, max 90) | Score | Stage clear | Bombs |",
             "|---|---|---|---|---|---|---|"]
    b = res.get("baselines", {})
    rows = [row("Random inputs", b.get("random")), row("Scripted demonstrator (the 'human')", b.get("demonstrator")),
            row("Study (BC)", res.get("bc", {}).get("play_sampled"))]
    for d in res.get("dagger", []):
        rows.append(row(f"+ Play & teach, DAgger round {d['round'] + 1}", d.get("play_sampled")))
    rows.append(row("+ Consolidation (PPO, KL-anchored)", res.get("rl", {}).get("play_sampled")))
    lines += [r for r in rows if r]
    return "\n".join(lines)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/results")
    for p in sorted(root.glob("dojo_*/results.json")):
        print(dojo_table(p))
        print()
    s2 = root / "system2" / "README.md"
    if s2.exists():
        print(s2.read_text())


if __name__ == "__main__":
    main()
