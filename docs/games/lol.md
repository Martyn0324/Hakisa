# League of Legends (ARAM / normal games, duo)

> **Risk notice.** Automating inputs in League of Legends violates Riot Games' Terms of Service and
> Riot Vanguard may detect injected input; accounts can be banned. Hakisa contains **no** anti-cheat
> evasion. Use it in the Practice Tool or custom games against bots, at your own risk.

Config: [`hakisa/games/lol.yaml`](../../hakisa/games/lol.yaml) · Network: **LoLNet**.

## What Hakisa sees

* **Main view**: the full 1920×1080 frame resized to 256×144 (2 stacked frames).
* **Minimap**: a 64×64 crop (calibrate `obs.minimap.region`; depends on your minimap scale).
* **Game state**: 32 numbers from Riot's local **Live Client Data API**
  (`https://127.0.0.1:2999/liveclientdata/allgamedata`): time, level, gold, HP/mana, K/D/A, CS,
  ward score, death timer, team kills, turrets/dragons/barons/inhibitors, items, ability ranks...

## What Hakisa does

One command per tick (10 Hz), chosen in two autoregressive steps like AlphaStar:

1. **action type**: noop · move (right-click) · minimap move · attack-move (A) · stop (S) ·
   **Q W E R** · **D F** · **items 1–9** · recall (B) · level up Q/W/E/R (Ctrl+key)
2. **target**, conditioned on the action type: a pointer over the main view (144×256 positions,
   ≈7.5 px at 1080p) or over the minimap.

A skill = move the cursor to the target, then tap the key (**Quick Cast**). Required settings:
Quick Cast for abilities/items, **Attack Move on Cursor**, camera **locked**, 1920×1080.

## Reward (OpenAI-Five-style shaping, all in `reward:`)

net-worth gain · CS · kills · assists · deaths · level-ups · HP lost/healed · turrets taken/lost ·
dragons · barons · inhibitors · win/loss. Every term is a difference between consecutive snapshots.

## Duo features

During `hakisa play --game lol`, **F1–F6** send instructions to System 2 ("follow me", "go top",
"go mid", "go bot", "retreat now", "all in"); edit `runtime.instruction_keys`.

## Workflow

```bash
hakisa calibrate --game lol           # check the minimap region
hakisa record    --game lol --out data/lol         # play your games; live-client stats are logged at 2 Hz
hakisa train-bc  --game lol --data data/lol --out runs/lol/bc
hakisa play      --game lol --ckpt runs/lol/bc/best.pt --takeover-dir data/lol_takeover
```

RL fine-tuning on real games works (`hakisa train-rl --game lol ...`) but an episode is a whole game
(20–40 min) — expect it to be slow; collect many hours of your own games for BC first.

## Honest status

Everything except the live game itself is tested in the container (action mapping incl. quick cast
and Ctrl+level-ups, label extraction from recorded clicks/keys, LoLNet forward/backward, the live
client against a mock server, the full env step). It has **not** been run against a real League of
Legends client. The Live Client schema is only community-documented; the parser is defensive
(missing fields become 0), but check the first recording's `extras.jsonl`.

Things you will most likely want to adjust: `tick_hz`, `obs.minimap.region`, the action list
(add/remove items or summoner keys), `reward` weights, `intents.slots`.
