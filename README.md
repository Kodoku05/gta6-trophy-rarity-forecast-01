# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kodoku05/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kodoku05/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Cohort-model forecaster for GTA VI trophy earn rates on Exophase: for each of the game's 55 PS5 trophies it forecasts five quantiles (5/25/50/75/95%) of the earn rate at 2026-11-26T00:00:00Z, from the case file's history of Exophase reads. The model combines backtest-weighted player-growth curves with per-trophy cohort fits (S, lambda), 15% tier shrinkage, and backtest-calibrated uncertainty spreads. Local validation: practice pinball loss 0.7930 vs 1.5084 for both house baselines; final-horizon simulations 0.2353 / 0.1877 / 0.2402, each beating both baselines.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=p2xp6s2q&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
