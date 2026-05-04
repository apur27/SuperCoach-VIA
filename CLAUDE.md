## Data verification rule — AFL stats

**This repo contains the complete AFL match and player statistics history (1897–present).**

Before writing any player stat into a document — games played, goals, Brownlow votes, premierships, career averages — you MUST verify it against the actual data files in this repo. Do NOT rely on training-data memory or general knowledge for specific numbers.

### Where to look

| Stat type | Location |
|-----------|----------|
| Per-player per-game stats | `data/player_data/<surname>_<firstname>_<DDMMYYYY>_performance_details.csv` |
| Match results and team scores | `data/matches/matches_<year>.csv` |
| All-time aggregated rankings | `all_time_top_100.csv` and `data/top100/all_time_top_100.csv` |

### How to verify a specific player

```python
import pandas as pd, glob, os

VENV_PYTHON = "/home/abhi/sourceCode/python/coding/.venv/bin/python"
PLAYER_DIR = "data/player_data"

# Find a player's performance file (partial name match — files are named surname_firstname_DDMMYYYY_performance_details.csv)
files = glob.glob(f"{PLAYER_DIR}/*newman*performance*.csv")
if files:
    df = pd.read_csv(files[0])
    print(f"Games: {len(df)}")
    print(f"Goals: {df['goals'].sum()}")
    print(f"Disposals: {df['disposals'].sum()}")
    print(f"Years: {df['year'].min()}–{df['year'].max()}")
```

Run with: `/home/abhi/sourceCode/python/coding/.venv/bin/python`

### Non-negotiable

- Never write a specific games total, goals tally, or Brownlow count without first reading it from the data.
- If the player's data file is missing or the stat is genuinely unavailable (pre-1965 incomplete records), say so explicitly and tag the claim `**[historical record — unverified in data]**` rather than inventing a number.
