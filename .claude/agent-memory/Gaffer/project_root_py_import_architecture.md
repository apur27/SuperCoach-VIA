---
name: root-py-import-architecture
description: Cross-import graph of root Python scripts in SuperCoach-VIA — minimal and acyclic; config.py resolved via cwd=root; root charts/ is orphaned.
metadata:
  type: project
---

The root `.py` cross-import graph in SuperCoach-VIA is **minimal and acyclic** (verified by exhaustive grep 2026-05-31). Only these inter-root imports exist:
- `backtest.py` -> `prediction` (`from prediction import AFLDisposalPredictor, extract_round_number`)
- `main.py` -> `game_scraper`, `player_scraper`
- `refresh_data.py` -> `game_scraper`, `player_scraper`
- `refresh_readme.py` -> `update_team_analysis`, `generate_readme_charts`
- `update_team_analysis.py` -> `generate_readme_charts`

Everything else is a leaf. **Confirmed imported by NOBODY:** `prediction_cpu`, `top_players_comprehensive`, `era_based_statistical_analysis`, `prediction_accuracy`, `analysis`, `charts`, `bar_chart`, `helper_functions`, `testGPU`, `cuDF_test`, `gpu_disposal_prediction_old`, `top_players_comprehensive_v1_backup` (last two are the dead `_old`/`_backup` files).

**config.py resolution:** consumers do a plain top-level `import config` with NO sys.path bootstrap — they rely on cwd=repo-root. Subdir scripts (scripts/, scratch/, docs/.../*.py) DO add `sys.path.insert(0, os.path.join(dirname, ".."))` before `import config`. So any root script MOVED into a subdir must gain that same bootstrap or it breaks.

**Why:** A restructuring plan was requested; the feared tangled graph does not exist, so module moves are safe as long as (1) the config bootstrap is added and (2) the 5 inter-root import statements above are updated to package-qualified paths when moved together.

**How to apply:**
- Dead/untracked safe to delete/archive: `gpu_disposal_prediction_old.py`, `top_players_comprehensive_v1_backup.py` (git-tracked, dead), plus `cuDF_test.py`/`testGPU.py`/`helper_functions.py` (tracked, unimported).
- `analysis.py` (tracked, unimported) is the ONLY producer of the 7 loose root heatmap PNGs (`team_performance_heatmap*.png`, `team_performance_comparison.png`) via bare `plt.savefig('...')` with no config path — these PNGs are **untracked** and orphaned.
- Root `charts/` dir (49 PNGs) is **untracked and orphaned**: `config.CHARTS_DIR = assets/charts/`, all docs reference `../assets/charts/`, and live generators (`update_team_analysis`, `generate_readme_charts`, `refresh_readme`) write to `assets/charts/`. Nothing live reads root `charts/`. Safe to delete.
- See [[parallel-council-commits]] for git-race caution when Scientist commits the restructure.
