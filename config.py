"""
config.py — centralised, environment-overridable configuration for SuperCoach-VIA.
================================================================================

This module is the single source of truth for the "safe envelope" of project
configuration: filesystem paths, data/output directories, the active season
year, and the AFL home-and-away season length. It exists so these values stop
being copy-pasted (and silently diverging) across 20+ scripts.

What lives here
---------------
- REPO_ROOT and every derived data/docs/charts directory path.
- SEASON_YEAR — the active season the season-specific docs target.
- HOME_AND_AWAY_GAMES — AFL home-and-away season length.
- ACTIVE_SINCE — cutoff string used by refresh_data to decide who to re-scrape.

What deliberately does NOT live here
------------------------------------
Model hyperparameters, random seeds (random_state=42), the ranking-formula
constants (Z_CAP, RANK_GAMMA, Z_BLEND, ERAS, WEIGHTS, ...) and the CLI argument
defaults for backtest / prediction are intentionally left in their owning
modules. Those are Scientist-owned: changing them changes derived `[data]`
numbers, and they must move only with Scientist sign-off. This module touches
only the presentation/orchestration envelope.

Zero-dependency .env support
----------------------------
This module reads an optional `.env` file at the repo root with a small built-in
parser (no python-dotenv dependency). Precedence, lowest to highest:

    1. Built-in defaults defined in this file.
    2. Values from `.env` at the repo root (if present).
    3. Real process environment variables (os.environ) — these win.

So a CI job or operator can export SUPERCOACH_SEASON_YEAR=2027 and override the
default without editing any file, and a local developer can drop the same key in
`.env`. See `.env.example` for the full list of supported keys.

Usage
-----
    import config
    df = pd.read_csv(os.path.join(config.PLAYER_DATA_DIR, fname))
    print(config.SEASON_YEAR)             # -> 2026 (or the override)
    path = config.season_doc("afl-finals")  # -> .../docs/afl-finals-2026.md
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Minimal .env parser (zero dependencies)
# ---------------------------------------------------------------------------
# We resolve REPO_ROOT first from this file's own location so that an .env at
# the repo root can be found before we read any overrides from it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_env_file(path: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. No external dependency.

    Rules (deliberately minimal):
      - Blank lines and lines beginning with '#' are ignored.
      - A leading 'export ' prefix is stripped, so `export FOO=bar` works.
      - Everything after the first '=' is the value; surrounding single or
        double quotes are stripped. No variable interpolation, no multiline.
      - Inline comments are NOT stripped (a '#' inside a value is literal),
        which keeps paths containing '#' intact.
    """
    values: dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                if key:
                    values[key] = val
    except OSError:
        # A malformed/unreadable .env must never crash a pipeline run; defaults
        # and the real environment still apply.
        return {}
    return values


# Resolve REPO_ROOT before reading .env: env var wins, else this file's dir.
REPO_ROOT = os.environ.get("SUPERCOACH_REPO_ROOT") or _THIS_DIR
_ENV_FILE = _parse_env_file(os.path.join(REPO_ROOT, ".env"))


def _get(key: str, default: str) -> str:
    """Resolve a string setting: os.environ > .env file > default."""
    if key in os.environ and os.environ[key] != "":
        return os.environ[key]
    if key in _ENV_FILE and _ENV_FILE[key] != "":
        return _ENV_FILE[key]
    return default


def _get_int(key: str, default: int) -> int:
    raw = _get(key, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# If .env supplied a REPO_ROOT but the environment did not, honour it now and
# re-anchor everything below to that root.
if "SUPERCOACH_REPO_ROOT" not in os.environ and _ENV_FILE.get("SUPERCOACH_REPO_ROOT"):
    REPO_ROOT = _ENV_FILE["SUPERCOACH_REPO_ROOT"]

# ---------------------------------------------------------------------------
# Season configuration
# ---------------------------------------------------------------------------
# The active season that the season-specific docs (afl-*-<year>.md) target.
# Most analysis scripts AUTO-DETECT the current season from the freshest row in
# the data and should keep doing so; SEASON_YEAR is for the places that name a
# fixed season in an output filename or a section title.
SEASON_YEAR: int = _get_int("SUPERCOACH_SEASON_YEAR", 2026)

# AFL home-and-away season length (each team plays this many games in total).
HOME_AND_AWAY_GAMES: int = _get_int("SUPERCOACH_HOME_AND_AWAY_GAMES", 22)

# Players with any game on/after this date are refresh candidates (delta scrape).
ACTIVE_SINCE: str = _get("SUPERCOACH_ACTIVE_SINCE", "2024-01-01")

# ---------------------------------------------------------------------------
# Filesystem layout — all derived from REPO_ROOT
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(REPO_ROOT, "data")
PLAYER_DATA_DIR = os.path.join(DATA_DIR, "player_data")
MATCHES_DIR = os.path.join(DATA_DIR, "matches")
LINEUPS_DIR = os.path.join(DATA_DIR, "lineups")
LIVE_SNAPSHOTS_DIR = os.path.join(DATA_DIR, "live_snapshots")
PREDICTION_DIR = os.path.join(DATA_DIR, "prediction")
BACKTEST_DIR = os.path.join(PREDICTION_DIR, "backtest")
TOP100_DIR = os.path.join(DATA_DIR, "top100")

DOCS_DIR = os.path.join(REPO_ROOT, "docs")
NEWS_DIR = os.path.join(DOCS_DIR, "news")
CHARTS_DIR = os.path.join(REPO_ROOT, "assets", "charts")

# All-time top-100 outputs (two historical locations, both still read/written).
TOP100_CSV = os.path.join(REPO_ROOT, "all_time_top_100.csv")
TOP100_SCORES_CSV = os.path.join(TOP100_DIR, "all_time_top_100.csv")


def season_doc(stem: str, year: int | None = None) -> str:
    """Absolute path to a season-specific doc, e.g. season_doc('afl-finals').

    Returns <REPO_ROOT>/docs/<stem>-<year>.md, defaulting year to SEASON_YEAR.
    Centralising this means a season rollover changes one constant, not 6 paths.
    """
    y = SEASON_YEAR if year is None else year
    return os.path.join(DOCS_DIR, f"{stem}-{y}.md")


def player_perf_glob(name_fragment: str = "*") -> str:
    """Glob pattern for player performance CSVs under PLAYER_DATA_DIR."""
    return os.path.join(PLAYER_DATA_DIR, f"{name_fragment}_performance_details.csv")
