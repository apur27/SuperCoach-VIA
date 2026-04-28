import glob
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Z-scores are capped at ±Z_CAP before any aggregation.
Z_CAP = 3.0

# Number of best seasons used for the core mean_adj signal.
# Top-10 rewards sustained excellence: long careers get more candidate elite
# seasons but also dilute their average vs short-peak players. Top-8 let
# 2-3-year peaks dominate; top-15 crushed consistent midfielders with weaker tails.
TOP_N_SEASONS = 10

# Curvature on the year_score = ((101 - rank) / 100) ** RANK_GAMMA mapping.
# γ < 1 is concave: compresses the gap between rank #1 and rank #25, allowing
# consistent top-20 finishers to accumulate comparable mean_adj to dominant
# #1 finishers. γ > 0.5 is too convex and locks out consistent mid-peak careers.
#
#   γ=0.27 reference:
#   rank #1  = 1.000   rank #4  = 0.992   rank #10 = 0.974   rank #25 = 0.911
#   rank #50 = 0.851   rank #100 = 0.000
RANK_GAMMA = 0.22

# Blend weight for the per-year z-score dominance signal in year_score.
# year_score = (1 - Z_BLEND) * rank_score + Z_BLEND * z_signal
# where z_signal = (z_adj + Z_CAP) / (2 * Z_CAP) maps z_adj∈[-3,+3] → [0,1].
#
# The rank-based score rewards consistency (top-100 is top-100). The z signal
# rewards within-cohort statistical dominance — true legends like Martin '17
# (z≈+2.88) or Pendlebury '11 (z≈+2.82) score ~0.97-0.98, while merely-good
# top-30 finishers sit at z≈1.2-1.8 → 0.70-0.80. Blending lets era-relative
# dominance separate legends from accumulators, using the data, not rules.
#
# Z_BLEND = 0 → pure rank (old behaviour). Z_BLEND > 0.35 risks losing the
# career-consistency signal. Tuning sweet spot is 0.20-0.30.
Z_BLEND = 0.15

# ERA_COMPLETENESS reflects how much of a player's true contribution is
# captured by the available stats in each era. Under the rank-based all-time
# formula, ec is applied LINEARLY to year_score (0..1). Rank #1 already
# captures era-fair dominance, so ec serves only as an epistemic discount.
#
# Graduated discount: 1965_1990 is the anchor at 0.92. Eras with richer stats
# (post_2010) get a mild recency discount to avoid over-representing modern
# players. pre_1965 gets an epistemic discount for having only 2 stats.
ERA_COMPLETENESS = {
    'pre_1965':  0.84,   # 2 stats — goals+behinds only; raised from 0.80 to lift Lee Dick/Titus/Whitten/Goggin into top-30 (0.85 over-promoted Coventry to #4).
    '1965_1990': 0.92,   # 4 stats — anchor; left unchanged.
    '1990_2010': 0.89,   # graduated recency discount; 2000s cohort was 20/100, nudging down.
    'post_2010': 0.89,   # recency discount targeting ~7 post-2010 players in top-30.
    'unknown':   0.85,
}

# Maximum fraction of the raw score any single stat may contribute.
# Raised from 0.40 → 0.55: the old 40% cap was too aggressive for pure goal
# kickers in 2-4 stat eras where goals dominate.
SINGLE_STAT_CAP = 0.55

# Minimum players in a position group before we fall back to full-cohort z-scores.
MIN_POSITION_GROUP = 5

# Career-completeness discount for still-active players.
# Players whose careers are still in progress have not had their declining
# years averaged in — they are being measured only on their current peak/near-peak
# form. To compare fairly against retired legends whose full careers (including
# tail decline) feed the formula, we multiply active players' all_time_score by
# this constant. Active = appeared in any yearly_top_100 list for a year in
# RECENT_YEARS (auto-detected from data; no player names hardcoded).
RECENT_YEARS = {2025, 2026}
ACTIVE_PLAYER_DISCOUNT = 0.90


# Eras define which stats are available for scoring.
# 'disposals' is intentionally excluded from all eras: disposals = kicks + handballs,
# so weighting both 'kicks' and 'disposals' double-counts every kick. Instead, kicks
# and handballs are weighted separately, giving each proper credit.
# brownlow_votes carries weight=0 and is retained only for potential future use.
ERAS = {
    'pre_1965':  (1897, 1964, ['goals', 'behinds']),
    '1965_1990': (1965, 1990, ['goals', 'behinds', 'kicks', 'handballs']),
    '1990_2010': (1991, 2010, ['goals', 'behinds', 'kicks', 'handballs', 'marks']),
    'post_2010': (2011, 2030, ['goals', 'behinds', 'kicks', 'handballs', 'marks',
                               'tackles', 'one_percenters', 'clearances', 'contested_possessions',
                               'contested_marks', 'goal_assist']),
}

# Flat union of all stats across eras — used for efficient CSV column filtering.
ALL_STATS: List[str] = sorted({s for _, _, stats in ERAS.values() for s in stats})


def initialize_df_lib():
    try:
        import cudf
        cudf.Series([1])
        logging.info("GPU acceleration enabled via CuDF")
        return cudf, True
    except (ImportError, RuntimeError, OSError, AttributeError) as e:
        logging.info(f"GPU unavailable: {e}. Falling back to CPU with pandas.")
        return pd, False

df_lib, USE_GPU = initialize_df_lib()


def get_era(year: int) -> Tuple[str, List[str]]:
    for era_name, (start, end, stats) in ERAS.items():
        if start <= year <= end:
            return era_name, stats
    return 'unknown', []


def safe_sum(df, col: str, df_lib) -> float:
    series = df[col]
    if USE_GPU:
        try:
            series = series.astype('float64')
        except Exception as e:
            logging.error(f"cuDF conversion failed for {col}: {e}")
            series = series.applymap(lambda x: float(x) if x else 0)
        return series.sum()
    else:
        return pd.to_numeric(series, errors='coerce').fillna(0).sum()


def parse_dates(df, df_lib, col='date'):
    if USE_GPU:
        try:
            df[col] = df_lib.to_datetime(df[col])
        except Exception as e:
            logging.warning(f"cuDF date parsing failed: {e}. Using pandas fallback.")
            dates_cpu = pd.to_datetime(df[col].to_pandas(), errors='coerce')
            df[col] = df_lib.Series(dates_cpu)
    else:
        df[col] = df_lib.to_datetime(df[col], errors='coerce')
    return df.dropna(subset=[col])


def process_player_file(
    filepath: str, year: int, weights: dict, df_lib
) -> Optional[Tuple[str, int, Dict[str, int], int]]:
    try:
        if USE_GPU:
            df_dates = df_lib.read_csv(filepath, usecols=['date'])
        else:
            df_dates = df_lib.read_csv(filepath, usecols=['date'], low_memory=False)

        df_dates = parse_dates(df_dates, df_lib, 'date')
        df_dates['year'] = df_dates['date'].dt.year

        if year not in (df_dates['year'].values_host if USE_GPU else df_dates['year'].values):
            return None

        if USE_GPU:
            df = df_lib.read_csv(filepath)
        else:
            df = df_lib.read_csv(filepath, low_memory=False)

        df = parse_dates(df, df_lib, 'date')
        df['year'] = df['date'].dt.year
        df = df[df['year'] == year]

        if df.empty:
            return None

        player_name = "_".join(os.path.basename(filepath).split("_")[:-2])
        _, era_stats = get_era(year)
        available_stats = [stat for stat in era_stats if stat in df.columns]
        if not available_stats:
            return None

        totals: Dict[str, int] = {}
        stat_contributions: Dict[str, float] = {}
        for stat in available_stats:
            total = safe_sum(df, stat, df_lib)
            totals[stat] = int(total)
            stat_contributions[stat] = total * weights.get(stat, 0)

        uncapped_score = sum(stat_contributions.values())

        if uncapped_score > 0:
            excess = sum(
                max(0.0, c - SINGLE_STAT_CAP * uncapped_score)
                for c in stat_contributions.values()
            )
            score = max(0, int(uncapped_score - excess))
        else:
            score = 0

        games_played = len(df)
        return (player_name, score, totals, games_played)

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None


def calculate_percentile_ranks(scores: List[int]) -> np.ndarray:
    return pd.Series(scores).rank(pct=True).to_numpy() * 100


def compute_within_era_z_scores(scores: List[float]) -> np.ndarray:
    """Z-score within a cohort, capped at ±Z_CAP.

    Empty → empty array. Zero variance → all zeros.
    """
    if not scores:
        return np.array([], dtype=float)
    arr = np.asarray(scores, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros_like(arr)
    return np.clip((arr - mu) / sigma, -Z_CAP, Z_CAP)


def _career_position_group(career_gpg: float) -> str:
    """Four-way career-based position classification using career goals/game.

    Groups (thresholds calibrated against actual career g/game data):
      key_forward  >= 3.0 g/game — elite goal machines (≥3 goals per game)
      forward_mid  0.80–2.99 g/game — dominant forwards + forward-midfielders
      midfielder   0.30–0.79 g/game — pure midfielders
      backline     < 0.30 g/game — defenders and rucks who rarely score

    Threshold 3.0 (not 2.5) is the critical design choice: it keeps big-scoring
    forwards (2.6–2.9 g/game) in forward_mid where they dominate their peer group,
    rather than mixing them with pure key forwards (4+ g/game) which would suppress
    their z-scores. Within-group z rewards dominance over genuine peers.
    """
    if career_gpg >= 3.0:
        return 'key_forward'
    if career_gpg >= 0.80:
        return 'forward_mid'
    if career_gpg >= 0.30:
        return 'midfielder'
    return 'backline'


def _compute_position_stratified_z_scores(
    scores: List[float],
    positions: List[str],
) -> np.ndarray:
    """Z-score within position groups; fall back to full-cohort if group is too small."""
    n = len(scores)
    z_out = np.zeros(n, dtype=float)
    full_z = compute_within_era_z_scores(scores)

    groups: Dict[str, List[int]] = {}
    for i, pos in enumerate(positions):
        groups.setdefault(pos, []).append(i)

    for pos, indices in groups.items():
        group_scores = [scores[i] for i in indices]
        if len(group_scores) >= MIN_POSITION_GROUP:
            group_z = compute_within_era_z_scores(group_scores)
            for j, i in enumerate(indices):
                z_out[i] = group_z[j]
        else:
            for i in indices:
                z_out[i] = full_z[i]

    return z_out


def create_ranking_dataframe(players_data: List[Tuple[str, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(players_data, columns=['player', 'score', 'percentile_rank', 'games_played'])


def generate_yearly_top_100(
    data_dir: str,
    year: int,
    weights: dict,
    output_dir: str,
    df_lib,
) -> List[Tuple[str, int, Dict[str, int], int, float, float]]:
    """Generate the top 100 players for a specific year.

    Returns (player, score, totals, games_played, percentile_rank, z_score_adj).
    z_score_adj is already shrunk by sqrt(era_completeness) so it carries an
    era-fair dominance signal directly into the all-time aggregator.
    """
    player_scores = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_performance_details.csv"):
            filepath = os.path.join(data_dir, filename)
            result = process_player_file(filepath, year, weights, df_lib)
            if result:
                player_scores.append(result)

    if not player_scores:
        logging.info(f"No player data found for year {year}")
        return []

    era_name, _ = get_era(year)
    shrinkage = math.sqrt(ERA_COMPLETENESS.get(era_name, 0.40))

    scores = [s for _, s, _, _ in player_scores]
    percentile_ranks = calculate_percentile_ranks(scores)

    # Position-stratified z-scores: forwards vs non-forwards, so prolific
    # goal-kickers are compared to peers in their role, not to midfielders
    # who recorded zero goals.
    positions = [_position_key(totals, games) for _, _, totals, games in player_scores]
    raw_z = _compute_position_stratified_z_scores(scores, positions)

    # Apply sqrt(completeness) shrinkage: a pre-1965 z of +3.0 becomes +1.9,
    # reflecting that only 2 stats were measured (epistemic humility, not punishment).
    z_scores = raw_z * shrinkage

    enriched = list(zip(player_scores, percentile_ranks, z_scores))
    sorted_players = sorted(enriched, key=lambda x: (x[0][1], x[0][3]), reverse=True)
    top_100 = sorted_players[:100]

    df = create_ranking_dataframe([(p[0], p[1], pr, p[3]) for p, pr, _ in top_100])
    df.to_csv(os.path.join(output_dir, 'yearly', f'year_{year}.csv'), index=False)

    return [(p[0], p[1], p[2], p[3], pr, z) for p, pr, z in top_100]


def compile_all_time_top_100(
    yearly_top_100: Dict[int, List[Tuple[str, int, Dict[str, int], int, float, float]]],
    output_dir: str,
    career_games: Optional[Dict[str, int]] = None,
) -> None:
    """Compile the all-time top 100 using a rank-based formula.

    The signal is "how well a player ranked within each year's top 100":
    if you consistently appeared in the top 100, and consistently ranked
    near the top (rank #1 is better than rank #50), you are a great player.

    Formula
    -------
    For each player p:

        For each year y in which p appears in yearly_top_100:
            rank_y       = p's 1-based rank within yearly_top_100[y]
            year_score_y = ((101 - rank_y) / 100) ** RANK_GAMMA   # γ=0.20 (concave)
            ec_y         = ERA_COMPLETENESS[era of y]            # epistemic discount
            adj_y        = year_score_y * ec_y

        Take the TOP_N_SEASONS (=10) largest adj_y values:
            mean_adj     = mean(top_10 adj_y)

        Career bonus (additive, SEASONS-based):
            seasons      = count of years p appears in yearly_top_100
            career_bonus = 0.60 * min(seasons / 18, 1.0)

        all_time_score   = mean_adj * (1.0 + career_bonus)

    Eligibility: career_games >= 150 (true career games from full aggregate,
    not games-while-in-yearly-top-100). Players with fewer games are dropped.

    Why this works
    --------------
    - rank within the year's top 100 is already a transparent dominance signal:
      raw scores are era-specific, but rank #1 in 1995 means the same thing
      semantically as rank #1 in 2023.
    - The career bonus uses SEASONS-IN-TOP-100, not GAMES. Pre-1990 players
      had ~16-18 games/season; modern players play 22-24, so a games-based
      bonus systematically rewarded modern longevity. Seasons-in-top-100 is
      era-neutral: every era ranks 100 players per year, and elite players
      from every decade accumulate 12-19 such seasons.
    - ERA_COMPLETENESS spread compressed (0.88 / 0.92 / 0.92 / 0.92): the
      former 0.95 multiplier on 1990-2010 was injecting structural advantage
      that, combined with the games-based bonus, caused the 2000s decade to
      dominate the top 100 (~25 players). Equalising eliminates that bias.
    - RANK_GAMMA = 0.27 (concave): compresses the gap between top ranks so
      consistent top-20 finishers can compete with multi-#1 peak performers
      without forcing a quota.
    - Career bonus cap at 18 seasons lets longevity-at-elite-level surface
      naturally: the 18-cap differentiates the longest-serving top-100 players
      from peers who peaked in fewer seasons.

    Position stratification, peak bonus, and decade-quota constraints have been
    removed: rank #1 is already captured by year_score=1.0, and the calibration
    above gives natural era diversity without quotas.
    """
    # Auto-detect still-active players: anyone appearing in a yearly top-100
    # list for a RECENT_YEARS season is considered active. Their final
    # all_time_score is multiplied by ACTIVE_PLAYER_DISCOUNT.
    active_players = {
        entry[0]
        for year, entries in yearly_top_100.items()
        if year in RECENT_YEARS
        for entry in entries
    }
    logging.info(
        f"Active-player discount: {len(active_players)} players appeared in "
        f"{sorted(RECENT_YEARS)} → multiplied by {ACTIVE_PLAYER_DISCOUNT}"
    )

    # player -> list of era-adjusted year_scores, one per yearly-top-100 appearance
    player_year_scores: Dict[str, List[float]] = defaultdict(list)
    player_seasons: Dict[str, int] = defaultdict(int)
    # Track each player's best (rank, year) for the debug log.
    player_best_rank: Dict[str, Tuple[int, int]] = {}

    for year, top_list in yearly_top_100.items():
        era_name, _ = get_era(year)
        ec = ERA_COMPLETENESS.get(era_name, ERA_COMPLETENESS['unknown'])

        # Sort by raw score desc to compute rank within the year. This matches
        # the order already written to the yearly CSV (sorted by x[0][1] desc).
        sorted_list = sorted(top_list, key=lambda x: (x[1], x[3]), reverse=True)

        for rank, entry in enumerate(sorted_list, start=1):
            player = entry[0]
            z_adj = entry[5]  # within-cohort z-score, already capped at ±Z_CAP and shrunk
            rank_score = ((101 - rank) / 100.0) ** RANK_GAMMA
            # Map z_adj from [-Z_CAP, +Z_CAP] → [0, 1]. A z of +Z_CAP (rare,
            # truly dominant season) reaches 1.0; z=0 (median of cohort) is 0.5.
            z_signal = (float(z_adj) + Z_CAP) / (2.0 * Z_CAP)
            z_signal = max(0.0, min(1.0, z_signal))
            year_score = (1.0 - Z_BLEND) * rank_score + Z_BLEND * z_signal
            adj = year_score * ec
            player_year_scores[player].append(adj)
            player_seasons[player] += 1
            best = player_best_rank.get(player)
            if best is None or rank < best[0]:
                player_best_rank[player] = (rank, year)

    # Eligibility filter and final score.
    all_time_scores: List[Tuple[str, float, int, int, float]] = []
    for player, adj_scores in player_year_scores.items():
        career_g = career_games.get(player, 0) if career_games else 0
        if career_g < 150:
            continue

        top_n = sorted(adj_scores, reverse=True)[:TOP_N_SEASONS]
        if not top_n:
            continue
        mean_adj = sum(top_n) / len(top_n)
        # Career bonus is SEASONS-based (era-fair) instead of GAMES-based.
        # Pre-1990 players had ~16-18 games/season; modern players play 22-24/season,
        # so a games-based bonus systematically rewarded modern longevity. A "season
        # in the yearly top-100" is era-neutral: every era ranks 100 players per year.
        # Cap at 18 seasons differentiates the longest-serving elite players from
        # those who peaked in fewer seasons.
        seasons = player_seasons[player]
        # Blended: 0.50 for longevity (seasons/18 cap) + 0.20 flat for reaching
        # 8+ top-100 seasons — the flat component stops short elite careers
        # from being crushed vs 18-season careers.
        career_bonus = 0.55 * min(seasons / 18.0, 1.0) + 0.20 * min(seasons / 8.0, 1.0)
        all_time_score = mean_adj * (1.0 + career_bonus)

        # Career-completeness discount for active players (see constant comment).
        if player in active_players:
            all_time_score *= ACTIVE_PLAYER_DISCOUNT

        all_time_scores.append((player, all_time_score, seasons, career_g, mean_adj))

    if not all_time_scores:
        logging.info("No all-time top 100 players found")
        return

    # Sort by score, then by mean_adj as a quality tiebreaker, then by games.
    all_sorted = sorted(
        all_time_scores,
        key=lambda x: (x[1], x[4], x[3]),
        reverse=True,
    )

    final_100 = all_sorted[:100]

    df = pd.DataFrame(
        [(p, s) for p, s, *_ in final_100],
        columns=['player', 'all_time_score'],
    )
    df.to_csv(os.path.join(output_dir, 'all_time_top_100.csv'), index=False)

    # ---- Diagnostic: top-35 with active flags + decade distribution ----
    def _peak_decade(player: str) -> Optional[int]:
        """Birth year + 28 → peak year → decade (e.g., 1979 → 1970)."""
        parts = player.rsplit('_', 1)
        if len(parts) != 2 or len(parts[1]) != 8:
            return None
        try:
            birth_year = int(parts[1][-4:])
        except ValueError:
            return None
        return ((birth_year + 28) // 10) * 10

    logging.info("=" * 78)
    logging.info("TOP 35 (rank | score | seasons | active | peak-decade | player)")
    logging.info("-" * 78)
    pre_1980_in_top30 = 0
    for rank, (p, s, seasons, _g, _mean_adj) in enumerate(final_100[:35], start=1):
        is_active = p in active_players
        decade = _peak_decade(p)
        decade_str = f"{decade}s" if decade is not None else "????"
        if rank <= 30 and decade is not None and decade < 1980:
            pre_1980_in_top30 += 1
        active_flag = "ACTIVE" if is_active else "      "
        logging.info(
            f"  #{rank:>3} | {s:.4f} | {seasons:>2}s | {active_flag} | {decade_str:>5} | {p}"
        )
    logging.info("-" * 78)
    logging.info(f"Pre-1980-peak in top-30: {pre_1980_in_top30} (need >= 10)")

    decade_counts: Dict[int, int] = defaultdict(int)
    for p, *_ in final_100:
        d = _peak_decade(p)
        if d is not None:
            decade_counts[d] += 1
    logging.info("Decade distribution (top-100):")
    for d in sorted(decade_counts):
        logging.info(f"  {d}s: {decade_counts[d]}")
    logging.info("=" * 78)


# ---------------------------------------------------------------------------
# Fast single-pass ingestion
# ---------------------------------------------------------------------------

def _aggregate_one_file(path: str) -> List[Tuple[str, int, Dict[str, float], int]]:
    """Read one player CSV once → [(player_name, year, totals, games_played), ...]."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        logging.warning(f"Could not read {path}: {e}")
        return []

    if df.empty:
        return []

    # Derive year from the 'year' column (preferred) or parse from 'date'
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    elif 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    else:
        return []

    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    if df.empty:
        return []

    available = [s for s in ALL_STATS if s in df.columns]
    if not available:
        return []
    for s in available:
        df[s] = pd.to_numeric(df[s], errors='coerce').fillna(0)

    player_name = "_".join(os.path.basename(path).split("_")[:-2])
    out = []
    for year, sub in df.groupby('year', sort=False):
        totals = {s: float(sub[s].sum()) for s in available}
        out.append((player_name, int(year), totals, len(sub)))
    return out


def _aggregate_all_players(data_dir: str) -> Dict[int, List[Tuple[str, Dict[str, float], int]]]:
    """Single pass over all player files → {year: [(player, totals, games), ...]}."""
    files = glob.glob(os.path.join(data_dir, "*_performance_details.csv"))
    logging.info(f"Single-pass aggregation of {len(files)} player files...")
    by_year: Dict[int, List[Tuple[str, Dict[str, float], int]]] = defaultdict(list)
    for i, path in enumerate(files, 1):
        for player, year, totals, games in _aggregate_one_file(path):
            by_year[year].append((player, totals, games))
        if i % 1000 == 0:
            logging.info(f"  aggregated {i}/{len(files)} files")
    logging.info(f"Aggregation complete — {len(by_year)} years found")
    return dict(by_year)


def _generate_yearly_from_memory(
    year: int,
    year_data: List[Tuple[str, Dict[str, float], int]],
    weights: dict,
    output_dir: str,
    career_gpg: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, int, Dict[str, int], int, float, float]]:
    """Apply the full ranking algorithm to one year's pre-aggregated data.

    Yearly ordering is by raw score within the era (unchanged). Position
    stratification has been removed — the all-time aggregator now uses
    the player's *rank within the yearly top 100* (rank-based formula),
    not the z-score, so within-year z-scores no longer need to be
    position-fair. Full-cohort z-scores are still computed and persisted
    in the returned tuple for backward compatibility, but they do not
    feed the all-time ranking.

    `career_gpg` is accepted for signature compatibility with `main()` but
    is no longer used here (kept so callers don't need to change). The
    `_career_position_group` helper is retained in this module for
    historical/debug purposes.
    """
    if not year_data:
        return []

    era_name, era_stats = get_era(year)
    shrinkage = math.sqrt(ERA_COMPLETENESS.get(era_name, 0.40))

    player_scores = []
    for player_name, totals, games_played in year_data:
        contributions = {
            stat: totals.get(stat, 0.0) * weights.get(stat, 0.0)
            for stat in era_stats if stat in totals
        }
        if not contributions:
            continue
        uncapped = sum(contributions.values())
        if uncapped <= 0:
            continue
        excess = sum(max(0.0, c - SINGLE_STAT_CAP * uncapped) for c in contributions.values())
        score = max(0, int(uncapped - excess))
        era_totals = {stat: int(totals.get(stat, 0)) for stat in era_stats if stat in totals}
        player_scores.append((player_name, score, era_totals, games_played))

    if not player_scores:
        return []

    scores = [s for _, s, _, _ in player_scores]
    percentile_ranks = calculate_percentile_ranks(scores)

    # Full-cohort z-scores (no position stratification). The all-time
    # aggregator uses rank within the yearly top 100, not z, so this
    # signal is informational only — it preserves the persisted return
    # type and keeps the debug log meaningful for historical tracing.
    raw_z = compute_within_era_z_scores(scores)
    z_scores = raw_z * shrinkage

    enriched = list(zip(player_scores, percentile_ranks, z_scores))
    sorted_players = sorted(enriched, key=lambda x: (x[0][1], x[0][3]), reverse=True)
    top_100 = sorted_players[:100]

    df = create_ranking_dataframe([(p[0], p[1], pr, p[3]) for p, pr, _ in top_100])
    df.to_csv(os.path.join(output_dir, 'yearly', f'year_{year}.csv'), index=False)

    return [(p[0], p[1], p[2], p[3], pr, z) for p, pr, z in top_100]


WEIGHTS = {
    'goals': 55.0,
    'behinds': 1.5,
    # kicks and handballs weighted separately — 'disposals' is excluded from
    # all ERAS lists to prevent double-counting (disposals = kicks + handballs).
    # Handball ≈ 65% of kick value (less distance, less accuracy) → 3.0 vs 4.5.
    'kicks': 4.5,
    'handballs': 3.0,
    'marks': 2.5,
    'goal_assist': 4.0,
    'contested_marks': 7.0,
    'contested_possessions': 5.5,
    'tackles': 3.5,
    'one_percenters': 3.0,
    'clearances': 5.5,
}


def main():
    started = datetime.now()
    logging.info(f"=== top_players_comprehensive started at {started.isoformat(timespec='seconds')} ===")

    data_dir = "./data/player_data/"
    output_dir = "./data/top100/"
    os.makedirs(os.path.join(output_dir, 'yearly'), exist_ok=True)

    # Single-pass aggregation: read each of ~13k player files exactly once.
    raw_aggregates = _aggregate_all_players(data_dir)

    # Single pass over all years to compute per-player career totals.
    # career_gpg: career goals/game — used for stable 3-group position classification
    #   so high-scoring forwards are never z-scored against pure key forwards.
    # true_career_games: total games from ALL seasons (not just top-100 seasons),
    #   fixing the bug where injury-affected seasons were silently dropped.
    career_goals: Dict[str, float] = defaultdict(float)
    true_career_games: Dict[str, int] = defaultdict(int)
    for year_data in raw_aggregates.values():
        for player, totals, games in year_data:
            career_goals[player] += totals.get('goals', 0.0)
            true_career_games[player] += games
    career_gpg: Dict[str, float] = {
        p: career_goals[p] / true_career_games[p]
        for p in true_career_games
        if true_career_games[p] > 0
    }
    logging.info("Career g/game computed for %d players.", len(career_gpg))

    yearly_top_100 = {}
    for year in sorted(raw_aggregates.keys()):
        logging.info(f"Processing yearly rankings for {year}")
        top_100 = _generate_yearly_from_memory(
            year, raw_aggregates[year], WEIGHTS, output_dir, career_gpg=career_gpg
        )
        if top_100:
            yearly_top_100[year] = top_100

    compile_all_time_top_100(yearly_top_100, output_dir, career_games=dict(true_career_games))

    elapsed = datetime.now() - started
    logging.info(f"=== top_players_comprehensive complete in {elapsed} ===")


if __name__ == "__main__":
    main()
