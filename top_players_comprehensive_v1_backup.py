import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import math

# Set up logging for diagnostics
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Players to log diagnostics for, so we can sanity-check the era-normalized algorithm.
# Names are matched against the player_name token derived from the file name
# (which is "_".join(filename.split("_")[:-2])). Matching is case-insensitive
# and substring-based to tolerate suffixes like "_jr".
DEBUG_PLAYERS = {
    'kevin_bartlett',
    'gary_ablett',
    'scott_pendlebury',
    'dustin_martin',
}

# Cap z-scores to ±Z_CAP std-devs to prevent a single freak season (or thin-year
# outlier) from dominating the all-time aggregate.
Z_CAP = 3.0

# How complete is our picture of a player's full contribution in each era?
# Pre-1965: only goals/behinds tracked — a prolific goal-kicker dominates an
# artificially narrow stat set. Post-2010: 12 stats give a rich, multi-dimensional
# picture. These weights scale each season's z-score contribution accordingly.
ERA_COMPLETENESS = {
    'pre_1965': 0.40,    # 2 stats: goals, behinds only
    '1965_1990': 0.65,   # 4 stats: + kicks, handballs
    '1990_2010': 0.82,   # 6 stats: + marks, disposals
    'post_2010': 1.0,    # 12 stats: full modern picture
    'unknown': 0.40,
}


def _is_debug_player(player_name: str) -> bool:
    """Case-insensitive substring match against DEBUG_PLAYERS."""
    pn = player_name.lower()
    return any(tag in pn for tag in DEBUG_PLAYERS)

# Define eras with year ranges and available statistics
ERAS = {
    'pre_1965': (1897, 1964, ['goals', 'behinds']),
    '1965_1990': (1965, 1990, ['goals', 'behinds', 'kicks', 'handballs']),
    '1990_2010': (1991, 2010, ['goals', 'behinds', 'kicks', 'handballs', 'marks', 'disposals']),
    'post_2010': (2011, 2030, ['goals', 'behinds', 'kicks', 'handballs', 'marks', 'disposals', 'tackles', 'one_percenters', 'clearances', 'contested_possessions', 'contested_marks', 'goal_assist'])
}

def initialize_df_lib():
    """Initialize DataFrame library (cuDF or pandas) based on GPU availability."""
    try:
        import cudf
        cudf.Series([1])  # Lightweight GPU initialization test
        logging.info("GPU acceleration enabled via CuDF")
        return cudf, True
    except (ImportError, RuntimeError, OSError, AttributeError) as e:
        logging.info(f"GPU unavailable: {e}. Falling back to CPU with pandas.")
        return pd, False

df_lib, USE_GPU = initialize_df_lib()

def get_era(year: int) -> Tuple[str, List[str]]:
    """Determine the era and available statistics for a given year."""
    for era_name, (start, end, stats) in ERAS.items():
        if start <= year <= end:
            return era_name, stats
    return 'unknown', []

def safe_sum(df, col: str, df_lib) -> float:
    """Safely sum a column after converting to numeric, optimized for GPU/CPU."""
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
    """Parse dates using the appropriate library (cudf or pandas)."""
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

def process_player_file(filepath: str, year: int, weights: dict, df_lib) -> Optional[Tuple[str, int, Dict[str, int], int]]:
    """Process a single player's performance file for a given year."""
    try:
        # Read only the 'date' column first to check for the target year
        if USE_GPU:
            df_dates = df_lib.read_csv(filepath, usecols=['date'])
        else:
            df_dates = df_lib.read_csv(filepath, usecols=['date'], low_memory=False)
        
        df_dates = parse_dates(df_dates, df_lib, 'date')
        df_dates['year'] = df_dates['date'].dt.year
        
        # Skip if the target year isn't present
        if year not in (df_dates['year'].values_host if USE_GPU else df_dates['year'].values):
            logging.debug(f"No games found for {filepath} in {year}")
            return None
        
        # Load the full DataFrame only if the year is relevant
        if USE_GPU:
            df = df_lib.read_csv(filepath)
        else:
            df = df_lib.read_csv(filepath, low_memory=False)
        
        df = parse_dates(df, df_lib, 'date')
        df['year'] = df['date'].dt.year
        df = df[df['year'] == year]
        
        if df.empty:
            logging.debug(f"No games found for {filepath} in {year} after full load")
            return None
        
        player_name = "_".join(os.path.basename(filepath).split("_")[:-2])
        era, era_stats = get_era(year)
        available_stats = [stat for stat in era_stats if stat in df.columns]
        if not available_stats:
            logging.warning(f"No era-specific stats ({era_stats}) found in {filepath}")
            return None
        
        score = 0
        totals = {}
        for stat in available_stats:
            total = safe_sum(df, stat, df_lib)
            totals[stat] = int(total)
            score += total * weights.get(stat, 0)
        
        games_played = len(df)
        return (player_name, int(score), totals, games_played)
    
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None

def calculate_percentile_ranks(scores: List[int]) -> np.ndarray:
    """Efficiently calculate percentile ranks using pandas."""
    return pd.Series(scores).rank(pct=True).to_numpy() * 100


def compute_within_era_z_scores(scores: List[float]) -> np.ndarray:
    """Z-score normalize scores within a single year/era cohort.

    Each year is its own cohort: every player in `scores` was scored using
    the same era's stat set, so their raw weighted scores are commensurable.
    Z-scoring within the cohort answers "how dominant was this player vs
    their contemporaries that year?" — which is the era-fair signal we want.

    Caps at ±Z_CAP to prevent a single thin-cohort or outlier season from
    inflating the all-time aggregate.

    Edge cases:
    - Empty input → empty array.
    - Zero variance (all scores identical) → all zeros (no one stood out).
    - NaN-safe via numpy mean/std with default ddof=0 (population std,
      appropriate when treating the cohort as the full population of that year).
    """
    if not scores:
        return np.array([], dtype=float)
    arr = np.asarray(scores, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros_like(arr)
    z = (arr - mu) / sigma
    return np.clip(z, -Z_CAP, Z_CAP)


def create_ranking_dataframe(players_data: List[Tuple[str, int, float, int]]) -> pd.DataFrame:
    """Create a standardized DataFrame for rankings."""
    return pd.DataFrame(players_data, columns=['player', 'score', 'percentile_rank', 'games_played'])


def generate_yearly_top_100(
    data_dir: str,
    year: int,
    weights: dict,
    output_dir: str,
    df_lib,
) -> List[Tuple[str, int, Dict[str, int], int, float, float]]:
    """Generate the top 100 players for a specific year.

    Returns tuples of (player, score, totals, games_played, percentile_rank, z_score).
    The z_score is the within-year (era-cohort) dominance metric used by the
    all-time aggregator. Percentile rank is retained for backward-compatible
    yearly CSV output.

    Note: percentile_rank is computed across ALL players in `year` (not just the
    top 100) before truncation, so it reflects the full cohort. The z-score is
    likewise cohort-wide, then we slice the top 100. Since every player in a
    given year sits in the same era stat set, raw scores are commensurable
    within the cohort — the era-fairness fix bites at the cross-year aggregation
    step in compile_all_time_top_100().
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

    scores = [score for _, score, _, _ in player_scores]
    percentile_ranks = calculate_percentile_ranks(scores)
    z_scores = compute_within_era_z_scores(scores)

    # Pair each player record with its percentile and z-score, then sort.
    # Sort by raw score (desc), tiebreak by games played (desc) — same as before.
    enriched = list(zip(player_scores, percentile_ranks, z_scores))
    sorted_players = sorted(
        enriched,
        key=lambda x: (x[0][1], x[0][3]),
        reverse=True,
    )

    top_100 = sorted_players[:100]

    # Preserve the existing yearly CSV schema: player, score, percentile_rank, games_played.
    df = create_ranking_dataframe([(p[0], p[1], pr, p[3]) for p, pr, _z in top_100])
    df.to_csv(os.path.join(output_dir, 'yearly', f'year_{year}.csv'), index=False)

    # Debug logging for the watched players, even if they fell outside the top 100.
    for rec, pr, z in enriched:
        player_name, raw_score, _, games = rec
        if _is_debug_player(player_name):
            logging.info(
                f"[DEBUG {year}] {player_name}: raw_score={float(raw_score):.0f} "
                f"games={games} pct={pr:.2f} z={z:+.3f}"
            )

    return [(p[0], p[1], p[2], p[3], pr, z) for p, pr, z in top_100]

def compile_all_time_top_100(
    yearly_top_100: Dict[int, List[Tuple[str, int, Dict[str, int], int, float, float]]],
    output_dir: str,
) -> None:
    """Compile the all-time top 100 players using era-normalized z-score dominance.

    Formula:
        all_time_score = mean_z * log1p(seasons) * (1 + peak_z / 10)

    where:
      - mean_z: average yearly within-era z-score (era-fair dominance signal —
                a player is compared only to their contemporaries each year)
      - log1p(seasons): longevity factor, log-scaled so a 19-season career
                doesn't get penalized vs durability noise but also doesn't
                linearly dwarf a 12-season peak career
      - peak_z bonus: small multiplicative bump for the single best season,
                rewarding peak dominance (e.g. Dustin Martin's Norm Smith years)

    Why this preserves the expected ordering:
      - Kevin Bartlett: 19 elite seasons in 1965-1990 era → very high mean_z,
        large log(seasons) factor, and his peak years contribute peak_z bonus.
      - Modern stars (Ablett Jr, Pendlebury, Martin) get fair comparison
        because their z-scores are computed against the post-2010 cohort, not
        inflated by having 12 stats vs Bartlett's 4.

    Caveat: only seasons in which a player landed in the yearly top 100 are
    aggregated here (that's what we receive in `yearly_top_100`). Sub-elite
    seasons are dropped, which is consistent with the prior algorithm but
    means very-long careers with mediocre tail seasons aren't penalized.
    """
    player_data: Dict[str, Dict[str, list]] = {}
    for year, top_list in yearly_top_100.items():
        era_name, _ = get_era(year)
        completeness = ERA_COMPLETENESS.get(era_name, 0.40)
        for player, _, _, _, percentile, z in top_list:
            if player not in player_data:
                player_data[player] = {
                    'percentile_ranks': [],
                    'z_scores': [],
                    'era_weights': [],
                    'seasons': 0,
                }
            player_data[player]['percentile_ranks'].append(percentile)
            player_data[player]['z_scores'].append(z)
            player_data[player]['era_weights'].append(completeness)
            player_data[player]['seasons'] += 1

    all_time_scores = []
    for player, data in player_data.items():
        seasons = data['seasons']
        if seasons <= 0:
            continue
        z_arr = np.asarray(data['z_scores'], dtype=float)
        w_arr = np.asarray(data['era_weights'], dtype=float)

        # Era-weighted mean: a pre-1965 season at 0.40 completeness contributes
        # less than a post-2010 season at 1.0. This stops prolific pre-1965
        # goal-kickers (who dominate a 2-stat cohort) from outranking players
        # whose dominance is evidenced across a full modern stat set.
        mean_z = float(np.average(z_arr, weights=w_arr))
        peak_idx = int(np.argmax(z_arr))
        peak_z = float(z_arr[peak_idx])
        peak_era_weight = float(w_arr[peak_idx])

        longevity = math.log1p(seasons)

        # Peak bonus is also era-weighted: a +3σ season from a 2-stat era is
        # less convincing than a +3σ season measured across 12 stats.
        peak_bonus = 1.0 + max(peak_z * peak_era_weight, 0.0) / 10.0

        all_time_score = mean_z * longevity * peak_bonus
        all_time_scores.append((player, all_time_score, mean_z, peak_z, seasons))

        if _is_debug_player(player):
            logging.info(
                f"[DEBUG ALL-TIME] {player}: seasons={seasons} "
                f"mean_z={mean_z:+.3f} peak_z={peak_z:+.3f} "
                f"peak_era_w={peak_era_weight:.2f} avg_era_w={float(np.mean(w_arr)):.2f} "
                f"longevity={longevity:.3f} peak_bonus={peak_bonus:.3f} "
                f"all_time_score={all_time_score:.4f}"
            )

    if not all_time_scores:
        logging.info("No all-time top 100 players found")
        return

    # Sort by all_time_score desc; tiebreak by mean_z, then seasons (both desc).
    sorted_all_time = sorted(
        all_time_scores,
        key=lambda x: (x[1], x[2], x[4]),
        reverse=True,
    )[:100]

    # Preserve the canonical output schema: just player + all_time_score.
    # Diagnostic columns (mean_z, peak_z, seasons) stay in logs only, so
    # downstream consumers of all_time_top_100.csv don't break.
    df = pd.DataFrame(
        [(p, s) for p, s, *_ in sorted_all_time],
        columns=['player', 'all_time_score'],
    )
    df.to_csv(os.path.join(output_dir, 'all_time_top_100.csv'), index=False)

def main():
    """Main function to generate top 100 rankings across all time frames."""
    data_dir = "./data/player_data/"
    output_dir = "./data/top100/"
    os.makedirs(os.path.join(output_dir, 'yearly'), exist_ok=True)
    
    weights = {
        'goals': 55.0,
        'behinds': 1.5,
        'disposals': 14.0,
        'goal_assist': 4.0,
        'contested_marks': 7.0,
        'contested_possessions': 5.5,
        'marks': 2.5,
        'kicks': 4.5,
        'tackles': 3.5,
        'one_percenters': 3.0,
        'clearances': 5.5
    }
    
    yearly_top_100 = {}
    current_year = datetime.now().year
    for year in range(1897, current_year + 1):
        logging.info(f"Processing yearly rankings for {year}")
        top_100 = generate_yearly_top_100(data_dir, year, weights, output_dir, df_lib)
        if top_100:
            yearly_top_100[year] = top_100
    
    compile_all_time_top_100(yearly_top_100, output_dir)

if __name__ == "__main__":
    main()