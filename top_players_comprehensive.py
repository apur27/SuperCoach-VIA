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

DEBUG_PLAYERS = {
    'kevin_bartlett',
    'gary_ablett',
    'scott_pendlebury',
    'dustin_martin',
}

# Z-scores are capped at ±Z_CAP before any aggregation.
Z_CAP = 3.0

# Number of best seasons used for the core mean_z signal.
# 5 (reduced from 8) focuses on peak excellence rather than career accumulation.
# Expert consensus values "being the best player in the comp" over sustained
# mediocrity — Carey's 5 exceptional seasons should outrank 8 good Johnson seasons.
TOP_N_SEASONS = 5

# ERA_COMPLETENESS reflects how much of a player's true contribution is
# captured by the available stats in each era. Applied as sqrt(completeness)
# shrinkage on raw z-scores (z_adj = z * sqrt(c)) so a pre-1965 +3σ season
# is treated with appropriate epistemic humility vs a post-2010 +3σ season.
# Calibrated to close the structural gap: pre-1965 players were historically
# undervalued (0.40 was too aggressive). Post-2010 is set to 0.92, not 1.0,
# because modern stats still omit GPS, pressure acts, and defensive ratings.
ERA_COMPLETENESS = {
    'pre_1965':  0.65,   # raised from 0.40 — 2 stats, but shrinkage was too harsh
    '1965_1990': 0.78,   # raised from 0.65 — 4 stats
    '1990_2010': 0.90,   # raised from 0.82 — 6 stats
    'post_2010': 0.92,   # lowered from 1.0  — modern stats still incomplete
    'unknown':   0.65,
}

# Maximum fraction of the raw score any single stat may contribute.
# Raised from 0.40 → 0.55: the old 40% cap was too aggressive for pure goal
# kickers (e.g. Lockett, Dunstall) in 2-4 stat eras where goals dominate.
SINGLE_STAT_CAP = 0.55


def _is_debug_player(player_name: str) -> bool:
    pn = player_name.lower()
    return any(tag in pn for tag in DEBUG_PLAYERS)


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


def _position_key(totals: Dict[str, int], games_played: int) -> str:
    """Classify a player's role using goals-per-game as a proxy.

    Position is not recorded in the dataset. Goals/game >= 1.0 reliably
    identifies forwards across all eras. This ensures pre-1965 goal-kickers
    are z-scored against other forwards, not against midfielders who recorded
    zero goals — which was inflating their dominance signal.
    """
    if games_played <= 0:
        return 'other'
    return 'forward' if totals.get('goals', 0) / games_played >= 1.0 else 'other'


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

    for rec, pr, z in enriched:
        player_name, raw_score, _, games = rec
        if _is_debug_player(player_name):
            logging.info(
                f"[DEBUG {year}] {player_name}: raw_score={float(raw_score):.0f} "
                f"games={games} pos={_position_key(rec[2], games)} "
                f"pct={pr:.2f} z_adj={z:+.3f} shrinkage={shrinkage:.3f}"
            )

    return [(p[0], p[1], p[2], p[3], pr, z) for p, pr, z in top_100]


def compile_all_time_top_100(
    yearly_top_100: Dict[int, List[Tuple[str, int, Dict[str, int], int, float, float]]],
    output_dir: str,
    career_games: Optional[Dict[str, int]] = None,
) -> None:
    """Compile the all-time top 100 using era-fair z-score dominance.

    Formula:
        all_time_score = mean_z_top8 * longevity + peak_bonus

    Components:
      - mean_z_top8: era-adjusted mean z-score of the player's best TOP_N_SEASONS
                     seasons (already shrunk by sqrt(completeness) in generate_yearly).
      - longevity = min(career_games / 250, 1.5): games-based rather than seasons-based
                     so that a 9-season player with 185 games (e.g. Clayton Oliver) is
                     meaningfully separated from a 19-season, 400-game player (Bartlett).
                     Capped at 1.5× so durability doesn't overwhelm dominance.
                     Minimum 150 career games required to appear in the list at all.
      - peak_bonus = 0.15 * max(peak_z_adj, 0): additive nudge for the single best
                     season. Rewards Norm-Smith-tier peaks without dominating the score.

    Brownlow bonus deliberately excluded: it was creating a recency bias because modern
    players accumulate Brownlow votes across far more seasons of tracked data, pushing
    current midfielders unreasonably high relative to pre-1990 legends.
    """
    player_data: Dict[str, Dict] = {}

    for _, top_list in yearly_top_100.items():
        for player, _, _, games, percentile, z_adj in top_list:
            if player not in player_data:
                player_data[player] = {
                    'percentile_ranks': [],
                    'z_scores': [],
                    'total_games': 0,
                    'seasons': 0,
                }
            player_data[player]['percentile_ranks'].append(percentile)
            player_data[player]['z_scores'].append(z_adj)
            # Use games from this yearly-top-100 season only for z-score tracking;
            # true career game count is overridden below from the full aggregates.
            player_data[player]['total_games'] += games
            player_data[player]['seasons'] += 1

    # Override total_games with the true career count if provided.
    # The yearly-top-100 loop above only counts games from seasons where the
    # player cracked the top 100, silently dropping injury-affected or early/late
    # career seasons and artificially depressing longevity for fragmented careers
    # (Carey lost 60 games, Voss 83, Hird 89 under the old approach).
    if career_games:
        for player in player_data:
            if player in career_games:
                player_data[player]['total_games'] = career_games[player]

    all_time_scores = []
    for player, data in player_data.items():
        seasons = data['seasons']
        total_games = data['total_games']
        if seasons <= 0 or total_games < 150:
            continue

        z_arr = np.asarray(data['z_scores'], dtype=float)

        top_idx = np.argsort(z_arr)[::-1][:TOP_N_SEASONS]
        mean_z = float(z_arr[top_idx].mean())
        peak_z_adj = float(z_arr[top_idx[0]])

        # Career bonus — ADDITIVE, not multiplicative.
        #
        # Old formula  : score = mean_z × min(games/250, 1.5)
        # Problem      : a 364-game player (1.456×) beat a 272-game peak-dominator
        #                (1.088×) even with 20% worse z-scores — Brad Johnson at #1,
        #                Wayne Carey at #16. Expert consensus (AFL-commissioned Sheahan
        #                list, 21-greats poll) puts Carey #1 by a wide margin.
        #
        # New formula  : score = mean_z × (1 + career_bonus) + peak_bonus
        # Career bonus : max 30% additive for 300+ games — longevity is a modest
        #                differentiator, not a dominant multiplier. A 272-game legend
        #                (Carey) gets 27% bonus; a 364-game good-but-not-GOAT player
        #                (Johnson) gets the same 30% cap. Dominance wins.
        # Peak bonus   : raised 0.15 → 0.25 — experts explicitly value "seasons where
        #                you were the #1 player in the competition" (Carey's 1996,
        #                Matthews' 1975, Ablett Sr's 1989 Norm Smith year).
        career_bonus = 0.30 * min(total_games / 300.0, 1.0)
        peak_bonus = 0.25 * max(peak_z_adj, 0.0)

        all_time_score = mean_z * (1.0 + career_bonus) + peak_bonus
        all_time_scores.append((player, all_time_score, mean_z, peak_z_adj, total_games))

        if _is_debug_player(player):
            logging.info(
                f"[DEBUG ALL-TIME] {player}: seasons={seasons} games={total_games} "
                f"mean_z_top{TOP_N_SEASONS}={mean_z:+.3f} peak_z_adj={peak_z_adj:+.3f} "
                f"career_bonus={career_bonus:.3f} peak_bonus={peak_bonus:.3f} "
                f"all_time_score={all_time_score:.4f}"
            )

    if not all_time_scores:
        logging.info("No all-time top 100 players found")
        return

    # Full list sorted by score — used as the base for both constraints below.
    all_sorted = sorted(
        all_time_scores,
        key=lambda x: (x[1], x[2], x[4]),  # score, mean_z, total_games
        reverse=True,
    )

    # --- Decade representation constraint ---
    # Determine each player's decade from their first year in the yearly top 100.
    player_decade: Dict[str, int] = {}
    for year in sorted(yearly_top_100.keys()):
        decade = (year // 10) * 10
        for entry in yearly_top_100[year]:
            if entry[0] not in player_decade:
                player_decade[entry[0]] = decade

    # For each decade, reserve the top-3 scorers as guaranteed inclusions.
    by_decade: Dict[int, list] = defaultdict(list)
    for entry in all_sorted:
        by_decade[player_decade.get(entry[0], 0)].append(entry)

    # Reduced from 3 → 1: the old 3-per-decade reservation locked up 39 of 100
    # slots for decade anchors, displacing players like Lockett (#90 on merit)
    # in favour of 1890s/1900s players whose stats are too sparse to rank fairly.
    # Top-1 per decade (13 slots) still guarantees historical breadth while
    # keeping the remaining 87 spots merit-driven.
    MIN_PER_DECADE = 1
    must_include: set = set()
    for decade_entries in by_decade.values():
        for entry in decade_entries[:MIN_PER_DECADE]:
            must_include.add(entry[0])

    # Build final 100: guaranteed decade reps first (in score order), then fill by score.
    seen: set = set()
    final_100 = []
    for entry in all_sorted:
        if entry[0] in must_include:
            final_100.append(entry)
            seen.add(entry[0])
    for entry in all_sorted:
        if len(final_100) >= 100:
            break
        if entry[0] not in seen:
            final_100.append(entry)
            seen.add(entry[0])

    # Re-sort so the final output is still ordered by score (guaranteed reps may have
    # lower scores than some non-guaranteed players displaced by the constraint).
    final_100.sort(key=lambda x: (x[1], x[2], x[4]), reverse=True)
    final_100 = final_100[:100]

    df = pd.DataFrame(
        [(p, s) for p, s, *_ in final_100],
        columns=['player', 'all_time_score'],
    )
    df.to_csv(os.path.join(output_dir, 'all_time_top_100.csv'), index=False)


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
) -> List[Tuple[str, int, Dict[str, int], int, float, float]]:
    """Apply the full ranking algorithm to one year's pre-aggregated data."""
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
    # Full-cohort z-scores: compare ALL players within the same year regardless
    # of position. The original position stratification (forwards vs non-forwards)
    # was intended to stop pre-1965 goal kickers from dominating, but that is
    # already handled by era_completeness shrinkage and the single-stat cap.
    # Position stratification was confirmed to severely penalise all-round
    # forwards (Carey, Ablett Sr, Bartlett) by forcing them into the same
    # forward pool as pure goal machines (Lockett, Dunstall).
    raw_z = compute_within_era_z_scores(scores)
    z_scores = raw_z * shrinkage

    enriched = list(zip(player_scores, percentile_ranks, z_scores))
    sorted_players = sorted(enriched, key=lambda x: (x[0][1], x[0][3]), reverse=True)
    top_100 = sorted_players[:100]

    df = create_ranking_dataframe([(p[0], p[1], pr, p[3]) for p, pr, _ in top_100])
    df.to_csv(os.path.join(output_dir, 'yearly', f'year_{year}.csv'), index=False)

    for rec, pr, z in enriched:
        player_name, raw_score, _, games = rec
        if _is_debug_player(player_name):
            logging.info(
                f"[DEBUG {year}] {player_name}: raw_score={float(raw_score):.0f} "
                f"games={games} pct={pr:.2f} z_adj={z:+.3f} shrinkage={shrinkage:.3f}"
            )

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

    # Compute true career game counts before any top-100 filtering.
    # This fixes the bug where compile_all_time_top_100 only counted games from
    # seasons that cracked the yearly top 100, silently dropping injury/fringe
    # seasons and depressing longevity for players like Carey (-60 games),
    # Voss (-83 games), and Hird (-89 games).
    true_career_games: Dict[str, int] = defaultdict(int)
    for year_data in raw_aggregates.values():
        for player, _, games in year_data:
            true_career_games[player] += games

    yearly_top_100 = {}
    for year in sorted(raw_aggregates.keys()):
        logging.info(f"Processing yearly rankings for {year}")
        top_100 = _generate_yearly_from_memory(year, raw_aggregates[year], WEIGHTS, output_dir)
        if top_100:
            yearly_top_100[year] = top_100

    compile_all_time_top_100(yearly_top_100, output_dir, career_games=dict(true_career_games))

    elapsed = datetime.now() - started
    logging.info(f"=== top_players_comprehensive complete in {elapsed} ===")


if __name__ == "__main__":
    main()
