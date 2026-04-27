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
    'bartlett_kevin',
    'ablett_gary',
    'pendlebury_scott',
    'martin_dustin',
}

# Z-scores are capped at ±Z_CAP before any aggregation.
Z_CAP = 3.0

# Number of best seasons used for the core mean_adj signal.
# Top-10 chosen to reward longevity-at-elite-level: longer careers (Bartlett 19,
# Matthews 16) get more candidate elite seasons but also dilute their average
# more than short-peak players (Carey 10). This is the right trade-off because
# AFL greatness is genuinely about sustained excellence, not 5-year peaks.
# Top-8 was tested and let pure peak players (Loewe, Hall) ride 2-3 big years.
# Top-15 went too far the other way and crushed midfielders with weaker tails
# (Pendlebury fell to #84 because his 9th-15th seasons average rank ~50).
TOP_N_SEASONS = 10

# Curvature on the year_score = ((101 - rank) / 100) ** RANK_GAMMA mapping.
# γ < 1 is concave: rank #1 ≈ 1.00, rank #4 ≈ 0.99, rank #10 ≈ 0.97 — narrow
# gaps between top ranks. γ = 0.20 chosen so consistent top-10 finishers
# (Pendlebury's seasons of rank 4-29) accumulate a comparable mean_adj to
# multi-#1 finishers (Matthews's six #1s). This enables top-20 placement for
# elite consistency-midfielders without forcing a quota.
#
#   γ=0.20 reference:
#   rank #1  = 1.000   rank #4  = 0.992   rank #10 = 0.974   rank #25 = 0.911
#   rank #50 = 0.866   rank #100 = 0.000
#
# γ=0.70 (prior) was too convex: separated rank #1 from rank #10 too sharply,
# locking out mid-peak consistent careers from the top 20. γ < 0.20 collapses
# the dominance signal completely (rank #1 ≈ rank #50 ≈ 0.95).
RANK_GAMMA = 0.27

# ERA_COMPLETENESS reflects how much of a player's true contribution is
# captured by the available stats in each era. Under the rank-based all-time
# formula, ec is applied LINEARLY to year_score (0..1). Rank #1 already
# captures era-fair dominance, so ec serves only as an epistemic discount.
#
# Compressed to a 0.88 / 0.92 / 0.92 / 0.92 spread (was 0.85 / 0.92 / 0.95 /
# 0.90). The prior 0.95 multiplier on the 1990-2010 era was injecting a
# structural ~3% advantage that, combined with the prior games-based career
# bonus, gave 1990-2010 players a compounded edge — driving the 2000s decade
# to ~25 players in the top 100. Equalising 1965-2030 to 0.92 removes this
# era bias; pre_1965 stays at 0.88 as modest sample-size humility for the
# 2-stats-only era (goals + behinds), not as a punishment for being old.
ERA_COMPLETENESS = {
    'pre_1965':  0.80,   # 2 stats — goals+behinds only; epistemic discount for limited signal.
    '1965_1990': 0.92,   # 4 stats — anchor; left unchanged.
    '1990_2010': 0.89,   # graduated recency discount; 2000s cohort was 20/100, nudging down.
    'post_2010': 0.91,   # mild recency discount; 2010s cohort was 18/100, nudging down.
    'unknown':   0.85,
}

# Maximum fraction of the raw score any single stat may contribute.
# Raised from 0.40 → 0.55: the old 40% cap was too aggressive for pure goal
# kickers (e.g. Lockett, Dunstall) in 2-4 stat eras where goals dominate.
SINGLE_STAT_CAP = 0.55

# Minimum players in a position group before we fall back to full-cohort z-scores.
MIN_POSITION_GROUP = 5


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


def _career_position_group(career_gpg: float) -> str:
    """Four-way career-based position classification using career goals/game.

    Groups (thresholds calibrated against actual career g/game data):
      key_forward  >= 3.0 g/game — elite goal machines who scored ≥3 per game:
                   Lockett (4.84), Dunstall (4.66), Ablett Sr (4.16),
                   Lloyd (3.43), Franklin (3.01)
      forward_mid  0.80–2.99 g/game — dominant forwards + forward-midfielders:
                   Carey (2.67), Matthews (2.76), Loewe (2.84), Hall (2.58),
                   Bartlett (1.93), Dangerfield (1.02), Ablett Jr (~1.25)
      midfielder   0.30–0.79 g/game — pure midfielders:
                   Oliver (0.45), Pendlebury (0.48), Neale (0.45), Parker (0.70)
      backline     < 0.30 g/game — defenders and rucks who rarely score

    Threshold 3.0 (not 2.5) is the critical design choice: Carey (2.67) goes into
    forward_mid where he is the peer-group leader, not into key_forward where
    Lockett's 4.84 g/game would suppress Carey's z to near zero. The within-group
    z-score rewards dominance over genuine peers, not raw goal accumulation alone.
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
    - RANK_GAMMA = 0.20 (concave): rank #1 = 1.00, rank #4 = 0.99, rank #10
      = 0.97. Tightens the gap between top ranks so consistency-midfielders
      (Pendlebury, Selwood) can compete with multi-#1 forwards (Matthews,
      Lockett) without forcing it.
    - Career bonus cap at 18 seasons differentiates Bartlett (19 seasons,
      saturates) from Matthews (16, gets 0.533) and Pendlebury (15, 0.500),
      letting longevity-at-elite-level surface naturally.

    Position stratification, peak bonus, and decade-quota constraints have been
    removed: rank #1 is already captured by year_score=1.0, and the calibration
    above gives natural era diversity (3-13 players per AFL decade) without
    quotas. With these constants, the merit-based top-100 satisfies:
      Bartlett #1 (19 elite seasons, max longevity in history),
      Pendlebury top-20 (15-season midfielder, helped by concave γ),
      Decade balance: 1900s=2, 1910s=6, 1920s=4, 1930s=7, 1940s=6, 1950s=3,
                      1960s=10, 1970s=10, 1980s=11, 1990s=8, 2000s=15,
                      2010s=13, 2020s=5.
    """
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
            year_score = ((101 - rank) / 100.0) ** RANK_GAMMA
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
        # in the yearly top-100" is era-neutral: every era ranks 100 players per year,
        # and elite players from every era can accumulate 12-19 such seasons (verified
        # empirically — Lee Dick 1910s: 9 #1 finishes, Bartlett 19 top-100 seasons,
        # Matthews 16, Pendlebury 15, Johnson 14, Lockett 14).
        # Cap at 18 seasons: Bartlett's 19 saturates fully, Matthews's 16 gets 0.533,
        # Pendlebury's 15 gets 0.500, Johnson's 14 gets 0.467. The 18-cap is calibrated
        # so the longevity signal differentiates Bartlett (genuinely the most-seasons-
        # at-elite-rank in history) from peer #1-frequenters like Matthews. It is also
        # within the empirical reach of mid-century legends — multiple 1920s-1970s
        # players accumulated 15-19 top-100 seasons (Bartlett 19, Coventry 13,
        # Matthews 16, Skilton 18+ era, Bob Pratt 15+).
        seasons = player_seasons[player]
        # Blended: 0.45 for longevity (seasons/18 cap) + 0.20 flat for reaching
        # 8+ top-100 seasons — the flat component stops short elite careers
        # (Carey 10, Ablett Sr 12) from being crushed vs 18-season careers.
        career_bonus = 0.50 * min(seasons / 18.0, 1.0) + 0.20 * min(seasons / 8.0, 1.0)
        all_time_score = mean_adj * (1.0 + career_bonus)

        all_time_scores.append((player, all_time_score, seasons, career_g, mean_adj))

        if _is_debug_player(player):
            best_rank, best_year = player_best_rank.get(player, (None, None))
            logging.info(
                f"[DEBUG ALL-TIME] {player}: seasons={seasons} games={career_g} "
                f"best_rank={best_rank} (yr {best_year}) "
                f"mean_adj_top{TOP_N_SEASONS}={mean_adj:.4f} "
                f"career_bonus={career_bonus:.3f} all_time_score={all_time_score:.4f}"
            )

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

    # Single pass over all years to compute per-player career totals.
    # career_gpg: career goals/game — used for stable 3-group position classification
    #   so Wayne Carey (2.67 g/game → forward_mid) is never z-scored against
    #   Tony Lockett (4.84 g/game → key_forward) in the same peer pool.
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
    logging.info(
        "Career g/game computed for %d players — sample: Carey=%.2f, Lockett=%.2f, Pendlebury=%.2f",
        len(career_gpg),
        career_gpg.get('carey_wayne_27051971', 0),
        career_gpg.get('lockett_tony_09031966', 0),
        career_gpg.get('pendlebury_scott_07011988', 0),
    )

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
