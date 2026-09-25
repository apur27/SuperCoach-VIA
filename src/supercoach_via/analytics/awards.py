"""Brownlow PROXY (a stat-profile index; A04). Never a vote prediction or probability.

Formula (preserved from ``update_team_analysis._build_brownlow_proxy_table`` at b4ce74770):

* player-season rows (every row of the season, as legacy), blank stat cells counted as 0
  (legacy imputation, disclosed), ``effective_disposals = max(disposals - clangers, 0)``;
* per-game means of disposals, clearances, contested possessions, goals, effective disposals;
* players with at least ``min_games`` (3) rows form the cohort; each mean is z-scored across
  the cohort with population SD (0 when SD is 0);
* ``proxy_per_game = 0.30 z_disp + 0.22 z_clear + 0.18 z_cp + 0.15 z_eff + 0.15 z_goals``;
  rank by ``proxy_per_game`` desc (ties: player_id).

Deliberate correction (versioned): the season-scaled column multiplies by the number of
regular-season games per club found in the FIXTURE (legacy hard-coded 22, but 2024-2026 seasons
have 23), falling back to the legacy constant only when the fixture is absent and saying so.
It is a pure rescale for legibility; rank order is unchanged.

Observed Brownlow votes (``brownlow_votes``) are reported separately and never mixed into the
proxy. Ineligibility (in-season suspension) comes from an explicit, sourced registry.
"""

from __future__ import annotations

# SQL is composed only from whitelisted identifiers (canonical stat/column names) and int-cast
# literals; every caller-supplied value is a bound parameter.
# ruff: noqa: S608
from collections.abc import Mapping
from dataclasses import dataclass

from supercoach_via.storage.queries import SnapshotQuery

BROWNLOW_PROXY_VERSION = "brownlow_proxy_v1.1"  # v1 formula; v1.1 = fixture-derived season scale
BROWNLOW_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("disposals", 0.30),
    ("clearances", 0.22),
    ("contested_possessions", 0.18),
    ("effective_disposals", 0.15),
    ("goals", 0.15),
)
LEGACY_SEASON_GAMES = 22  # config.HOME_AND_AWAY_GAMES at baseline (known off by one, BL-20)
MIN_GAMES = 3

LABEL = "Brownlow stat-profile proxy"
METHOD = (
    "Weighted sum of cohort z-scores of per-game disposals (0.30), clearances (0.22), contested "
    "possessions (0.18), effective disposals = disposals - clangers (0.15) and goals (0.15), for "
    "players with 3+ games; blank cells count as zero (legacy imputation). An index of statistical "
    "output that correlates with umpire votes. It is not a vote count and not a probability."
)


@dataclass(frozen=True)
class Ineligibility:
    reason: str
    source: str


#: Registry ported from update_team_analysis.BROWNLOW_INELIGIBLE_2026 (keys -> legacy player ids).
_LEGACY_REGISTRY_SOURCE = "update_team_analysis.BROWNLOW_INELIGIBLE_2026 @ b4ce74770"
INELIGIBILITY_REGISTRY: dict[int, dict[str, Ineligibility]] = {
    2026: {
        "legacy:xerri_tristan_15031999": Ineligibility("Suspended (2026 season)", _LEGACY_REGISTRY_SOURCE),
    },
}


def default_ineligibility(season: int) -> dict[str, Ineligibility]:
    return dict(INELIGIBILITY_REGISTRY.get(season, {}))


@dataclass(frozen=True)
class BrownlowProxyRow:
    rank: int
    player_id: str
    name: str
    club: str
    games: int
    disposals_pg: float
    clearances_pg: float
    contested_possessions_pg: float
    goals_pg: float
    effective_disposals_pg: float
    tackles_pg: float
    proxy_per_game: float
    season_proxy_scaled: float
    observed_votes: int | None  # recorded brownlow_votes; None when none recorded
    votes_observed_games: int
    ineligible: bool
    ineligible_reason: str | None
    ineligible_source: str | None


@dataclass(frozen=True)
class BrownlowProxyTable:
    season: int
    version: str
    label: str
    method: str
    season_games: int
    scale_basis: str
    scale_label: str
    min_games: int
    cohort_players: int
    rows: tuple[BrownlowProxyRow, ...]


def _season_games(q: SnapshotQuery, season: int) -> tuple[int, str]:
    if "matches" in q.manifest.tables:
        n = q.scalar(
            """SELECT MAX(c) FROM (SELECT club, COUNT(*) AS c FROM (
                   SELECT home_club_id AS club FROM matches WHERE season = ? AND stage_type = 'regular'
                          AND status <> 'cancelled'
                   UNION ALL SELECT away_club_id FROM matches WHERE season = ? AND stage_type = 'regular'
                          AND status <> 'cancelled') GROUP BY club)""",
            [int(season), int(season)],
        )
        if n:
            return int(n), "fixture: regular-season matches per club"
    return LEGACY_SEASON_GAMES, "legacy constant (no fixture in the data)"


def brownlow_proxy(
    q: SnapshotQuery,
    season: int,
    *,
    ineligible: Mapping[str, Ineligibility] | None = None,
    min_games: int = MIN_GAMES,
) -> BrownlowProxyTable:
    import numpy as np

    registry = default_ineligibility(season) if ineligible is None else dict(ineligible)
    names_join, name_expr = "", "g.player_id"
    if "players" in q.manifest.tables:
        names_join, name_expr = "LEFT JOIN players p USING (player_id)", "COALESCE(p.display_name, g.player_id)"
    club_expr, club_join = "g.club_id", ""
    if "clubs" in q.manifest.tables:
        club_expr, club_join = "COALESCE(c.name, g.club_id)", "LEFT JOIN clubs c ON c.club_id = g.club_id"
    df = q.df(
        f"""WITH g AS (SELECT * FROM player_games WHERE season = ?),
            agg AS (
              SELECT g.player_id, ANY_VALUE({name_expr}) AS name, COUNT(*) AS games,
                     AVG(COALESCE(disposals, 0)) AS disposals_pg,
                     AVG(COALESCE(clearances, 0)) AS clearances_pg,
                     AVG(COALESCE(contested_possessions, 0)) AS contested_possessions_pg,
                     AVG(COALESCE(goals, 0)) AS goals_pg,
                     AVG(GREATEST(COALESCE(disposals, 0) - COALESCE(clangers, 0), 0)) AS effective_disposals_pg,
                     AVG(COALESCE(tackles, 0)) AS tackles_pg,
                     SUM(brownlow_votes) AS observed_votes, COUNT(brownlow_votes) AS votes_observed_games
              FROM g {names_join} GROUP BY g.player_id),
            clubs AS (
              SELECT player_id, club FROM (
                SELECT player_id, club,
                       ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY n DESC, club) AS rn
                FROM (SELECT g.player_id, {club_expr} AS club, COUNT(*) AS n
                      FROM g {club_join} GROUP BY g.player_id, club)) WHERE rn = 1)
            SELECT agg.*, clubs.club FROM agg JOIN clubs USING (player_id)
            WHERE games >= ? ORDER BY player_id""",
        [int(season), int(min_games)],
    )
    season_games, basis = _season_games(q, season)
    if df.empty:
        return BrownlowProxyTable(season, BROWNLOW_PROXY_VERSION, LABEL, METHOD, season_games, basis,
                                  f"Season-scaled proxy (x{season_games})", min_games, 0, ())

    def z(col: str) -> np.ndarray:
        x = df[col].to_numpy(dtype=float)
        sd = x.std(ddof=0)
        return np.zeros_like(x) if sd == 0 or not np.isfinite(sd) else (x - x.mean()) / sd

    cols = {
        "disposals": "disposals_pg",
        "clearances": "clearances_pg",
        "contested_possessions": "contested_possessions_pg",
        "effective_disposals": "effective_disposals_pg",
        "goals": "goals_pg",
    }
    proxy = sum(w * z(cols[k]) for k, w in BROWNLOW_WEIGHTS)
    df["proxy"] = proxy
    df = df.sort_values(["proxy", "player_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    rows = []
    for i, r in enumerate(df.to_dict("records"), start=1):
        inel = registry.get(str(r["player_id"]))
        votes = None if r["votes_observed_games"] == 0 else int(r["observed_votes"])
        rows.append(
            BrownlowProxyRow(
                rank=i,
                player_id=str(r["player_id"]),
                name=str(r["name"]),
                club=str(r["club"]),
                games=int(r["games"]),
                disposals_pg=float(r["disposals_pg"]),
                clearances_pg=float(r["clearances_pg"]),
                contested_possessions_pg=float(r["contested_possessions_pg"]),
                goals_pg=float(r["goals_pg"]),
                effective_disposals_pg=float(r["effective_disposals_pg"]),
                tackles_pg=float(r["tackles_pg"]),
                proxy_per_game=float(r["proxy"]),
                season_proxy_scaled=float(r["proxy"]) * season_games,
                observed_votes=votes,
                votes_observed_games=int(r["votes_observed_games"]),
                ineligible=inel is not None,
                ineligible_reason=inel.reason if inel else None,
                ineligible_source=inel.source if inel else None,
            )
        )
    return BrownlowProxyTable(
        season=season,
        version=BROWNLOW_PROXY_VERSION,
        label=LABEL,
        method=METHOD,
        season_games=season_games,
        scale_basis=basis,
        scale_label=f"Season-scaled proxy (x{season_games}; index, not votes)",
        min_games=min_games,
        cohort_players=len(rows),
        rows=tuple(rows),
    )
