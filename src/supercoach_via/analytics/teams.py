"""Team analytics: one team-game aggregate, fixture-based ladder, form, profiles (A05).

Ladder rules (documented; section 6):

* Only matches with ``stage_type='regular'``, ``status='complete'`` and BOTH scores present
  count. A missing score is unknown, never zero. Finals are returned separately
  (:func:`finals_matches`) and never enter the ladder.
* Premiership points: 4 for a win, 2 for a draw, 0 for a loss.
* ``percentage = 100 * points_for / points_against``; ``None`` when points_against is 0.
* Order: premiership points desc, percentage desc (a ``None`` percentage with points_for > 0
  sorts as infinite, with points_for = 0 as zero), points_for desc, club_id asc.
* Historical exceptions NOT implemented automatically: matches cancelled with points shared
  (e.g. 2015 Adelaide v Geelong, 2 points each) — the canonical data carries no source flag,
  so callers must pass ``points_adjustments`` explicitly; pre-1994 finals systems only
  affect :func:`finals_places`, not the ladder.

Season structure (rounds, remaining games) always comes from match/fixture rows; no fixed
season length is assumed. Form is ordered by match chronology: match_date, local_start
(nulls last), stage_order, replay_occurrence, match_id.
"""

from __future__ import annotations

# SQL is composed only from whitelisted identifiers (canonical stat/column names) and int-cast
# literals; every caller-supplied value is a bound parameter.
# ruff: noqa: S608
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

from supercoach_via.domain.metrics import aggregate
from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.storage.queries import SnapshotQuery

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from supercoach_via.publish.view_models import (
        FormEntry,
        Heuristic,
        LadderRow,
        LeaderRow,
        MatchSummary,
        StatValue,
        TeamSeason,
    )

LADDER_METHOD = (
    "Completed regular-season matches with verified scores only; 4 points per win, 2 per draw; "
    "percentage = 100 x points for / points against (blank when nothing conceded); finals shown "
    "separately."
)
WIN_POINTS = 4
DRAW_POINTS = 2

CHRONO_ORDER = "match_date NULLS LAST, local_start NULLS LAST, stage_order, replay_occurrence, match_id"

#: Legacy 5-year profile stats (update_team_analysis.PROFILE_CORE_STATS) in canonical names.
PROFILE_CORE_STATS: tuple[str, ...] = (
    "kicks", "handballs", "disposals", "marks", "goals", "tackles", "clearances", "inside_50s",
    "rebound_50s", "contested_possessions", "uncontested_possessions", "clangers", "frees_for",
    "frees_against", "hitouts", "marks_inside_50", "contested_marks",
)
PROFILE_HIGH_GOOD: tuple[str, ...] = (
    "kicks", "handballs", "disposals", "marks", "goals", "tackles", "clearances", "inside_50s",
    "rebound_50s", "contested_possessions", "uncontested_possessions", "hitouts", "marks_inside_50",
    "contested_marks", "handball_ratio", "marks_per_inside50", "tackle_rate", "frees_for",
)
PROFILE_LOW_GOOD: tuple[str, ...] = ("clangers", "frees_against")

#: Conceded-stat columns of the legacy file (data/conceded_stats) in canonical stat names.
CONCEDED_STATS: tuple[str, ...] = (
    "disposals", "kicks", "handballs", "marks", "goals", "behinds", "tackles", "hitouts",
    "inside_50s", "clearances",
)


def _has(q: SnapshotQuery, name: str) -> bool:
    return name in q.manifest.tables and (q.tables is None or name in q.tables)


def _club_names(q: SnapshotQuery) -> dict[str, str]:
    return dict(q.rows("SELECT club_id, name FROM clubs")) if _has(q, "clubs") else {}


def _stat(stat: str) -> str:
    if stat not in PLAYER_STAT_COLUMNS:
        raise ValueError(f"unknown stat {stat!r}")
    return stat


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointsAdjustment:
    club_id: str
    points: int
    reason: str


def _completed_regular(q: SnapshotQuery, season: int) -> list[tuple[str, str, int, int]]:
    return [
        (h, a, int(hs), int(as_))
        for h, a, hs, as_ in q.rows(
            """SELECT home_club_id, away_club_id, home_score, away_score FROM matches
               WHERE season = ? AND stage_type = 'regular' AND status = 'complete'
                 AND home_score IS NOT NULL AND away_score IS NOT NULL""",
            [int(season)],
        )
    ]


def ladder(
    q: SnapshotQuery, season: int, *, points_adjustments: Iterable[PointsAdjustment] = ()
) -> list[LadderRow]:
    from supercoach_via.publish.view_models import LadderRow

    clubs = [
        r[0]
        for r in q.rows(
            """SELECT DISTINCT c FROM (
                 SELECT home_club_id AS c FROM matches WHERE season = ? AND stage_type = 'regular'
                 UNION SELECT away_club_id FROM matches WHERE season = ? AND stage_type = 'regular')
               ORDER BY c""",
            [int(season), int(season)],
        )
    ]
    tally = {c: {"p": 0, "w": 0, "l": 0, "d": 0, "pf": 0, "pa": 0, "pts": 0} for c in clubs}
    for home, away, hs, as_ in _completed_regular(q, season):
        for club, pf, pa in ((home, hs, as_), (away, as_, hs)):
            t = tally[club]
            t["p"] += 1
            t["pf"] += pf
            t["pa"] += pa
            if pf > pa:
                t["w"] += 1
                t["pts"] += WIN_POINTS
            elif pf < pa:
                t["l"] += 1
            else:
                t["d"] += 1
                t["pts"] += DRAW_POINTS
    for adj in points_adjustments:
        tally[adj.club_id]["pts"] += adj.points

    def pct(t: Mapping[str, int]) -> float | None:
        return None if t["pa"] == 0 else 100.0 * t["pf"] / t["pa"]

    def sort_pct(t: Mapping[str, int]) -> float:
        p = pct(t)
        if p is None:
            return math.inf if t["pf"] > 0 else 0.0
        return p

    order = sorted(clubs)
    order.sort(key=lambda c: (tally[c]["pts"], sort_pct(tally[c]), tally[c]["pf"]), reverse=True)
    names = _club_names(q)
    return [
        LadderRow(
            position=i,
            club_id=c,
            name=names.get(c, c),
            played=tally[c]["p"],
            won=tally[c]["w"],
            lost=tally[c]["l"],
            drawn=tally[c]["d"],
            points_for=tally[c]["pf"],
            points_against=tally[c]["pa"],
            percentage=pct(tally[c]),
            premiership_points=tally[c]["pts"],
        )
        for i, c in enumerate(order, start=1)
    ]


# ---------------------------------------------------------------------------
# Match summaries / finals
# ---------------------------------------------------------------------------

_MATCH_COLS = (
    "match_id, season, stage_id, stage_label, stage_type, round_number, stage_order, "
    "replay_occurrence, local_start, match_date, date_precision, status, venue_source_name, "
    "home_club_id, away_club_id, home_final_goals, home_final_behinds, home_score, "
    "away_final_goals, away_final_behinds, away_score"
)


def _summaries(q: SnapshotQuery, where: str, params: list[object]) -> list[MatchSummary]:
    from supercoach_via.publish.view_models import MatchSummary, TeamScore

    names = _club_names(q)
    out = []
    for r in q.rows(f"SELECT {_MATCH_COLS} FROM matches WHERE {where} ORDER BY {CHRONO_ORDER}", params):
        (mid, season, stage_id, label, stype, rnd, sorder, replay, local, mdate, prec, status, venue,
         home, away, hg, hb, hs, ag, ab, as_) = r
        winner = None
        if status == "complete" and hs is not None and as_ is not None and hs != as_:
            winner = home if hs > as_ else away
        out.append(
            MatchSummary(
                match_id=mid,
                season=season,
                stage_id=stage_id,
                stage_label=label,
                stage_type=stype,
                round_number=rnd,
                stage_order=sorder,
                replay_occurrence=replay,
                local_start=local,
                match_date=mdate,
                date_precision=prec,
                status=status,
                venue=venue,
                home=TeamScore(club_id=home, name=names.get(home, home), goals=hg, behinds=hb, score=hs),
                away=TeamScore(club_id=away, name=names.get(away, away), goals=ag, behinds=ab, score=as_),
                winner_club_id=winner,
            )
        )
    return out


def finals_matches(q: SnapshotQuery, season: int) -> list[MatchSummary]:
    """Finals of a season (any status), chronological; separate from the ladder."""
    return _summaries(q, "season = ? AND stage_type = 'final'", [int(season)])


def club_fixtures(q: SnapshotQuery, club_id: str, season: int) -> list[MatchSummary]:
    return _summaries(
        q, "season = ? AND (home_club_id = ? OR away_club_id = ?)", [int(season), club_id, club_id]
    )


# ---------------------------------------------------------------------------
# Season structure and finals pathway heuristic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonStructure:
    season: int
    regular_rounds: tuple[int, ...]
    stage_labels: tuple[str, ...]  # distinct source labels in stage order
    scheduled_regular: dict[str, int]  # per club, regular matches in the fixture (not cancelled)
    completed_regular: dict[str, int]
    remaining_regular: dict[str, int]  # scheduled/postponed/in-progress/unknown regular matches
    has_future_fixture: bool
    schedule_complete: bool | None  # seasons table; None = unknown
    finals_started: bool  # a final exists in the data => the regular season is over


def season_structure(q: SnapshotQuery, season: int) -> SeasonStructure:
    rounds = tuple(
        int(r[0])
        for r in q.rows(
            """SELECT DISTINCT round_number FROM matches WHERE season = ? AND stage_type = 'regular'
               AND round_number IS NOT NULL ORDER BY 1""",
            [int(season)],
        )
    )
    labels = tuple(
        r[0]
        for r in q.rows(
            "SELECT stage_label FROM matches WHERE season = ? GROUP BY stage_label ORDER BY MIN(stage_order), 1",
            [int(season)],
        )
    )
    per_club = q.rows(
        """SELECT club,
                  COUNT(*) FILTER (WHERE status <> 'cancelled') AS scheduled,
                  COUNT(*) FILTER (WHERE status = 'complete' AND hs IS NOT NULL AND as_ IS NOT NULL) AS done,
                  COUNT(*) FILTER (WHERE status IN ('scheduled', 'postponed', 'in_progress', 'unknown')
                                   OR (status = 'complete' AND (hs IS NULL OR as_ IS NULL))) AS remaining
           FROM (SELECT home_club_id AS club, status, home_score AS hs, away_score AS as_ FROM matches
                 WHERE season = ? AND stage_type = 'regular'
                 UNION ALL
                 SELECT away_club_id, status, home_score, away_score FROM matches
                 WHERE season = ? AND stage_type = 'regular')
           GROUP BY club ORDER BY club""",
        [int(season), int(season)],
    )
    future = bool(
        q.scalar(
            "SELECT COUNT(*) FROM matches WHERE season = ? AND status IN ('scheduled', 'postponed')",
            [int(season)],
        )
    )
    finals = bool(
        q.scalar("SELECT COUNT(*) FROM matches WHERE season = ? AND stage_type = 'final'", [int(season)])
    )
    declared: bool | None = None
    if _has(q, "seasons"):
        declared = q.scalar("SELECT schedule_complete FROM seasons WHERE season = ?", [int(season)])
    return SeasonStructure(
        season=int(season),
        regular_rounds=rounds,
        stage_labels=labels,
        scheduled_regular={c: int(s) for c, s, _, _ in per_club},
        completed_regular={c: int(d) for c, _, d, _ in per_club},
        remaining_regular={c: int(r) for c, _, _, r in per_club},
        has_future_fixture=future,
        schedule_complete=declared,
        finals_started=finals,
    )


#: Finals-system rule table (number of finalists) used only when the season's finals are not
#: in the data yet. Source: VFL/AFL finals-system history; 2026 adds the Wildcard round.
FINALS_PLACES_RULES: tuple[tuple[int, int, int], ...] = (
    (2026, 9999, 10),
    (1994, 2025, 8),
    (1991, 1993, 6),
    (1972, 1990, 5),
    (1931, 1971, 4),
)


def finals_places(q: SnapshotQuery, season: int) -> tuple[int | None, str]:
    """(number of finalists, basis). Data first: a season with a Wildcard Final has 10."""
    labels = {
        str(r[0]).lower()
        for r in q.rows("SELECT DISTINCT stage_label FROM matches WHERE season = ?", [int(season)])
    }
    if {"wildcard final", "wf"} & labels:
        return 10, "data: Wildcard Final stage present (top 10)"
    for lo, hi, n in FINALS_PLACES_RULES:
        if lo <= season <= hi:
            return n, f"rule table {lo}-{hi if hi < 9999 else 'present'}: top {n}"
    return None, "unknown finals system for this season"


_finals_places = finals_places  # alias: finals_pathway's keyword shadows the function name

PathwayStatus = Literal["clinched", "eliminated", "in_contention"]


def pathway_status(table: Mapping[str, tuple[int, int]], finals_places: int) -> dict[str, PathwayStatus]:
    """Mathematical status from ``{club: (points_now, remaining_games)}``.

    ``clinched``: fewer than ``finals_places`` other clubs can still reach this club's current
    points (ties counted as a threat because percentage is undecided). ``eliminated``: at least
    ``finals_places`` clubs already have more points than this club's maximum.
    """
    out: dict[str, PathwayStatus] = {}
    for club, (pts, rem) in table.items():
        max_pts = pts + WIN_POINTS * rem
        threats = sum(1 for o, (p, r) in table.items() if o != club and p + WIN_POINTS * r >= pts)
        above = sum(1 for o, (p, _r) in table.items() if o != club and p > max_pts)
        if above >= finals_places:
            out[club] = "eliminated"
        elif threats < finals_places:
            out[club] = "clinched"
        else:
            out[club] = "in_contention"
    return out


PATHWAY_METHOD = (
    "Rule-based heuristic, not a probability or prediction: current premiership points and the "
    "regular-season matches still in the fixture for each club; a club is 'clinched' when fewer "
    "clubs than finals places can still reach its points, 'eliminated' when that many already "
    "exceed its maximum. Ties on points are treated as undecided (percentage)."
)


def finals_pathway(
    q: SnapshotQuery, season: int, *, finals_places: int | None = None
) -> dict[str, list[Heuristic]]:
    """Per-club labelled heuristics for the finals race, derived from the fixture."""
    from supercoach_via.publish.view_models import Heuristic

    places, basis = (finals_places, "caller-supplied") if finals_places else _finals_places(q, season)
    lad = ladder(q, season)
    struct = season_structure(q, season)
    remaining_known = (
        struct.has_future_fixture or struct.schedule_complete is True or struct.finals_started
    )
    out: dict[str, list[Heuristic]] = {}
    status: dict[str, PathwayStatus] = {}
    if remaining_known and places:
        status = pathway_status(
            {r.club_id: (r.premiership_points, struct.remaining_regular.get(r.club_id, 0)) for r in lad},
            places,
        )
    cutoff = lad[places - 1] if places and len(lad) >= places else None
    for row in lad:
        rem = struct.remaining_regular.get(row.club_id, 0)
        if not remaining_known:
            text = (
                f"{row.name} are {row.position} on {row.premiership_points} points; remaining "
                "regular-season games are unknown (no future fixture in the data), so no finals "
                "status is asserted."
            )
        elif places is None:
            text = f"{row.name}: finals system for {season} is unknown; no status asserted."
        else:
            gap = "" if cutoff is None else f"; the club in position {places} has {cutoff.premiership_points} points"
            text = (
                f"{row.name} are {row.position} on {row.premiership_points} points with {rem} "
                f"regular-season game{'s' if rem != 1 else ''} left in the fixture{gap}. "
                f"Status for the top {places} ({basis}): {status[row.club_id].replace('_', ' ')}."
            )
        out[row.club_id] = [Heuristic(label="Finals race (heuristic)", text=text, method=PATHWAY_METHOD)]
    return out


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------


def team_form(
    q: SnapshotQuery, club_id: str, *, n: int = 5, before: date | None = None
) -> list[FormEntry]:
    """Last ``n`` completed matches (any stage) in match chronology, oldest first."""
    from supercoach_via.publish.view_models import FormEntry

    names = _club_names(q)
    cond = "" if before is None else "AND match_date < ?"
    params: list[object] = [club_id, club_id]
    if before is not None:
        params.append(before)
    rows = q.rows(
        f"""SELECT match_id, match_date, home_club_id, away_club_id, home_score, away_score FROM matches
            WHERE (home_club_id = ? OR away_club_id = ?) AND status = 'complete'
              AND home_score IS NOT NULL AND away_score IS NOT NULL {cond}
            ORDER BY {CHRONO_ORDER}""",
        params,
    )
    out = []
    for mid, mdate, home, away, hs, as_ in rows[-n:] if n > 0 else []:
        own, opp, opp_id = (hs, as_, away) if home == club_id else (as_, hs, home)
        margin = int(own - opp)
        result: Literal["W", "L", "D"] = "W" if margin > 0 else "L" if margin < 0 else "D"
        out.append(FormEntry(match_id=mid, match_date=mdate, opponent=names.get(opp_id, opp_id),
                             result=result, margin=margin))
    return out


# ---------------------------------------------------------------------------
# Team-game aggregate
# ---------------------------------------------------------------------------


def team_games(
    q: SnapshotQuery,
    *,
    stats: Sequence[str] = PLAYER_STAT_COLUMNS,
    seasons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """ONE row per (match_id, club_id): player-stat sums with null semantics + match context.

    Per stat: ``<stat>`` = SUM over player rows (NULL when no row observed the stat) and
    ``<stat>_observed`` = player rows with the stat recorded; ``player_rows`` = player rows.
    Match context (points_for/against, result) comes from ``matches`` when present.
    """
    stats = tuple(_stat(s) for s in stats)
    season_filter = ""
    params: list[object] = []
    if seasons is not None:
        season_filter = "WHERE season IN (" + ",".join("?" for _ in seasons) + ")"
        params = [int(s) for s in seasons]
    sums = ", ".join(f'SUM("{s}") AS "{s}", COUNT("{s}") AS "{s}_observed"' for s in stats)
    base = f"""SELECT match_id, club_id, season, COUNT(*) AS player_rows, {sums}
               FROM player_games {season_filter} GROUP BY match_id, club_id, season"""
    if not _has(q, "matches"):
        return q.df(f"SELECT * FROM ({base}) ORDER BY season, match_id, club_id", params)
    return q.df(
        f"""WITH tg AS ({base}),
            m AS (SELECT match_id, stage_type, stage_label, round_number, match_date, local_start,
                         stage_order, replay_occurrence, status, home_club_id, away_club_id,
                         home_score, away_score FROM matches)
            SELECT tg.*, m.stage_type, m.stage_label, m.round_number, m.match_date, m.status,
                   CASE WHEN tg.club_id = m.home_club_id THEN m.away_club_id ELSE m.home_club_id END
                       AS opponent_club_id,
                   CASE WHEN tg.club_id = m.home_club_id THEN m.home_score ELSE m.away_score END AS points_for,
                   CASE WHEN tg.club_id = m.home_club_id THEN m.away_score ELSE m.home_score END AS points_against,
                   CASE WHEN m.home_score IS NULL OR m.away_score IS NULL THEN NULL
                        WHEN m.home_score = m.away_score THEN 'D'
                        WHEN (tg.club_id = m.home_club_id) = (m.home_score > m.away_score) THEN 'W'
                        ELSE 'L' END AS result
            FROM tg LEFT JOIN m USING (match_id)
            ORDER BY tg.season, m.match_date NULLS LAST, m.local_start NULLS LAST, m.stage_order,
                     m.replay_occurrence, tg.match_id, tg.club_id""",
        params,
    )


def conceded_stats(tg: pd.DataFrame, *, stats: Sequence[str] = CONCEDED_STATS) -> pd.DataFrame:
    """Per (match, club): the opponent's team-game totals as ``<stat>_conceded``.

    Equivalent semantics to ``data/conceded_stats`` (what the opposition recorded against the
    club); coverage is disclosed via ``opponent_player_rows`` and ``<stat>_conceded_observed``.
    """
    keep = ["match_id", "club_id", "player_rows", *stats, *(f"{s}_observed" for s in stats)]
    opp = tg[keep].rename(
        columns={
            "club_id": "opponent_club_id",
            "player_rows": "opponent_player_rows",
            **{s: f"{s}_conceded" for s in stats},
            **{f"{s}_observed": f"{s}_conceded_observed" for s in stats},
        }
    )
    base = tg[[c for c in ("match_id", "club_id", "season", "stage_label", "match_date") if c in tg.columns]]
    pairs = tg[["match_id", "club_id"]].merge(
        tg[["match_id", "club_id"]].rename(columns={"club_id": "opponent_club_id"}), on="match_id"
    )
    pairs = pairs[pairs["club_id"] != pairs["opponent_club_id"]]
    return base.merge(pairs, on=["match_id", "club_id"]).merge(opp, on=["match_id", "opponent_club_id"])


_ROT = ("disposals_conceded", "kicks_conceded", "handballs_conceded", "marks_conceded")


def fix_conceded_rotation(df: pd.DataFrame) -> pd.DataFrame:
    """Port of ``scripts/build_conceded_stats.fix_conceded_columns`` (idempotent).

    Rows with the corrupt signature ``marks == kicks + disposals`` (and not the correct
    ``disposals == kicks + handballs``) are rotated back; unexplained rows raise.
    """
    disp, kicks, hb, marks = _ROT
    out = df.copy()
    correct = out[disp] == out[kicks] + out[hb]
    corrupt = out[marks] == out[kicks] + out[disp]
    unexplained = ~correct & ~corrupt
    if bool(unexplained.any()):
        raise ValueError(f"{int(unexplained.sum())} conceded row(s) match neither invariant; refusing to guess")
    fix = corrupt & ~correct
    old_d, old_h, old_m = out.loc[fix, disp].copy(), out.loc[fix, hb].copy(), out.loc[fix, marks].copy()
    out.loc[fix, disp] = old_m
    out.loc[fix, hb] = old_d
    out.loc[fix, marks] = old_h
    return out


def team_season_stats(
    tg: pd.DataFrame, club_id: str, season: int, *, stats: Sequence[str]
) -> list[StatValue]:
    """Per-team-game means with observed team-game denominators (coverage disclosed)."""
    sub = tg[(tg["club_id"] == club_id) & (tg["season"] == season)]
    out = []
    for s in stats:
        values = [None if v != v else float(v) for v in sub[s].tolist()]  # NaN -> None
        out.append(aggregate(s, values).to_stat_value())
    return out


def club_season_leaders(
    q: SnapshotQuery, club_id: str, season: int, *, stats: Sequence[str] = ("disposals", "goals", "tackles")
) -> list[LeaderRow]:
    """Top player per stat for one club-season (season totals over observed games)."""
    from supercoach_via.analytics.players import season_leaders

    out: list[LeaderRow] = []
    for s in stats:
        out.extend(season_leaders(q, s, season, n=1, club_id=club_id))
    return out


# ---------------------------------------------------------------------------
# Five-year profile
# ---------------------------------------------------------------------------


def five_year_profile(
    q: SnapshotQuery, *, end_season: int, years: int = 5, stats: Sequence[str] = PROFILE_CORE_STATS
) -> pd.DataFrame:
    """Port of the legacy 5-year profile (mean of per-season team-game means + ratios + ranks).

    Window = the ``years`` seasons before ``end_season`` (exclusive). Differences from legacy:
    team-games come from match_id (not team/round/opponent strings), and an all-blank stat in a
    team-game is excluded from the mean instead of counting as zero. ``attrs['window']`` holds
    the seasons.
    """
    import numpy as np
    import pandas as pd

    window = list(range(end_season - years, end_season))
    tg = team_games(q, stats=stats, seasons=window)
    per_year = tg.groupby(["club_id", "season"], as_index=False)[list(stats)].mean()
    games = tg.groupby(["club_id", "season"]).size().rename("games").reset_index()
    per_year = per_year.merge(games, on=["club_id", "season"])
    per_year["handball_ratio"] = per_year["handballs"] / per_year["disposals"].replace(0, np.nan)
    if "marks_inside_50" in per_year and "inside_50s" in per_year:
        per_year["marks_per_inside50"] = per_year["marks_inside_50"] / per_year["inside_50s"].replace(0, np.nan)
    if "tackles" in per_year:
        per_year["tackle_rate"] = per_year["tackles"] / per_year["disposals"].replace(0, np.nan)
    metrics = [c for c in per_year.columns if c not in ("club_id", "season", "games")]
    profile = per_year.groupby("club_id")[metrics].mean().reset_index()
    seasons = per_year.groupby("club_id")["season"].nunique().rename("seasons").reset_index()
    team_game_n = per_year.groupby("club_id")["games"].sum().rename("team_games").reset_index()
    profile = profile.merge(seasons, on="club_id").merge(team_game_n, on="club_id")
    n = len(profile)
    for s in PROFILE_HIGH_GOOD:
        if s in profile:
            profile[f"{s}_rank"] = profile[s].rank(ascending=False, method="min").astype("Int64")
            profile[f"{s}_pct"] = (n - profile[f"{s}_rank"]) / max(n - 1, 1) * 100.0
    for s in PROFILE_LOW_GOOD:
        if s in profile:
            profile[f"{s}_rank"] = profile[s].rank(ascending=True, method="min").astype("Int64")
            profile[f"{s}_pct"] = (n - profile[f"{s}_rank"]) / max(n - 1, 1) * 100.0
    # trend: linear slope x span / mean (legacy compute_trend_changes); needs >= 3 seasons
    span = window[-1] - window[0] if len(window) > 1 else 1
    trends = []
    for club, sub in per_year.sort_values("season").groupby("club_id"):
        row: dict[str, object] = {"club_id": club}
        x = sub["season"].astype(float).to_numpy()
        for c in metrics:
            y = sub[c].astype(float).to_numpy()
            mask = ~np.isnan(y)
            mean_y = float(np.nanmean(y)) if mask.any() else float("nan")
            if mask.sum() < 3 or not mean_y > 0:
                row[f"{c}_rel_change"] = np.nan
            else:
                row[f"{c}_rel_change"] = float(np.polyfit(x[mask], y[mask], 1)[0] * span / mean_y)
        trends.append(row)
    profile = profile.merge(pd.DataFrame(trends), on="club_id", how="left")
    profile = profile.sort_values("club_id").reset_index(drop=True)
    profile.attrs["window"] = window
    return profile


def five_year_ladder(q: SnapshotQuery, club_id: str, *, end_season: int, years: int = 5) -> list[LadderRow]:
    """The club's ladder row for each of the ``years`` seasons before ``end_season``."""
    out: list[LadderRow] = []
    for season in range(end_season - years, end_season):
        out.extend(r for r in ladder(q, season) if r.club_id == club_id)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_team_season(
    q: SnapshotQuery,
    club_id: str,
    season: int,
    *,
    form_window: int = 5,
    stats: Sequence[str] = ("disposals", "kicks", "handballs", "marks", "goals", "tackles",
                            "clearances", "inside_50s", "contested_possessions"),
    tg: pd.DataFrame | None = None,
) -> TeamSeason:
    from supercoach_via.publish.view_models import ClubRef, TeamSeason

    lad = ladder(q, season)
    names = _club_names(q)
    tg = tg if tg is not None else team_games(q, stats=stats, seasons=[season])
    position = next((r.position for r in lad if r.club_id == club_id), None)
    return TeamSeason(
        club=ClubRef(club_id=club_id, name=names.get(club_id, club_id)),
        season=season,
        ladder=lad,
        ladder_note=LADDER_METHOD,
        position=position,
        form_window=form_window,
        form=team_form(q, club_id, n=form_window),
        fixtures=club_fixtures(q, club_id, season),
        team_stats=team_season_stats(tg, club_id, season, stats=stats),
        leaders=club_season_leaders(q, club_id, season),
        five_year=five_year_ladder(q, club_id, end_season=season),
        heuristics=finals_pathway(q, season).get(club_id, []),
    )
