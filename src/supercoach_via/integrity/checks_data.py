"""Families B/C: the promotion gate's verdict, relationships, season aggregates, football rules.

``validate_dataset`` is reused as-is (it is the promotion gate); its issues are mapped
into findings with their own severity and acceptance. The other rules here cover what that
gate does not check, and recompute every invariant from the fact rows.
"""

from __future__ import annotations

from typing import Any

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS, Severity
from supercoach_via.integrity.context import AuditContext, CheckSkipped, CheckSpec, rule
from supercoach_via.integrity.report import Kind, Status

B, E, W, INFO = Severity.BLOCKING, Severity.ERROR, Severity.WARNING, Severity.INFO

#: validate_dataset rule ids -> default severity (the issue's own severity is what counts)
VALIDATE_RULES: dict[str, Severity] = {
    "table_missing": B,
    "table_unknown": B,
    "schema_mismatch": B,
    "key_duplicate": B,
    "non_null_violation": B,
    "enum_violation": B,
    "fk_parent_missing": B,
    "fk_orphan": B,
    "fact_on_noncanonical_identity": B,
    "player_game_club_not_in_match": E,
    "stage_unrecognized": E,
    "complete_without_scores": B,
    "score_arithmetic": B,
    "quarter_scores_decrease": W,
    "score_negative": E,
    "stat_negative": E,
    "time_on_ground_over_100": W,
    "disposals_arithmetic": W,
    "match_player_goals_mismatch": W,
    "match_players_missing": W,
    "career_counter_vs_rows": INFO,
    "career_counter_gap": W,
    "stat_before_recorded_from": INFO,
    "exception_rejected_current_season": B,
    "quarantine_resolution_unverified": B,
    "current_season_row_quarantined": E,
}
#: counts are never zero; a blank on the source means zero there
NO_ZERO_EXPECTED = frozenset({"time_on_ground_pct"})

RULES = [
    *(
        rule(
            f"dataset.{rid}",
            "dataset.validate",
            sev,
            f"validate_dataset rule {rid} (the promotion gate)",
            "read the finding; repair the input or add a documented historical exception to coverage.yaml",
            kind=Kind.ANOMALY,
        )
        for rid, sev in sorted(VALIDATE_RULES.items())
    ),
    rule(
        "dataset.import_issue",
        "dataset.validate",
        E,
        "an import-time quality issue escalated by validate_dataset",
        "repair the current-season input row named in the finding",
        kind=Kind.ANOMALY,
    ),
    rule(
        "relations.player_in_both_sides",
        "relations.membership",
        B,
        "one player has fact rows for both clubs of a match",
        "fix the identity link at import",
    ),
    rule(
        "relations.season_mismatch",
        "relations.membership",
        B,
        "a player-game's season differs from its match's season",
        "re-import; the fact is in the wrong partition",
    ),
    rule(
        "relations.stage_mismatch",
        "relations.membership",
        B,
        "a player-game's stage differs from its match's stage",
        "re-link the player row to the right match",
    ),
    rule(
        "relations.verified_date_mismatch",
        "relations.membership",
        B,
        "a fixture_verified player-game date differs from its match date",
        "re-link the row or downgrade its date quality; a verified date cannot disagree",
    ),
    rule(
        "relations.result_mismatch",
        "relations.membership",
        E,
        "a player-game's W/L/D disagrees with the match score",
        "re-link the row or correct the source result",
        kind=Kind.ANOMALY,
        current_blocks=True,
    ),
    rule(
        "relations.lineup_not_participant",
        "relations.membership",
        E,
        "a lineup row names a club that did not play the match",
        "re-link the lineup row",
        kind=Kind.ANOMALY,
        current_blocks=True,
    ),
    rule(
        "relations.alias_target",
        "relations.identity",
        B,
        "an alias or duplicate identity does not resolve to one canonical identity (missing, chained or cyclic)",
        "repair the identity registry so every alias points at a canonical player",
    ),
    rule(
        "relations.canonical_self_link",
        "relations.identity",
        B,
        "a canonical identity points at a different canonical_player_id",
        "clear canonical_player_id or mark it as an alias",
    ),
    rule(
        "relations.alias_ambiguous",
        "relations.identity",
        W,
        "one alias string names more than one player",
        "disambiguate the alias with evidence",
        kind=Kind.ANOMALY,
    ),
    rule(
        "aggregates.season_row",
        "aggregates.seasons",
        B,
        "a season with matches has no seasons row, or a seasons row has no matches",
        "rebuild season aggregates from the match rows",
    ),
    rule(
        "aggregates.season_value",
        "aggregates.seasons",
        B,
        "a seasons row disagrees with values recomputed from its accepted match rows",
        "rebuild season aggregates; a stale aggregate misstates the season",
    ),
    rule(
        "football.player_behinds_exceed_team",
        "football.arithmetic",
        E,
        "players' behinds sum to more than the team's behinds (rushed behinds only add to the team)",
        "re-check the match's player rows against the source",
        kind=Kind.ANOMALY,
        current_blocks=True,
    ),
    rule(
        "football.brownlow_range",
        "football.arithmetic",
        E,
        "Brownlow votes outside 0-3 in one game",
        "re-import the row",
        kind=Kind.ANOMALY,
        current_blocks=True,
    ),
    rule(
        "football.unusual_value",
        "football.values",
        W,
        "a value above the policy's plausible maximum (valid but unusual)",
        "confirm against the source; raise plausible_max only with evidence",
        kind=Kind.ANOMALY,
    ),
    rule(
        "football.zero_before_recorded",
        "football.coverage",
        E,
        "zeros stored before the stat's recorded_from season: a missing value was probably zero-filled",
        "re-import with null for unrecorded eras, or document the isolated fragment",
        kind=Kind.ANOMALY,
    ),
    rule(
        "football.blank_as_null",
        "football.coverage",
        W,
        "a statistic is null for a player who took the field although the match reports that statistic "
        "(AFL Tables prints 0 as a blank)",
        "import with domain.blanks or run scvia apply-corrections; observed-denominator means are inflated",
        kind=Kind.ANOMALY,
        current_blocks=True,
    ),
    rule(
        "football.unevidenced_zero",
        "football.coverage",
        W,
        "a statistic is 0 for every player with a value in a match, so nothing shows the match reported it",
        "confirm against the source; an unreported column must stay null, not become 0",
        kind=Kind.ANOMALY,
    ),
    rule(
        "football.brownlow_not_applicable",
        "football.coverage",
        E,
        "a finals row has a Brownlow value; votes are awarded only in home-and-away matches",
        "re-import the row with Brownlow votes null for finals",
        kind=Kind.ANOMALY,
        current_blocks=True,
    ),
]


def _coverage(ctx: AuditContext) -> Any:
    """The coverage/era policy (config/coverage.yaml), parsed once per audit."""
    from supercoach_via.ingest.reconcile import load_policy

    if "_coverage_policy" not in ctx.coverage:
        ctx.coverage["_coverage_policy"] = load_policy(ctx.policy.config_dir)
    return ctx.coverage["_coverage_policy"]


# ---------------------------------------------------------------------------
# dataset.validate: the promotion gate, reused
# ---------------------------------------------------------------------------


def check_validate(ctx: AuditContext) -> list[str]:
    from supercoach_via.ingest.legacy import DatasetCandidate
    from supercoach_via.ingest.reconcile import validate_dataset
    from supercoach_via.storage.snapshots import SnapshotCandidate, snapshot_hex

    snap = ctx.snapshot
    if snap is None or snap.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no parseable manifest")
    if not all(f.verified for f in snap.fragments):
        raise CheckSkipped(Status.UNKNOWN, "fragments failed verification; the gate would read unverified bytes")
    m = snap.manifest
    path = snap.data_root / "snapshots" / f"{snapshot_hex(m.snapshot_id)}.json"
    report = validate_dataset(DatasetCandidate(SnapshotCandidate(m, path), {}, snap.data_root), _coverage(ctx))
    ctx.coverage["validate_dataset"] = {
        "outcome": report.outcome.value,
        "checks": {k: v.value for k, v in sorted(report.checks.items())},
    }
    listed: dict[str, int] = {}
    for issue in report.issues:
        rid = str(issue["rule_id"])
        key = str(issue.get("row_key") or "")
        if key.endswith(":overflow"):
            continue
        name = f"dataset.{rid}" if rid in VALIDATE_RULES else "dataset.import_issue"
        listed[rid] = listed.get(rid, 0) + 1
        ctx.add(
            name,
            key or f"table:{issue.get('table_name')}",
            table=issue.get("table_name"),
            field=None if name != "dataset.import_issue" else rid,
            season=issue.get("season"),
            message=str(issue.get("explanation") or "")[:500],
            severity=Severity(issue["severity"]),
            accepted_by_producer=issue.get("acceptance_basis") if issue.get("status") == "accepted" else None,
        )
    for rid, total in sorted((k.split(":", 1)[1], v) for k, v in report.counts.items() if k.startswith("hits:")):
        sev = next((Severity(i["severity"]) for i in report.issues if i["rule_id"] == rid), VALIDATE_RULES.get(rid, W))
        ctx.collector.count_unlisted(
            f"dataset.{rid}" if rid in VALIDATE_RULES else "dataset.import_issue", total - listed.get(rid, 0), sev
        )
    ctx.count("rows", sum(e.row_count for e in m.tables.values()))
    return []


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------


def check_membership(ctx: AuditContext) -> list[str]:
    ctx.need("matches", "player_games")
    ctx.count("rows", ctx.rows("SELECT count(*) FROM player_games")[0][0])
    pg = "'player_game:' || g.match_id || '|' || g.player_id || '|' || g.club_id"
    base = (
        f"SELECT {pg}, g.season, {{e}}, {{a}} FROM player_games g "  # noqa: S608 - fixed fragments below
        "JOIN matches m USING (match_id) WHERE {cond} ORDER BY 1"
    )
    for rid, e, a, cond, field in (
        ("relations.season_mismatch", "m.season", "g.season", "g.season <> m.season", "season"),
        # stage_label vocabularies differ by design (player-file token vs fixture label); the id is the identity
        ("relations.stage_mismatch", "m.stage_id", "g.stage_id", "g.stage_id <> m.stage_id", "stage_id"),
        (
            "relations.verified_date_mismatch",
            "m.match_date::VARCHAR",
            "g.match_date::VARCHAR",
            "g.date_quality = 'fixture_verified' AND g.match_date IS DISTINCT FROM m.match_date",
            "match_date",
        ),
    ):
        for entity, season, expected, actual in ctx.rows(base.format(e=e, a=a, cond=cond)):
            ctx.add(rid, entity, table="player_games", field=field, season=season, expected=expected, actual=actual)
    for entity, season, expected, actual in ctx.rows(
        f"""SELECT {pg}, g.season, CASE WHEN own > oth THEN 'W' WHEN own < oth THEN 'L' ELSE 'D' END, g.result
        FROM (SELECT g.*, CASE WHEN g.club_id = m.home_club_id THEN m.home_score ELSE m.away_score END AS own,
                     CASE WHEN g.club_id = m.home_club_id THEN m.away_score ELSE m.home_score END AS oth
              FROM player_games g JOIN matches m USING (match_id)
              WHERE m.status = 'complete' AND g.club_id IN (m.home_club_id, m.away_club_id)) g
        WHERE g.result IS NOT NULL AND own IS NOT NULL AND oth IS NOT NULL
          AND g.result <> CASE WHEN own > oth THEN 'W' WHEN own < oth THEN 'L' ELSE 'D' END
        ORDER BY 1"""  # noqa: S608 - fixed SQL
    ):
        ctx.add(
            "relations.result_mismatch",
            entity,
            table="player_games",
            field="result",
            season=season,
            expected=expected,
            actual=actual,
        )
    for match_id, player_id, season, clubs in ctx.rows(
        "SELECT match_id, player_id, min(season), list(DISTINCT club_id ORDER BY club_id) FROM player_games "
        "GROUP BY match_id, player_id HAVING count(DISTINCT club_id) > 1 ORDER BY 1, 2"
    ):
        ctx.add(
            "relations.player_in_both_sides",
            f"player_game:{match_id}|{player_id}",
            table="player_games",
            season=season,
            actual=list(clubs),
        )
    if ctx.has("lineups"):
        ctx.need("lineups")
        for entity, season, club in ctx.rows(
            "SELECT 'lineup:' || l.match_id || '|' || l.club_id || '|' || l.player_id, l.season, l.club_id "
            "FROM lineups l JOIN matches m USING (match_id) WHERE l.club_id NOT IN (m.home_club_id, m.away_club_id) "
            "ORDER BY 1"
        ):
            ctx.add(
                "relations.lineup_not_participant", entity, table="lineups", field="club_id", season=season, actual=club
            )
    return []


def check_identity(ctx: AuditContext) -> list[str]:
    ctx.need("players")
    ctx.count("rows", ctx.rows("SELECT count(*) FROM players")[0][0])
    for pid, target, status in ctx.rows(
        """SELECT p.player_id, p.canonical_player_id, t.identity_status FROM players p
           LEFT JOIN players t ON t.player_id = p.canonical_player_id
           WHERE p.identity_status IN ('alias', 'quarantined_duplicate')
             AND (p.canonical_player_id IS NULL OR t.player_id IS NULL OR t.identity_status <> 'canonical')
           ORDER BY 1"""
    ):
        ctx.add(
            "relations.alias_target",
            f"player:{pid}",
            table="players",
            field="canonical_player_id",
            actual={"target": target, "target_status": status},
        )
    for pid, target in ctx.rows(
        "SELECT player_id, canonical_player_id FROM players WHERE identity_status = 'canonical' "
        "AND canonical_player_id IS NOT NULL AND canonical_player_id <> player_id ORDER BY 1"
    ):
        ctx.add(
            "relations.canonical_self_link",
            f"player:{pid}",
            table="players",
            field="canonical_player_id",
            actual=target,
        )
    if ctx.has("player_aliases"):
        ctx.need("player_aliases")
        for alias, pids in ctx.rows(
            "SELECT alias, list(DISTINCT player_id ORDER BY player_id) FROM player_aliases GROUP BY alias "
            "HAVING count(DISTINCT player_id) > 1 ORDER BY 1"
        ):
            ctx.add("relations.alias_ambiguous", f"alias:{alias}", table="player_aliases", actual=list(pids))
    return []


# ---------------------------------------------------------------------------
# aggregates: seasons recomputed from accepted match rows
# ---------------------------------------------------------------------------


def check_seasons(ctx: AuditContext) -> list[str]:
    ctx.need("matches", "seasons")
    rows = ctx.records(
        """WITH agg AS (
             SELECT season, min(match_date) AS first_match_date, max(match_date) AS last_match_date,
                    count(*) FILTER (WHERE status = 'complete') AS matches_complete,
                    count(*) FILTER (WHERE status = 'scheduled') AS matches_scheduled
             FROM matches GROUP BY season)
           SELECT coalesce(a.season, s.season) AS season, a.season IS NOT NULL AS has_matches,
                  s.season IS NOT NULL AS has_row,
                  a.first_match_date AS e_first, s.first_match_date AS a_first,
                  a.last_match_date AS e_last, s.last_match_date AS a_last,
                  a.matches_complete AS e_complete, s.matches_complete AS a_complete,
                  a.matches_scheduled AS e_scheduled, s.matches_scheduled AS a_scheduled
           FROM agg a FULL OUTER JOIN seasons s USING (season) ORDER BY 1"""
    )
    for r in rows:
        ctx.count("rows")
        season = int(r["season"])
        entity = f"season:{season}"
        if not (r["has_matches"] and r["has_row"]):
            ctx.add(
                "aggregates.season_row",
                entity,
                table="seasons",
                season=season,
                expected="row" if r["has_matches"] else "no row",
                actual="row" if r["has_row"] else "no row",
            )
            continue
        for field, e, a in (
            ("first_match_date", r["e_first"], r["a_first"]),
            ("last_match_date", r["e_last"], r["a_last"]),
            ("matches_complete", r["e_complete"], r["a_complete"]),
            ("matches_scheduled", r["e_scheduled"], r["a_scheduled"]),
        ):
            if e != a:
                ctx.add(
                    "aggregates.season_value",
                    entity,
                    table="seasons",
                    field=field,
                    season=season,
                    expected=None if e is None else (e.isoformat() if hasattr(e, "isoformat") else int(e)),
                    actual=None if a is None else (a.isoformat() if hasattr(a, "isoformat") else int(a)),
                )
    return []


# ---------------------------------------------------------------------------
# football arithmetic, values and coverage
# ---------------------------------------------------------------------------


def check_arithmetic(ctx: AuditContext) -> list[str]:
    ctx.need("matches", "player_games")
    cov = _coverage(ctx).coverage
    bh_from = int(cov.recorded_from.get("behinds", 1897))
    for entity, season, team, players in ctx.rows(
        f"""SELECT 'match:' || m.match_id || '|' || s.club, m.season, s.team_behinds, p.bh
            FROM matches m
            CROSS JOIN LATERAL (VALUES (m.home_club_id, m.home_final_behinds),
                                       (m.away_club_id, m.away_final_behinds)) s(club, team_behinds)
            JOIN (SELECT match_id, club_id, sum(behinds) AS bh FROM player_games
                  WHERE season >= {bh_from} GROUP BY ALL) p ON p.match_id = m.match_id AND p.club_id = s.club
            WHERE m.status = 'complete' AND s.team_behinds IS NOT NULL AND p.bh > s.team_behinds
            ORDER BY 1"""  # noqa: S608 - integer interpolated
    ):
        ctx.add(
            "football.player_behinds_exceed_team",
            entity,
            table="player_games",
            field="behinds",
            season=season,
            expected=f"<= {team}",
            actual=int(players),
        )
    for entity, season, votes in ctx.rows(
        "SELECT 'player_game:' || match_id || '|' || player_id || '|' || club_id, season, brownlow_votes "
        "FROM player_games WHERE brownlow_votes NOT BETWEEN 0 AND 3 ORDER BY 1"
    ):
        ctx.add(
            "football.brownlow_range",
            entity,
            table="player_games",
            field="brownlow_votes",
            season=season,
            expected="0-3",
            actual=votes,
        )
    ctx.count("rows", ctx.rows("SELECT count(*) FROM player_games")[0][0])
    return []


def check_values(ctx: AuditContext) -> list[str]:
    ctx.need("player_games")
    for stat, limit in sorted(ctx.policy.plausible_max.items()):
        for entity, season, value in ctx.rows(
            f"SELECT 'player_game:' || match_id || '|' || player_id || '|' || club_id, season, \"{stat}\" "  # noqa: S608
            f'FROM player_games WHERE "{stat}" > ? ORDER BY 1',
            [limit],
        ):
            ctx.add(
                "football.unusual_value",
                entity,
                table="player_games",
                field=stat,
                season=season,
                expected=limit,
                actual=value,
            )
    ctx.count("rows", ctx.rows("SELECT count(*) FROM player_games")[0][0])
    return []


#: counts every player who takes the field records whenever a match reports them
_UNIVERSAL_EVIDENCE = ("kicks", "marks", "handballs", "disposals", "time_on_ground_pct")


def _blank_semantics(ctx: AuditContext, cov: Any) -> None:
    """The checker's own reading of source blanks (written independently of ``domain.blanks``).

    AFL Tables prints 0 as a blank. Where a match reports a statistic (some row has a value)
    and a row took the field, a null for that statistic contradicts the source. A row took
    the field if it has any value other than Brownlow votes, or if its match reports none of
    the statistics every player on the field records (goals-only eras). Time on ground is never
    zero; Brownlow votes are not awarded in finals.
    """
    played_any = " OR ".join(f'g."{s}" IS NOT NULL' for s in PLAYER_STAT_COLUMNS if s != "brownlow_votes")
    universal = " OR ".join(f'"{s}" IS NOT NULL' for s in _UNIVERSAL_EVIDENCE)
    base = f"""
        WITH mf AS (SELECT match_id, bool_or({universal}) AS universal FROM player_games GROUP BY match_id),
             rows AS (SELECT g.*, m.stage_type, mf.universal, ({played_any}) OR NOT mf.universal AS played
                      FROM player_games g JOIN matches m USING (match_id) JOIN mf USING (match_id))"""  # noqa: S608 - schema column names only
    for stat in PLAYER_STAT_COLUMNS:
        start = cov.recorded_from.get(stat)
        if stat in NO_ZERO_EXPECTED or start is None:
            continue
        final_rule = "AND stage_type <> 'final'" if stat == "brownlow_votes" else ""
        for is_current, cells, matches, lo, hi in ctx.rows(
            base + f""", rep AS (SELECT match_id, count("{stat}") AS n FROM player_games GROUP BY match_id)
            SELECT season = ? AS cur, count(*), count(DISTINCT match_id), min(season), max(season)
            FROM rows JOIN rep USING (match_id)
            WHERE "{stat}" IS NULL AND rep.n > 0 AND played AND season >= {int(start)} {final_rule}
            GROUP BY 1 ORDER BY 1""",  # noqa: S608 - stat names are schema constants
            [ctx.current_season or -1],
        ):
            ctx.count("cells", int(cells))
            entity = f"stat:{stat}:{ctx.current_season}" if is_current else f"stat:{stat}"
            ctx.add("football.blank_as_null", entity, table="player_games", field=stat,
                    season=ctx.current_season if is_current else None, expected=0,
                    actual={"cells": int(cells), "matches": int(matches)},
                    evidence={"seasons": [lo, hi], "recorded_from": start})  # fmt: skip
        for cells, matches in ctx.rows(
            f"""SELECT sum(n0), count(*) FROM (SELECT match_id, count(*) FILTER (WHERE "{stat}" = 0) AS n0,
                       count(*) FILTER (WHERE "{stat}" <> 0) AS nz FROM player_games GROUP BY match_id)
                WHERE n0 > 0 AND nz = 0"""  # noqa: S608 - stat names are schema constants
        ):
            if matches:
                ctx.add("football.unevidenced_zero", f"stat:{stat}", table="player_games", field=stat,
                        expected="a non-zero value somewhere in the match",
                        actual={"zero_cells": int(cells), "matches": int(matches)})  # fmt: skip
    for entity, season, n in ctx.rows(
        "SELECT 'match:' || g.match_id, min(g.season), count(*) FROM player_games g JOIN matches m USING (match_id) "
        "WHERE m.stage_type = 'final' AND g.brownlow_votes IS NOT NULL GROUP BY 1 ORDER BY 1"
    ):
        ctx.add("football.brownlow_not_applicable", entity, table="player_games", field="brownlow_votes",
                season=season, expected=None, actual=int(n))  # fmt: skip


def check_coverage(ctx: AuditContext) -> list[str]:
    ctx.need("player_games", "matches")
    cov = _coverage(ctx).coverage
    for stat in PLAYER_STAT_COLUMNS:
        start = cov.recorded_from.get(stat)
        if start is None:
            continue
        isolated = "".join(
            f" AND NOT (season BETWEEN {int(a)} AND {int(b)})" for a, b in cov.isolated_fragments.get(stat, ())
        )
        for season, zeros in ctx.rows(
            f'SELECT season, count(*) FROM player_games WHERE season < {int(start)} AND "{stat}" = 0{isolated} '  # noqa: S608
            "GROUP BY season ORDER BY season"
        ):
            ctx.add(
                "football.zero_before_recorded",
                f"stat:{stat}:{season}",
                table="player_games",
                field=stat,
                season=season,
                expected=None,
                actual=int(zeros),
                evidence={"recorded_from": start},
            )
    _blank_semantics(ctx, cov)
    return []


CHECKS = [
    CheckSpec(
        "dataset.validate",
        "dataset",
        "the promotion gate (validate_dataset) re-run on the pinned snapshot",
        check_validate,
    ),
    CheckSpec(
        "relations.membership", "relations", "player-game membership, season, stage, date and result", check_membership
    ),
    CheckSpec(
        "relations.identity",
        "relations",
        "alias and duplicate identities resolve to one canonical player",
        check_identity,
    ),
    CheckSpec("aggregates.seasons", "aggregates", "season rows recomputed from accepted match rows", check_seasons),
    CheckSpec("football.arithmetic", "football", "behinds and Brownlow arithmetic", check_arithmetic),
    CheckSpec("football.values", "football", "unusual but valid values", check_values),
    CheckSpec("football.coverage", "football", "era coverage: zero-fill and blank-as-null", check_coverage),
]
