"""Dataset validation and reconciliation (PLAN section 4.4).

``validate_dataset`` runs read-only SQL checks over a candidate snapshot's fragments and
returns a ``ValidationReport`` whose ``issues`` are ``quality_issues``-shaped rows with
deterministic IDs. Severity is era-aware: a defect in the current season is BLOCKING and
can never be suppressed by a known exception; historical defects are reported (warning /
error) and may be accepted by an explicit exception (rule_id + row_key + reason). The
outcome is PASS only when no open blocking issue exists. Validation never upgrades a
legacy import to "verified" -- that needs source verification (refresh).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from supercoach_via.domain.schemas import (
    PLAYER_STAT_COLUMNS,
    TABLES,
    CheckOutcome,
    DateQuality,
    MatchStatus,
    Severity,
    SnapshotManifest,
    StageType,
    ValidationReport,
)
from supercoach_via.ingest.legacy import REPO_CONFIG_DIR, DatasetCandidate
from supercoach_via.storage.queries import SnapshotQuery

__all__ = [
    "CoveragePolicy",
    "KnownException",
    "ValidationPolicy",
    "ValidationReport",
    "dedupe_keys",
    "load_legacy_coverage",
    "load_policy",
    "validate_dataset",
]

#: Quarantine rows resolved by a later verified repair keep their evidence with this prefix.
RESOLVED_PREFIX = "resolved_by_repair:"
CORE_TABLES = ("players", "clubs", "matches", "player_games", "quality_issues", "quarantine", "source_files")
#: Rules whose per-row issues are emitted individually up to this count per rule/season.
ROW_ISSUE_CAP = 200


@dataclass(frozen=True)
class KnownException:
    rule_id: str
    row_key: str
    reason: str


@dataclass(frozen=True)
class Era:
    name: str
    start: int
    end: int | None


@dataclass(frozen=True)
class CoveragePolicy:
    recorded_from: dict[str, int]
    legacy_names: dict[str, str]
    eras: tuple[Era, ...]
    isolated_fragments: dict[str, tuple[tuple[int, int], ...]]
    policy_version: str


@dataclass(frozen=True)
class ValidationPolicy:
    coverage: CoveragePolicy
    current_season: int | None = None  # None -> latest season with a complete match
    known_exceptions: tuple[KnownException, ...] = ()

    def with_exceptions(self, extra: Iterable[KnownException]) -> ValidationPolicy:
        return replace(self, known_exceptions=(*self.known_exceptions, *extra))


def load_legacy_coverage(path: Path) -> dict[str, int]:
    """Legacy ``config/stat_coverage_eras.yaml`` -> {legacy stat: recorded_from}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {k: int(v["recorded_from"]) for k, v in data["stats"].items()}


def load_policy(
    config_dir: Path | None = None,
    *,
    current_season: int | None = None,
    coverage_path: Path | None = None,
) -> ValidationPolicy:
    path = coverage_path or (config_dir or REPO_CONFIG_DIR) / "coverage.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    stats = data["stats"]
    unknown = set(stats) - set(PLAYER_STAT_COLUMNS)
    if unknown:
        raise ValueError(f"coverage.yaml names non-canonical stats: {sorted(unknown)}")
    exceptions: list[KnownException] = []
    for e in data.get("known_exceptions") or []:
        if not isinstance(e, dict) or not all(e.get(k) for k in ("rule_id", "row_key", "reason")):
            raise ValueError(f"known exception needs rule_id, row_key and reason: {e!r}")
        exceptions.append(KnownException(str(e["rule_id"]), str(e["row_key"]), str(e["reason"])))
    cov = CoveragePolicy(
        recorded_from={k: int(v["recorded_from"]) for k, v in stats.items()},
        legacy_names={k: str(v.get("legacy_name", k)) for k, v in stats.items()},
        eras=tuple(
            Era(str(e["name"]), int(e["from"]), None if e["to"] is None else int(e["to"])) for e in data["eras"]
        ),
        isolated_fragments={
            k: tuple((int(a), int(b)) for a, b in v.get("isolated_fragments", [])) for k, v in stats.items()
        },
        policy_version=str(data.get("policy_version", "unknown")),
    )
    return ValidationPolicy(coverage=cov, current_season=current_season, known_exceptions=tuple(exceptions))


def dedupe_keys(rows: Sequence[tuple[str, str, bool, int]]) -> list[tuple[str, str, bool, int]]:
    """Canonical duplicate rule for (match, player, fixture_verified, source_row) rows:
    keep one row per (match, player), preferring a fixture-verified date, then the
    earliest source row. Idempotent; output sorted by key."""
    best: dict[tuple[str, str], tuple[str, str, bool, int]] = {}
    for r in rows:
        k = (r[0], r[1])
        cur = best.get(k)
        if cur is None or (not r[2], r[3]) < (not cur[2], cur[3]):
            best[k] = r
    return [best[k] for k in sorted(best)]


# ---------------------------------------------------------------------------
# Issue collection
# ---------------------------------------------------------------------------


class _Collector:
    def __init__(self, policy: ValidationPolicy, current: int | None):
        self.policy = policy
        self.current = current
        self.issues: dict[str, dict[str, Any]] = {}
        self.exceptions = {(e.rule_id, e.row_key): e for e in policy.known_exceptions}
        self.failed_checks: set[str] = set()

    def add(
        self,
        check: str,
        rule_id: str,
        base: Severity,
        explanation: str,
        *,
        table: str | None = None,
        row_key: str | None = None,
        season: int | None = None,
        current_blocks: bool = True,
        remediation: str | None = None,
    ) -> None:
        severity = base
        if current_blocks and season is not None and season == self.current and base is not Severity.INFO:
            severity = Severity.BLOCKING
        status, basis = "open", None
        exc = self.exceptions.get((rule_id, row_key or ""))
        if exc is not None:
            if season is not None and season == self.current:
                self._raw(
                    "exceptions",
                    "exception_rejected_current_season",
                    Severity.BLOCKING,
                    f"known exception for {rule_id} {row_key} rejected: current-season defects cannot be suppressed",
                    table,
                    row_key,
                    season,
                    None,
                )
            else:
                status, basis = "accepted", exc.reason
        self._raw(check, rule_id, severity, explanation, table, row_key, season, remediation, status, basis)

    def _raw(
        self,
        check: str,
        rule_id: str,
        severity: Severity,
        explanation: str,
        table: str | None,
        row_key: str | None,
        season: int | None,
        remediation: str | None,
        status: str = "open",
        basis: str | None = None,
    ) -> None:
        digest = hashlib.sha256(f"validate\x1f{rule_id}\x1f{table}\x1f{row_key}\x1f{explanation}".encode()).hexdigest()
        issue_id = f"qv:{digest[:24]}"
        self.issues[issue_id] = {
            "issue_id": issue_id,
            "severity": severity.value,
            "status": status,
            "table_name": table,
            "row_key": row_key,
            "source_path": None,
            "rule_id": rule_id,
            "explanation": explanation,
            "remediation": remediation,
            "acceptance_basis": basis,
            "season": season,
        }
        if severity is Severity.BLOCKING and status != "accepted":
            self.failed_checks.add(check)

    def rows(
        self,
        check: str,
        rule_id: str,
        base: Severity,
        table: str,
        hits: list[tuple[Any, ...]],
        describe: str,
        *,
        remediation: str | None = None,
        current_blocks: bool = True,
    ) -> int:
        """hits: (season, row_key, detail). Emit per-row issues (capped per historical season)."""
        per_season: dict[Any, int] = {}
        overflow: dict[Any, int] = {}
        for season, key, detail in hits:
            n = per_season.get(season, 0)
            per_season[season] = n + 1
            if season == self.current or n < ROW_ISSUE_CAP:
                self.add(
                    check,
                    rule_id,
                    base,
                    f"{describe}: {detail}",
                    table=table,
                    row_key=str(key),
                    season=season,
                    current_blocks=current_blocks,
                    remediation=remediation,
                )
            else:
                overflow[season] = overflow.get(season, 0) + 1
        for season, extra in sorted(overflow.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            self.add(
                check,
                rule_id,
                base,
                f"{describe}: {extra} further rows in {season} (first {ROW_ISSUE_CAP} listed individually)",
                table=table,
                row_key=f"season:{season}:overflow",
                season=season,
                current_blocks=current_blocks,
            )
        return len(hits)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset(candidate: DatasetCandidate, policy: ValidationPolicy | None = None) -> ValidationReport:
    import pyarrow.parquet as pq

    policy = policy or load_policy()
    manifest = candidate.candidate.manifest
    root = candidate.data_root
    counts: dict[str, int] = {}
    checks: dict[str, CheckOutcome] = {}

    with SnapshotQuery(root, manifest) as q:
        current = policy.current_season
        if current is None and "matches" in manifest.tables:
            current = q.scalar("SELECT max(season) FROM matches WHERE status = ?", [MatchStatus.COMPLETE.value])
        col = _Collector(policy, current)
        counts["current_season"] = int(current) if current is not None else -1

        # -- schema -------------------------------------------------------------
        for name in CORE_TABLES:
            if name not in manifest.tables:
                col.add("schema", "table_missing", Severity.BLOCKING, f"core table {name} missing", table=name)
        for name, entry in sorted(manifest.tables.items()):
            counts[f"rows:{name}"] = entry.row_count
            spec = TABLES.get(name)
            if spec is None:
                col.add("schema", "table_unknown", Severity.BLOCKING, f"table {name} not in TABLES", table=name)
                continue
            expected = spec.arrow_schema()
            for frag in entry.fragments:
                got = pq.read_schema(root / "fragments" / frag.path)
                if got.names != expected.names or [f.type for f in got] != [f.type for f in expected]:
                    col.add("schema", "schema_mismatch", Severity.BLOCKING, f"{name} fragment {frag.path}", table=name)
        # -- keys and non-null columns -----------------------------------------
        for name, entry in sorted(manifest.tables.items()):
            spec = TABLES.get(name)
            if spec is None or entry.row_count == 0:
                continue
            key = ", ".join(spec.key)
            dup = q.scalar(f"SELECT count(*) FROM (SELECT {key} FROM {name} GROUP BY {key} HAVING count(*) > 1)")  # noqa: S608
            if dup:
                col.add("keys", "key_duplicate", Severity.BLOCKING, f"{dup} duplicate keys ({key})", table=name)
            required = [c.name for c in spec.columns if not c.nullable]
            if required:
                sel = ", ".join(f"count(*) FILTER (WHERE {c} IS NULL)" for c in required)
                null_counts = q.rows(f"SELECT {sel} FROM {name}")[0]  # noqa: S608
                for cname, nulls in zip(required, null_counts, strict=True):
                    if nulls:
                        col.add(
                            "keys", "non_null_violation", Severity.BLOCKING, f"{nulls} nulls in {cname}", table=name
                        )
        # -- enums ---------------------------------------------------------------
        enum_checks = [
            ("matches", "stage_type", [s.value for s in StageType]),
            ("matches", "status", [s.value for s in MatchStatus]),
            ("player_games", "date_quality", [s.value for s in DateQuality]),
        ]
        for table, column, allowed in enum_checks:
            if table in manifest.tables:
                bad = q.scalar(
                    f"SELECT count(*) FROM {table} WHERE {column} NOT IN ({','.join('?' * len(allowed))})",  # noqa: S608
                    allowed,
                )
                if bad:
                    col.add("keys", "enum_violation", Severity.BLOCKING, f"{bad} bad {column}", table=table)
        # -- foreign keys ----------------------------------------------------------
        fks = [
            ("player_games", "match_id", "matches", "match_id"),
            ("player_games", "player_id", "players", "player_id"),
            ("player_games", "club_id", "clubs", "club_id"),
            ("player_games", "opponent_club_id", "clubs", "club_id"),
            ("matches", "home_club_id", "clubs", "club_id"),
            ("matches", "away_club_id", "clubs", "club_id"),
            ("matches", "venue_id", "venues", "venue_id"),
            ("lineups", "match_id", "matches", "match_id"),
            ("lineups", "player_id", "players", "player_id"),
            ("lineups", "club_id", "clubs", "club_id"),
            ("players", "canonical_player_id", "players", "player_id"),
            ("club_aliases", "club_id", "clubs", "club_id"),
            ("draft_events", "player_id", "players", "player_id"),
            ("legacy_predictions", "player_id", "players", "player_id"),
            ("legacy_rank_scores", "player_id", "players", "player_id"),
            ("live_snapshots", "match_id", "matches", "match_id"),
        ]
        for child, ccol, parent, pcol in fks:
            if child not in manifest.tables or manifest.tables[child].row_count == 0:
                continue
            if parent not in manifest.tables:
                col.add(
                    "foreign_keys", "fk_parent_missing", Severity.BLOCKING, f"{child}.{ccol} -> {parent}", table=child
                )
                continue
            orphans = q.scalar(
                f"SELECT count(*) FROM {child} c WHERE c.{ccol} IS NOT NULL "  # noqa: S608
                f"AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.{pcol} = c.{ccol})"
            )
            if orphans:
                col.add(
                    "foreign_keys",
                    "fk_orphan",
                    Severity.BLOCKING,
                    f"{orphans} {child}.{ccol} not in {parent}",
                    table=child,
                )
        if {"player_games", "players"} <= manifest.tables.keys():
            n = q.scalar(
                "SELECT count(*) FROM player_games g JOIN players p USING (player_id) "
                "WHERE p.identity_status <> 'canonical'"
            )
            if n:
                col.add(
                    "foreign_keys",
                    "fact_on_noncanonical_identity",
                    Severity.BLOCKING,
                    f"{n} rows",
                    table="player_games",
                )
        if {"player_games", "matches"} <= manifest.tables.keys():
            hits = q.rows(
                "SELECT g.season, g.match_id || '|' || g.player_id, "
                "g.club_id || ' v ' || coalesce(g.opponent_club_id, '?') "
                "FROM player_games g JOIN matches m USING (match_id) "
                "WHERE NOT ((g.club_id = m.home_club_id AND g.opponent_club_id = m.away_club_id) "
                "OR (g.club_id = m.away_club_id AND g.opponent_club_id = m.home_club_id))"
            )
            col.rows(
                "foreign_keys",
                "player_game_club_not_in_match",
                Severity.ERROR,
                "player_games",
                hits,
                "club/opponent disagree with match",
            )

        if "matches" in manifest.tables and manifest.tables["matches"].row_count:
            _match_checks(q, col)
        if "player_games" in manifest.tables and manifest.tables["player_games"].row_count:
            _player_checks(q, col, policy, counts)
        _import_issue_checks(q, col, manifest.tables.keys())

    for name in (
        "schema",
        "keys",
        "foreign_keys",
        "stages",
        "scores",
        "player_stats",
        "reconciliation",
        "coverage",
        "imports",
        "exceptions",
    ):
        checks[name] = CheckOutcome.FAIL if name in col.failed_checks else CheckOutcome.PASS
    issues = [col.issues[k] for k in sorted(col.issues)]
    for sev in Severity:
        counts[f"issues:{sev.value}"] = sum(1 for i in issues if i["severity"] == sev.value)
    blocking = any(i["severity"] == Severity.BLOCKING.value and i["status"] != "accepted" for i in issues)
    return ValidationReport(
        outcome=CheckOutcome.FAIL if blocking else CheckOutcome.PASS, checks=checks, issues=issues, counts=counts
    )


def _match_checks(q: SnapshotQuery, col: _Collector) -> None:
    hits = q.rows(
        "SELECT season, match_id, stage_label FROM matches WHERE stage_type = 'other' ORDER BY season, match_id"
    )
    col.rows("stages", "stage_unrecognized", Severity.ERROR, "matches", hits, "unrecognized source stage")
    hits = q.rows(
        "SELECT season, match_id, 'status=complete without final scores' FROM matches WHERE status = 'complete' AND "
        "(home_final_goals IS NULL OR home_final_behinds IS NULL "
        "OR away_final_goals IS NULL OR away_final_behinds IS NULL)"
    )
    col.rows("scores", "complete_without_scores", Severity.BLOCKING, "matches", hits, "completion")
    hits = q.rows(
        "SELECT season, match_id, 'score != 6*goals+behinds' FROM matches WHERE "
        "home_score IS DISTINCT FROM home_final_goals*6 + home_final_behinds OR "
        "away_score IS DISTINCT FROM away_final_goals*6 + away_final_behinds"
    )
    col.rows("scores", "score_arithmetic", Severity.BLOCKING, "matches", hits, "score arithmetic")
    conds = []
    for side in ("home", "away"):
        for k in ("goals", "behinds"):
            conds.append(
                f"{side}_q1_{k} > {side}_q2_{k} OR {side}_q2_{k} > {side}_q3_{k} OR {side}_q3_{k} > {side}_final_{k}"
            )
    hits = q.rows(
        "SELECT season, match_id, concat_ws(' ', home_q1_goals, home_q2_goals, home_q3_goals, home_final_goals, '|', "  # noqa: S608
        "away_q1_goals, away_q2_goals, away_q3_goals, away_final_goals) FROM matches WHERE "
        + " OR ".join(f"({c})" for c in conds)
        + " ORDER BY season, match_id"
    )
    col.rows(
        "scores", "quarter_scores_decrease", Severity.WARNING, "matches", hits, "cumulative quarter scores decrease"
    )
    hits = q.rows(
        "SELECT season, match_id, 'negative score cell' FROM matches WHERE least("  # noqa: S608
        + ", ".join(
            f"coalesce({s}_{p}_{k}, 0)"
            for s in ("home", "away")
            for p in ("q1", "q2", "q3", "final")
            for k in ("goals", "behinds")
        )
        + ") < 0"
    )
    col.rows("scores", "score_negative", Severity.ERROR, "matches", hits, "range")


def _player_checks(q: SnapshotQuery, col: _Collector, policy: ValidationPolicy, counts: dict[str, int]) -> None:
    # ranges
    neg = " OR ".join(f"{s} < 0" for s in PLAYER_STAT_COLUMNS)
    hits = q.rows(f"SELECT season, match_id || '|' || player_id, 'negative stat' FROM player_games WHERE {neg}")  # noqa: S608
    col.rows("player_stats", "stat_negative", Severity.ERROR, "player_games", hits, "range")
    hits = q.rows(
        "SELECT season, match_id || '|' || player_id, time_on_ground_pct FROM player_games "
        "WHERE time_on_ground_pct > 100"
    )
    col.rows("player_stats", "time_on_ground_over_100", Severity.WARNING, "player_games", hits, "time on ground %")
    # disposals = kicks + handballs, only where all three are observed
    hits = q.rows(
        "SELECT season, match_id || '|' || player_id, disposals || ' != ' || kicks || '+' || handballs "
        "FROM player_games WHERE disposals IS NOT NULL AND kicks IS NOT NULL AND handballs IS NOT NULL "
        "AND disposals <> kicks + handballs "
        "ORDER BY season, 2"
    )
    counts["disposals_arithmetic_violations"] = col.rows(
        "player_stats", "disposals_arithmetic", Severity.WARNING, "player_games", hits, "disposals != kicks + handballs"
    )
    # match vs player goals where both observed (player goals recorded from coverage policy)
    goals_from = policy.coverage.recorded_from.get("goals", 1897)
    hits = q.rows(
        f"""
        WITH pg AS (
          SELECT match_id, club_id, count(*) AS n, count(goals) AS n_obs, sum(goals) AS g
          FROM player_games WHERE season >= {int(goals_from)} GROUP BY match_id, club_id)
        SELECT m.season, m.match_id || '|' || s.club, 'match ' || s.final || ' vs players ' || coalesce(pg.g, 0)
               || ' (' || pg.n || ' player rows)'
        FROM matches m
        CROSS JOIN LATERAL (VALUES (m.home_club_id, m.home_final_goals),
                                   (m.away_club_id, m.away_final_goals)) s(club, final)
        JOIN pg ON pg.match_id = m.match_id AND pg.club_id = s.club
        WHERE m.status = 'complete' AND pg.n_obs > 0 AND s.final IS NOT NULL AND coalesce(pg.g, 0) <> s.final
        ORDER BY 1, 2
        """  # noqa: S608
    )
    counts["match_goal_mismatches"] = col.rows(
        "reconciliation", "match_player_goals_mismatch", Severity.WARNING, "matches", hits, "goal totals disagree"
    )
    hits = q.rows(
        """
        SELECT m.season, m.match_id || '|' || s.club, 'no player rows for club in complete match'
        FROM matches m
        CROSS JOIN LATERAL (VALUES (m.home_club_id), (m.away_club_id)) s(club)
        WHERE m.status = 'complete' AND NOT EXISTS (
          SELECT 1 FROM player_games g WHERE g.match_id = m.match_id AND g.club_id = s.club)
        ORDER BY 1, 2
        """
    )
    counts["match_clubs_without_player_rows"] = col.rows(
        "reconciliation", "match_players_missing", Severity.WARNING, "matches", hits, "participation"
    )
    # career counter vs observed rows (expose both, never invent rows)
    exceeds, below = q.rows(
        "SELECT count(*) FILTER (WHERE mx > n), count(*) FILTER (WHERE mx < n) FROM ("
        "SELECT player_id, max(career_game_counter) AS mx, count(*) AS n FROM player_games GROUP BY player_id)"
    )[0]
    counts["players_counter_exceeds_rows"] = int(exceeds)
    counts["players_counter_below_rows"] = int(below)
    if exceeds or below:
        col.add(
            "reconciliation",
            "career_counter_vs_rows",
            Severity.INFO,
            f"{exceeds} players whose max career counter exceeds observed rows; {below} below "
            "(both exposed, no rows invented)",
            table="player_games",
            row_key="corpus",
        )
    # counter gaps inside the current season (a missing current-season row is a defect)
    if col.current is not None:
        hits = q.rows(
            f"""
            SELECT season, player_id, prev || ' -> ' || career_game_counter FROM (
              SELECT season, player_id, career_game_counter,
                     lag(career_game_counter) OVER (PARTITION BY player_id ORDER BY career_game_counter) AS prev
              FROM player_games WHERE career_game_counter IS NOT NULL AND season >= {int(col.current) - 1})
            WHERE season = {int(col.current)} AND prev IS NOT NULL AND career_game_counter - prev > 1
            ORDER BY 2
            """  # noqa: S608
        )
        counts["current_season_counter_gaps"] = col.rows(
            "reconciliation", "career_counter_gap", Severity.WARNING, "player_games", hits, "career counter gap"
        )
    # coverage by era (reporting; values before recorded_from are flagged, never zero-filled)
    stats = sorted(policy.coverage.recorded_from)
    obs_sel = ", ".join(f"count({s})" for s in stats)
    for era in policy.coverage.eras:
        hi = era.end if era.end is not None else 9999
        n, *obs = q.rows(
            f"SELECT count(*), {obs_sel} FROM player_games WHERE season BETWEEN {era.start} AND {hi}"  # noqa: S608
        )[0]
        for stat, o in zip(stats, obs, strict=True):
            counts[f"coverage:{stat}:{era.name}:rows"] = int(n)
            counts[f"coverage:{stat}:{era.name}:observed"] = int(o)
    early_sel = []
    for stat in stats:
        rec_from = policy.coverage.recorded_from[stat]
        excl = "".join(
            f" AND NOT (season BETWEEN {a} AND {b})" for a, b in policy.coverage.isolated_fragments.get(stat, ())
        )
        early_sel.append(f"count(*) FILTER (WHERE season < {rec_from} AND {stat} IS NOT NULL{excl})")
    max_rec = max(policy.coverage.recorded_from.values(), default=0)
    early_rows = q.rows(
        f"SELECT season, {', '.join(early_sel)} FROM player_games WHERE season < {max_rec} "  # noqa: S608
        "GROUP BY season ORDER BY season"
    )
    for season_, *cnts in early_rows:
        for stat, cnt in zip(stats, cnts, strict=True):
            if cnt:
                col.add(
                    "coverage",
                    "stat_before_recorded_from",
                    Severity.INFO,
                    f"{cnt} observed {stat} values before recorded_from={policy.coverage.recorded_from[stat]}",
                    table="player_games",
                    row_key=f"{stat}:{season_}",
                    season=int(season_),
                    current_blocks=False,
                )


def _import_issue_checks(q: SnapshotQuery, col: _Collector, tables: Iterable[str]) -> None:
    """Escalate import-time defects in the current season; quarantined current rows block."""
    tables = set(tables)
    if col.current is None:
        return
    if "quality_issues" in tables:
        for rule_id, row_key, season, expl, sev in q.rows(
            "SELECT rule_id, row_key, season, explanation, severity FROM quality_issues "
            "WHERE season = ? AND severity IN ('error', 'blocking') ORDER BY issue_id",
            [col.current],
        ):
            if rule_id == "stage_unrecognized":
                continue  # re-derived from the matches table above
            col.add(
                "imports",
                rule_id,
                Severity(sev),
                f"import: {expl}",
                table="quality_issues",
                row_key=row_key,
                season=season,
            )
    if "quarantine" in tables:
        verified_sql = (
            "EXISTS (SELECT 1 FROM lineups l WHERE l.source_path = q.source_path "
            "AND l.source_row = q.source_row AND l.name_token = json_extract_string(q.raw, '$.token'))"
            if "lineups" in tables
            else "false"
        )
        sql = (
            f"SELECT q.table_name, q.reason, q.source_path, q.source_row, q.season, {verified_sql} "  # noqa: S608
            "FROM quarantine q WHERE q.season = ? AND q.reason <> 'duplicate_identity' ORDER BY q.quarantine_id"
        )
        for table, reason, path, row, season, verified in q.rows(
            sql,
            [col.current],
        ):
            if str(reason).startswith(RESOLVED_PREFIX):
                if table == "lineups" and verified:
                    continue  # re-linked after a verified repair; the lineup row exists
                col.add(
                    "imports",
                    "quarantine_resolution_unverified",
                    Severity.BLOCKING,
                    f"{table} row marked {reason} but no matching {table} row exists",
                    table=table,
                    row_key=f"{path}:{row}",
                    season=season,
                )
                continue
            col.add(
                "imports",
                "current_season_row_quarantined",
                Severity.ERROR,
                f"{table} row quarantined in the current season: {reason}",
                table=table,
                row_key=f"{path}:{row}",
                season=season,
            )


def relink_quarantined_lineups(
    data_root: Path, manifest: SnapshotManifest, *, season: int
) -> dict[str, list[dict[str, Any]]]:
    """Upserts linking ``season``'s unresolved lineup tokens to newly present participants.

    Uses the importer's pass-1 rule only: the token's normalized name must equal exactly one
    player with a ``player_games`` row for that lineup row's match and club who has no lineup
    row there yet. The match/club come from the tokens of the same source row that did link.
    Resolved quarantine rows keep their raw evidence under ``RESOLVED_PREFIX + reason``.
    """
    import json

    from supercoach_via.domain.ids import normalize_name

    out: dict[str, list[dict[str, Any]]] = {"lineups": [], "quarantine": []}
    need = {"players", "player_games", "lineups", "quarantine"}
    if not need <= set(manifest.tables):
        return out
    part = {"player_games": {str(season)}, "lineups": {str(season)}}
    with SnapshotQuery(data_root, manifest, tables=need, partitions=part) as q:
        pending = q.arrow(
            "SELECT * FROM quarantine WHERE season = ? AND table_name = 'lineups' "
            "AND reason = 'lineup_token_unresolved' ORDER BY quarantine_id",
            [season],
        ).to_pylist()
        for qrow in pending:
            token = str(json.loads(qrow["raw"]).get("token") or "")
            sib = q.rows(
                "SELECT DISTINCT match_id, club_id, confidence FROM lineups WHERE source_path = ? AND source_row = ?",
                [qrow["source_path"], qrow["source_row"]],
            )
            if len(sib) != 1 or not token:
                continue
            match_id, club_id, _conf = sib[0]
            participants = q.rows(
                "SELECT g.player_id, p.display_name FROM player_games g JOIN players p USING (player_id) "
                "WHERE g.match_id = ? AND g.club_id = ? AND NOT EXISTS (SELECT 1 FROM lineups l "
                "WHERE l.match_id = g.match_id AND l.club_id = g.club_id AND l.player_id = g.player_id)",
                [match_id, club_id],
            )
            want = normalize_name(token)
            hits = sorted(pid for pid, name in participants if normalize_name(str(name)) == want)
            if len(hits) != 1:
                continue
            out["lineups"].append(
                {c: None for c in TABLES["lineups"].column_names}
                | {"match_id": match_id, "club_id": club_id, "player_id": hits[0], "season": season,
                   "role": "played", "confidence": "high", "name_token": token,
                   "resolution": "match_participation", "provenance": qrow["provenance"],
                   "source_path": qrow["source_path"], "source_sha256": qrow["source_sha256"],
                   "source_row": qrow["source_row"]}
            )  # fmt: skip
            out["quarantine"].append({**qrow, "reason": RESOLVED_PREFIX + str(qrow["reason"])})
    return out
