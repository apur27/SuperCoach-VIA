"""Match-level comparison: inventory, mapping and attributes (DESIGN sections 7, 9; T28).

Source matches come from the season pages (the independent inventory) and their match pages;
local matches from the snapshot ``matches`` table or the legacy ``matches_YYYY.csv`` files. A
source match with no local counterpart is a confirmed missing match (this is how a missing latest
final is caught without any local seed); a local match inside the audited scope with no source
counterpart is reported as extra; same-pair finals are told apart by date order and replay ordinal.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from supercoach_via.reconciliation.findings import make_finding
from supercoach_via.reconciliation.identity import LocalMatchRec, MatchMap, SourceMatchHeader, map_matches


@dataclass(frozen=True)
class SourceMatchInfo:
    url: str
    season: int
    stage_text: str
    teams: tuple[str, str]
    date: str | None
    #: team name -> (final goals, final behinds) as printed on the match page
    scores: dict[str, tuple[int, int]]
    sha256: str | None
    usable: bool
    reason: str | None = None


@dataclass(frozen=True)
class LocalMatchInfo:
    rec: LocalMatchRec
    scores: dict[str, tuple[int, int]]
    origin: str
    precision: str = "day"


@dataclass
class MatchResult:
    layer: str
    mapping: MatchMap
    counters: Counter[str] = field(default_factory=Counter)
    findings: list[dict[str, Any]] = field(default_factory=list)


def compare_matches(
    layer: str, local: list[LocalMatchInfo], source: list[SourceMatchInfo], *, through: str
) -> MatchResult:
    """Pair local and source matches and compare date and final scores for the pairs."""
    headers = [
        SourceMatchHeader(s.url, s.season, s.stage_text, frozenset(s.teams), s.date, _drawn(s))
        for s in source
        if s.usable
    ]
    in_scope_local = [m for m in local if m.rec.date is None or m.rec.date[:10] <= through]
    mapping = map_matches([m.rec for m in in_scope_local], headers)
    res = MatchResult(layer, mapping)
    by_url = {s.url: s for s in source}
    by_id = {m.rec.match_id: m for m in in_scope_local}
    res.counters["source_matches_in_scope"] = len(source)
    res.counters["source_matches_usable"] = len(headers)
    res.counters["local_matches_in_scope"] = len(in_scope_local)
    res.counters["local_matches_out_of_scope"] = len(local) - len(in_scope_local)
    res.counters["matches_paired"] = len(mapping.local_to_source)
    for url in mapping.source_only:
        s = by_url[url]
        res.counters["matches_missing_local"] += 1
        res.findings.append(
            make_finding(
                "MATCH_MISSING_LOCAL",
                layer=layer,
                rule_id="R-MATCH-MISSING",
                season=s.season,
                match={"source_url": url},
                evidence={
                    "source_url": url,
                    "body_sha256": s.sha256,
                    "locator": f"{s.stage_text} {' v '.join(s.teams)}",
                },
                detail=f"{s.season} {s.stage_text}: {s.teams[0]} v {s.teams[1]} on {s.date}",
            )
        )
    for mid in mapping.local_only:
        m = by_id[mid]
        res.counters["matches_extra_local"] += 1
        res.findings.append(
            make_finding(
                "MATCH_EXTRA_LOCAL",
                layer=layer,
                rule_id="R-MATCH-EXTRA",
                season=m.rec.season,
                match={"local_id": mid},
                local={"origin": m.origin},
                detail=f"{m.rec.stage_text} {' v '.join(sorted(m.rec.teams))} on {m.rec.date}",
            )
        )
    for mid in mapping.ambiguous:
        m = by_id[mid]
        res.counters["matches_unresolved"] += 1
        res.findings.append(
            make_finding(
                "MATCH_UNRESOLVED",
                layer=layer,
                rule_id="R-MATCH-AMBIGUOUS",
                season=m.rec.season,
                match={"local_id": mid},
                local={"origin": m.origin},
                detail="same-pair finals could not be paired without guessing the replay ordinal",
            )
        )
    for mid, url in sorted(mapping.local_to_source.items()):
        m, s = by_id[mid], by_url[url]
        sd, ld = s.date, (m.rec.date or "")[:10]
        if sd is not None and ld:
            if sd == ld:
                res.counters["match_attr_equal"] += 1
            else:
                res.counters["match_attr_mismatch"] += 1
                res.findings.append(_attr(layer, s, m, "date", sd, ld))
        for team, (g, b) in sorted(s.scores.items()):
            have = m.scores.get(team)
            if have is None:
                continue
            if have == (g, b):
                res.counters["match_attr_equal"] += 1
            else:
                res.counters["match_attr_mismatch"] += 1
                res.findings.append(_attr(layer, s, m, f"final_score:{team}", f"{g}.{b}", f"{have[0]}.{have[1]}"))
    return res


def _drawn(s: SourceMatchInfo) -> bool:
    pts = [6 * g + b for g, b in s.scores.values()]
    return len(pts) == 2 and pts[0] == pts[1]


def _attr(layer: str, s: SourceMatchInfo, m: LocalMatchInfo, name: str, expected: str, actual: str) -> dict[str, Any]:
    return make_finding(
        "MATCH_ATTR_MISMATCH",
        layer=layer,
        rule_id=f"R-MATCH-{name.split(':')[0].upper()}",
        season=s.season,
        match={"source_url": s.url, "local_id": m.rec.match_id},
        field=name,
        expected=expected,
        actual=actual,
        evidence={"source_url": s.url, "body_sha256": s.sha256, "locator": name},
        local={"origin": m.origin},
    )


# ---------------------------------------------------------------------------
# Local match adapters
# ---------------------------------------------------------------------------


def snapshot_matches(rows: list[dict[str, Any]]) -> list[LocalMatchInfo]:
    out = []
    for r in rows:
        teams = frozenset({r["home_source_name"], r["away_source_name"]})
        scores = {
            r["home_source_name"]: (r.get("home_final_goals"), r.get("home_final_behinds")),
            r["away_source_name"]: (r.get("away_final_goals"), r.get("away_final_behinds")),
        }
        d = r.get("match_date")
        rec = LocalMatchRec(
            match_id=r["match_id"],
            season=int(r["season"]),
            stage_text=r["stage_label"],
            teams=teams,
            date=str(d)[:10] if d else None,
            replay=int(r.get("replay_occurrence") or 0),
        )
        out.append(
            LocalMatchInfo(
                rec,
                {t: (int(g), int(b)) for t, (g, b) in scores.items() if g is not None and b is not None},
                f"{r.get('source_path')}#{r.get('source_row')}" if r.get("source_path") else r["match_id"],
                r.get("date_precision") or "day",
            )
        )
    return sorted(out, key=lambda m: (m.rec.season, m.rec.date or "", m.rec.match_id))


def legacy_matches(root: Path) -> tuple[list[LocalMatchInfo], list[str]]:
    """The legacy ``matches_YYYY.csv`` files as raw rows; replay ordinals come from date order, never from text."""
    problems: list[str] = []
    raw: list[tuple[int, int, dict[str, str]]] = []
    for path in sorted((root / "data" / "matches").glob("matches_*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for n, row in enumerate(csv.DictReader(fh), 1):
                raw.append((int(row["year"]) if (row.get("year") or "").isdigit() else -1, n, row))
    groups: dict[tuple[int, str, frozenset[str]], list[tuple[str, int, dict[str, str]]]] = defaultdict(list)
    for year, n, row in raw:
        teams = frozenset({row["team_1_team_name"], row["team_2_team_name"]})
        groups[(year, row["round_num"], teams)].append(((row.get("date") or "")[:10], n, row))
    out: list[LocalMatchInfo] = []
    for (year, rnd, teams), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], sorted(kv[0][2]))):
        for ordinal, (d, n, row) in enumerate(sorted(rows, key=lambda x: (x[0], x[1]))):
            try:
                scores = {
                    row[f"team_{k}_team_name"]: (int(row[f"team_{k}_final_goals"]), int(row[f"team_{k}_final_behinds"]))
                    for k in (1, 2)
                }
            except (KeyError, ValueError):
                scores = {}
                problems.append(f"matches_{year}.csv row {n}: final score cells are not integers")
            rec = LocalMatchRec(f"legacy:{year}:{n}", year, rnd, teams, d or None, ordinal)
            out.append(LocalMatchInfo(rec, scores, f"data/matches/matches_{year}.csv#{n}"))
    return sorted(out, key=lambda m: (m.rec.season, m.rec.date or "", m.rec.match_id)), problems
