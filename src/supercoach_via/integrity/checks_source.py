"""Families D/E: freshness claims and source -> snapshot agreement.

Source pages are read from content-addressed evidence (``raw/objects`` or ``--evidence``)
by the SHA-256 the snapshot pinned, parsed by :mod:`integrity.sourcepages` (independent of
the production adapter), and compared cell by cell with the canonical rows. A missing
payload is missing evidence (UNKNOWN), never a PASS. A request timestamp alone proves
nothing: freshness needs a PASS observation whose content is the pinned revision.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS, Severity
from supercoach_via.integrity import sourcepages as sp
from supercoach_via.integrity.context import AuditContext, CheckSkipped, CheckSpec, rule
from supercoach_via.integrity.report import Kind, Status

B, E, W = Severity.BLOCKING, Severity.ERROR, Severity.WARNING
SEASON_ADAPTER, MATCH_ADAPTER = "afltables.season_fixture", "afltables.match_detail"

RULES = [
    rule(
        "freshness.checked_without_observation",
        "freshness.observations",
        B,
        "fixture_checked_at is set but no PASS season-fixture observation exists at or before it",
        "re-run the refresh; a check time needs a successful observation behind it",
    ),
    rule(
        "freshness.checked_from_failed_observation",
        "freshness.observations",
        B,
        "the latest season-fixture observation at the check time did not PASS (partial fetch marked fresh)",
        "revert fixture_checked_at to the last successful check and re-run the refresh",
    ),
    rule(
        "freshness.checked_at_behind_observation",
        "freshness.observations",
        B,
        "a later PASS observation exists but fixture_checked_at did not advance (stale metadata)",
        "rebuild season aggregates for the checked season",
    ),
    rule(
        "freshness.revision_mismatch",
        "freshness.observations",
        B,
        "the pinned season revision is not the content of the latest PASS observation",
        "re-pin source_revisions from the accepted observation",
    ),
    rule(
        "freshness.schedule_complete_unsupported",
        "freshness.observations",
        B,
        "schedule_complete is true without a source-declared season status",
        "leave schedule_complete unknown until the source declares the season complete",
    ),
    rule(
        "freshness.verified_claim",
        "freshness.observations",
        B,
        "the dataset is labelled verified but contains legacy-import rows",
        "label the snapshot legacy_unverified or partial; only source verification upgrades it",
    ),
    rule(
        "freshness.after_as_of",
        "freshness.as_of",
        B,
        "the inputs contain timestamps after the audit's --as-of (knowledge from the future)",
        "audit with an --as-of on or after the snapshot, or audit an earlier snapshot",
    ),
    rule(
        "freshness.fixture_stale",
        "freshness.as_of",
        W,
        "the current season's fixture was last checked longer ago than the policy allows at --as-of",
        "run a bounded refresh; stale is not wrong, but it is not current",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "freshness.evidence_missing",
        "freshness.fixture_inventory",
        E,
        "the pinned season page is not in the evidence store",
        "copy raw/objects from the refresh host or pass --evidence DIR",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "freshness.page_unreadable",
        "freshness.fixture_inventory",
        E,
        "the pinned season page could not be read by the independent reader",
        "inspect the payload; parser drift must be fixed before the comparison can run",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "freshness.team_unresolved",
        "freshness.fixture_inventory",
        E,
        "a source team name does not resolve through the snapshot's clubs/club_aliases",
        "add the alias to the club registry with evidence",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "freshness.missing_result",
        "freshness.fixture_inventory",
        E,
        "the pinned source lists a completed match the snapshot does not hold",
        "run a bounded repair for the season",
        current_blocks=True,
    ),
    rule(
        "freshness.scheduled_fixture_absent",
        "freshness.fixture_inventory",
        W,
        "the pinned source lists a scheduled fixture the snapshot does not hold",
        "refresh the fixture; forecasts cannot target a fixture the snapshot lacks",
        current_blocks=True,
    ),
    rule(
        "freshness.unexpected_match",
        "freshness.fixture_inventory",
        E,
        "the snapshot holds a match the pinned source does not list",
        "check the match identity against the source",
        current_blocks=True,
    ),
    rule(
        "freshness.fixture_value_mismatch",
        "freshness.fixture_inventory",
        B,
        "a match value differs from the pinned season page",
        "repair the match row from the pinned source",
    ),
    rule(
        "source.evidence_missing",
        "source.match_pages",
        E,
        "a required source payload (pinned or behind a source_fetch row) is not in the evidence store",
        "copy raw/objects from the refresh host or pass --evidence DIR",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "source.page_unreadable",
        "source.match_pages",
        E,
        "a match page could not be read by the independent reader",
        "inspect the payload for parser drift",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "source.revision_unobserved",
        "source.match_pages",
        B,
        "a pinned source revision has no observation with that content",
        "re-pin from a recorded observation",
    ),
    rule(
        "source.fetch_unrecorded",
        "source.match_pages",
        B,
        "a source_fetch row names a payload no observation recorded",
        "re-run the refresh; provenance must be recorded",
    ),
    rule(
        "source.match_unlinked",
        "source.match_pages",
        E,
        "a captured match page matches no canonical match",
        "check match identity and team aliases",
        kind=Kind.MISSING_EVIDENCE,
        current_blocks=True,
    ),
    rule(
        "source.match_value_mismatch",
        "source.match_pages",
        B,
        "a match value (date, venue, attendance, stage, score) differs from the captured page",
        "repair the match row from the captured source",
    ),
    rule(
        "source.player_missing",
        "source.match_pages",
        B,
        "a player row on the captured page has no canonical row",
        "re-import the match's player rows",
    ),
    rule(
        "source.player_extra",
        "source.match_pages",
        B,
        "a canonical player row is not on the captured page",
        "remove or re-link the row",
    ),
    rule(
        "source.player_ambiguous",
        "source.match_pages",
        E,
        "a captured player matches several canonical rows",
        "disambiguate the identity with the player's source URL",
        kind=Kind.MISSING_EVIDENCE,
        current_blocks=True,
    ),
    rule(
        "source.jersey_mismatch",
        "source.match_pages",
        E,
        "a player's jumper number differs from the page",
        "re-check the identity link",
        current_blocks=True,
    ),
    rule(
        "source.stat_cell_mismatch",
        "source.match_pages",
        B,
        "a statistic cell differs from the captured page (blank on the page = null)",
        "repair the row from the captured source; a swapped or edited value surfaces here",
    ),
    rule(
        "source.value_without_source_column",
        "source.match_pages",
        W,
        "a canonical statistic has a value although the page has no such column",
        "confirm where the value came from",
        kind=Kind.ANOMALY,
    ),
    rule(
        "source.page_totals_inconsistent",
        "source.match_pages",
        W,
        "the page's own Totals/Rushed rows disagree with its player rows",
        "note the source inconsistency",
        kind=Kind.ANOMALY,
    ),
    rule(
        "source.player_page_missing",
        "source.player_pages",
        E,
        "a captured player page that source-fetched rows depend on is not in the evidence store",
        "restore the payload from the archive, or pass --evidence DIR",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "source.player_page_unreadable",
        "source.player_pages",
        E,
        "a captured player page cannot be read by the checker's own reader",
        "inspect the archived page; the markup may have changed",
        kind=Kind.MISSING_EVIDENCE,
    ),
    rule(
        "source.player_page_unlinked",
        "source.player_pages",
        E,
        "a captured player page names no canonical player, or one of its games matches several rows",
        "record the player's source URL or resolve the duplicate rows",
        kind=Kind.MISSING_EVIDENCE,
        current_blocks=True,
    ),
    rule(
        "source.player_page_membership",
        "source.player_pages",
        B,
        "a game on the captured player page has no canonical row, or the player has a row the page does not list",
        "repair the player's rows from the captured page",
    ),
    rule(
        "source.player_page_value",
        "source.player_pages",
        B,
        "a canonical row differs from the captured player page (counter, result, jumper, non-blank cells; a "
        "blank cell only establishes that the count is not positive)",
        "repair the row from the captured page",
    ),
]

PLAYER_ADAPTER = "afltables.player_page"


def _season_url(season: int) -> str:
    from supercoach_via.ingest.afltables import season_url

    return season_url(season)


def _iso(v: Any) -> Any:
    return v.isoformat().replace("+00:00", "Z") if hasattr(v, "isoformat") else v


# ---------------------------------------------------------------------------
# freshness: observations and claims
# ---------------------------------------------------------------------------


def check_observations(ctx: AuditContext) -> list[str]:
    ctx.need("seasons", "source_observations", "matches", "player_games")
    assert ctx.snapshot is not None and ctx.snapshot.manifest is not None
    revisions = ctx.snapshot.manifest.source_revisions
    obs = ctx.records(
        "SELECT source_ref, url, fetched_at, content_sha256, outcome FROM source_observations "
        "WHERE adapter = ? ORDER BY fetched_at NULLS FIRST, source_ref",
        [SEASON_ADAPTER],
    )
    by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in obs:
        by_url[str(o["url"])].append(o)
    for srow in ctx.records("SELECT * FROM seasons ORDER BY season"):
        season = int(srow["season"])
        ctx.count("rows")
        entity = f"season:{season}"
        mine = by_url.get(_season_url(season), [])
        checked: datetime | None = srow["fixture_checked_at"]
        passes = [o for o in mine if o["outcome"] == "PASS" and o["fetched_at"] is not None]
        if checked is not None:
            before = [o for o in mine if o["fetched_at"] is not None and o["fetched_at"] <= checked]
            if not any(o["outcome"] == "PASS" for o in before):
                ctx.add(
                    "freshness.checked_without_observation",
                    entity,
                    table="seasons",
                    field="fixture_checked_at",
                    season=season,
                    actual=_iso(checked),
                )
            else:
                last = max(o["fetched_at"] for o in before)
                at_last = [o for o in before if o["fetched_at"] == last]
                if any(o["outcome"] != "PASS" for o in at_last):
                    ctx.add(
                        "freshness.checked_from_failed_observation",
                        entity,
                        table="seasons",
                        field="fixture_checked_at",
                        season=season,
                        actual=_iso(checked),
                        evidence={"observations": sorted(o["source_ref"] for o in at_last)},
                    )
        pinned = revisions.get(f"afltables:season:{season}")
        if passes:
            latest = max(passes, key=lambda o: (o["fetched_at"], o["source_ref"]))
            if checked is None or latest["fetched_at"] > checked:
                ctx.add(
                    "freshness.checked_at_behind_observation",
                    entity,
                    table="seasons",
                    field="fixture_checked_at",
                    season=season,
                    expected=_iso(latest["fetched_at"]),
                    actual=_iso(checked),
                    evidence={"source_ref": latest["source_ref"]},
                )
            if pinned != latest["content_sha256"]:
                ctx.add(
                    "freshness.revision_mismatch",
                    entity,
                    table="seasons",
                    season=season,
                    expected=latest["content_sha256"],
                    actual=pinned,
                )
        elif pinned is not None:
            ctx.add(
                "freshness.revision_mismatch",
                entity,
                table="seasons",
                season=season,
                expected=None,
                actual=pinned,
                message="a season revision is pinned but no PASS observation records it",
            )
        if srow["schedule_complete"] is True and srow["source_status"] is None:
            ctx.add(
                "freshness.schedule_complete_unsupported",
                entity,
                table="seasons",
                field="schedule_complete",
                season=season,
                expected=None,
                actual=True,
            )
    status = ctx.snapshot.manifest.status.value
    if status == "verified":
        legacy = ctx.rows(
            "SELECT (SELECT count(*) FROM matches WHERE provenance = 'legacy_import') + "
            "(SELECT count(*) FROM player_games WHERE provenance = 'legacy_import')"
        )[0][0]
        if legacy:
            ctx.add(
                "freshness.verified_claim",
                f"snapshot:{ctx.snapshot.snapshot_id}",
                expected="no legacy rows",
                actual={"status": status, "legacy_rows": int(legacy)},
            )
    return []


def check_as_of(ctx: AuditContext) -> list[str]:
    if ctx.as_of is None:
        raise CheckSkipped(Status.NOT_APPLICABLE, "no --as-of given; time-dependent rules were not evaluated")
    ctx.need("seasons", "source_observations", "player_games")
    snap = ctx.snapshot
    assert snap is not None and snap.manifest is not None
    as_of = ctx.as_of
    stamps: list[tuple[str, Any]] = [
        ("source_observations.fetched_at", ctx.rows("SELECT max(fetched_at) FROM source_observations")[0][0]),
        ("seasons.fixture_checked_at", ctx.rows("SELECT max(fixture_checked_at) FROM seasons")[0][0]),
        ("player_games.available_at", ctx.rows("SELECT max(available_at) FROM player_games")[0][0]),
        ("manifest.created_at", snap.manifest.created_at),
    ]
    if snap.pointer is not None:
        stamps.append(("current.promoted_at", snap.pointer.promoted_at))
    for name, value in stamps:
        ctx.count("timestamps")
        if value is not None and value > as_of:
            ctx.add(
                "freshness.after_as_of",
                f"timestamp:{name}",
                field=name,
                expected=f"<= {_iso(as_of)}",
                actual=_iso(value),
            )
    if ctx.current_season is not None:
        row = ctx.records("SELECT * FROM seasons WHERE season = ?", [ctx.current_season])
        if row and row[0]["schedule_complete"] is not True:
            checked = row[0]["fixture_checked_at"]
            entity = f"season:{ctx.current_season}"
            if checked is None:
                ctx.add(
                    "freshness.fixture_stale",
                    entity,
                    table="seasons",
                    field="fixture_checked_at",
                    season=None,
                    expected=f"within {ctx.policy.max_fixture_age_hours:g}h",
                    actual=None,
                )
            elif checked <= as_of:
                hours = (as_of - checked).total_seconds() / 3600
                if hours > ctx.policy.max_fixture_age_hours:
                    ctx.add(
                        "freshness.fixture_stale",
                        entity,
                        table="seasons",
                        field="fixture_checked_at",
                        expected=f"within {ctx.policy.max_fixture_age_hours:g}h",
                        actual=round(hours, 3),
                        evidence={"fixture_checked_at": _iso(checked)},
                    )
    return []


# ---------------------------------------------------------------------------
# club name resolution (the snapshot's own registry)
# ---------------------------------------------------------------------------


def _club_names(ctx: AuditContext, season: int) -> dict[str, set[str]]:
    names: dict[str, set[str]] = defaultdict(set)
    for cid, name in ctx.rows("SELECT club_id, name FROM clubs"):
        names[str(name)].add(str(cid))
    if ctx.has("club_aliases"):
        ctx.need("club_aliases")
        for alias, cid in ctx.rows(
            "SELECT alias, club_id FROM club_aliases WHERE valid_from_season <= ? "
            "AND (valid_to_season IS NULL OR valid_to_season >= ?)",
            [season, season],
        ):
            names[str(alias)].add(str(cid))
    return names


def _resolve(names: dict[str, set[str]], name: str) -> str | None:
    ids = names.get(name, set())
    return next(iter(ids)) if len(ids) == 1 else None


# ---------------------------------------------------------------------------
# freshness: pinned fixture inventory vs accepted matches
# ---------------------------------------------------------------------------


def check_fixture_inventory(ctx: AuditContext) -> list[str]:
    snap = ctx.snapshot
    if snap is None or snap.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no parseable manifest")
    pinned = sorted(
        (int(k.rsplit(":", 1)[1]), v)
        for k, v in snap.manifest.source_revisions.items()
        if k.startswith("afltables:season:") and k.rsplit(":", 1)[1].isdigit()
    )
    if not pinned:
        raise CheckSkipped(Status.NOT_APPLICABLE, "no season fixture revision is pinned in this snapshot")
    ctx.need("matches", "clubs")
    unknown: list[str] = []
    for season, sha in pinned:
        entity = f"season:{season}"
        raw = ctx.evidence.get(sha)
        if raw is None:
            ctx.add("freshness.evidence_missing", entity, season=None, evidence={"sha256": sha})
            unknown.append(f"season {season} page {sha[:12]} not in the evidence store")
            continue
        page = sp.read_season_page(raw)
        if page.problems:
            ctx.add("freshness.page_unreadable", entity, evidence={"sha256": sha, "problems": page.problems[:5]})
            unknown.append(f"season {season} page unreadable")
            continue
        ctx.count("resources")
        names = _club_names(ctx, season)
        canon: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
        for m in ctx.records("SELECT * FROM matches WHERE season = ? ORDER BY match_id", [season]):
            canon[(m["match_date"], m["home_club_id"], m["away_club_id"])].append(m)
        seen: set[str] = set()
        unresolved = 0
        for f in page.fixtures:
            home, away = _resolve(names, f.home), _resolve(names, f.away)
            if home is None or away is None:
                unresolved += 1
                ctx.add(
                    "freshness.team_unresolved",
                    f"source_team:{f.home if home is None else f.away}",
                    evidence={"season": season},
                )
                continue
            key = (f.match_date, home, away)
            fid = f"fixture:{_iso(f.match_date)}|{home}|{away}"
            candidates = [m for m in canon.get(key, []) if m["match_id"] not in seen]
            scored = f.home_points is not None and f.away_points is not None
            if not candidates:
                if scored:
                    ctx.add(
                        "freshness.missing_result", fid, season=season, evidence={"stage": f.stage_text, "sha256": sha}
                    )
                else:
                    ctx.add(
                        "freshness.scheduled_fixture_absent",
                        fid,
                        season=season,
                        evidence={"stage": f.stage_text, "sha256": sha},
                    )
                continue
            m = candidates[0]
            seen.add(m["match_id"])
            _compare_fixture(ctx, m, f, scored, sha)
        for _key, ms in sorted(canon.items(), key=lambda kv: str(kv[0])):
            for m in ms:
                if m["match_id"] not in seen:
                    ctx.add(
                        "freshness.unexpected_match",
                        f"match:{m['match_id']}",
                        table="matches",
                        season=season,
                        evidence={"sha256": sha},
                    )
        if unresolved:
            unknown.append(f"season {season}: {unresolved} source team names unresolved")
    return unknown


def _compare_fixture(ctx: AuditContext, m: dict[str, Any], f: sp.SourceFixture, scored: bool, sha: str) -> None:
    entity = f"match:{m['match_id']}"
    season = int(m["season"])
    checks: list[tuple[str, Any, Any]] = [
        ("stage_label", sp.stage_label_for(f.stage_text), m["stage_label"]),
        ("local_start", f.local_start, m["local_start"]),
        ("venue_source_name", f.venue, m["venue_source_name"]),
        ("attendance", f.attendance, m["attendance"]),
        ("status", "complete" if scored else "scheduled", m["status"]),
    ]
    if scored:
        assert f.home_quarters is not None and f.away_quarters is not None
        for side, qs, pts in (("home", f.home_quarters, f.home_points), ("away", f.away_quarters, f.away_points)):
            for (g, b), q in zip(qs, ("q1", "q2", "q3", "final"), strict=True):
                have = f"{m[f'{side}_{q}_goals']}.{m[f'{side}_{q}_behinds']}"
                checks.append((f"{side}_{q}", f"{g}.{b}", have))
            checks.append((f"{side}_score", pts, m[f"{side}_score"]))
    for field, want, have in checks:
        ctx.count("cells")
        if want is not None and want != have:
            ctx.add(
                "freshness.fixture_value_mismatch",
                entity,
                table="matches",
                field=field,
                season=season,
                expected=want,
                actual=have,
                evidence={"sha256": sha},
            )


# ---------------------------------------------------------------------------
# source: captured match pages vs canonical rows
# ---------------------------------------------------------------------------


def check_match_pages(ctx: AuditContext) -> list[str]:
    from supercoach_via.domain.ids import normalize_name

    snap = ctx.snapshot
    if snap is None or snap.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no parseable manifest")
    ctx.need("matches", "player_games", "players", "source_observations", "clubs")
    revisions = snap.manifest.source_revisions
    observations = ctx.records(
        "SELECT source_ref, adapter, url, fetched_at, content_sha256, outcome FROM source_observations "
        "ORDER BY url, fetched_at NULLS FIRST, source_ref"
    )
    observed = {o["content_sha256"] for o in observations if o["content_sha256"]}
    for key, sha in sorted(revisions.items()):
        if key.startswith("afltables:") and sha not in observed:
            ctx.add(
                "source.revision_unobserved", f"revision:{key}", expected="an observation with this content", actual=sha
            )
    fetched = ctx.rows(
        "SELECT DISTINCT source_sha256 FROM (SELECT source_sha256 FROM matches WHERE provenance = 'source_fetch' "
        "UNION ALL SELECT source_sha256 FROM player_games WHERE provenance = 'source_fetch') "
        "WHERE source_sha256 IS NOT NULL "
        "ORDER BY 1"
    )
    required = {str(s) for (s,) in fetched}
    for s in sorted(required - observed):
        ctx.add("source.fetch_unrecorded", f"payload:{s}", expected="a recorded observation", actual=s)
    pinned = {v for k, v in revisions.items() if k.startswith("afltables:game:")}
    required |= pinned
    pages: dict[str, dict[str, Any]] = {}
    for o in observations:
        if o["adapter"] == MATCH_ADAPTER and o["outcome"] == "PASS" and o["content_sha256"]:
            pages[str(o["url"])] = o  # last by fetched_at wins; the pinned one wins below
    for o in observations:
        if o["adapter"] == MATCH_ADAPTER and o["content_sha256"] in pinned:
            pages[str(o["url"])] = o
    unknown: list[str] = []
    cov: dict[str, Any] = {
        "pages_observed": len(pages),
        "pages_compared": 0,
        "pages_missing_optional": 0,
        "player_games_compared": 0,
        "cells_compared": 0,
        "player_games_source_fetch": 0,
        "player_games_total": 0,
    }
    compared_rows: set[tuple[str, str, str]] = set()
    names_by_id = {pid: str(n) for pid, n in ctx.rows("SELECT player_id, display_name FROM players")}
    url_to_player: dict[str, str] = {}
    for pid, urls in ctx.rows("SELECT player_id, source_urls FROM players WHERE source_urls IS NOT NULL ORDER BY 1"):
        try:
            for u in json.loads(urls):
                url_to_player.setdefault(str(u), str(pid))
        except ValueError:
            continue
    for url, o in sorted(pages.items()):
        sha = str(o["content_sha256"])
        raw = ctx.evidence.get(sha)
        if raw is None:
            if sha in required:
                ctx.add("source.evidence_missing", f"payload:{sha}", evidence={"url": url})
                unknown.append(f"required payload {sha[:12]} missing")
            else:
                cov["pages_missing_optional"] += 1
            continue
        page = sp.read_match_page(raw)
        if page.problems:
            ctx.add("source.page_unreadable", f"payload:{sha}", evidence={"url": url, "problems": page.problems[:5]})
            unknown.append(f"payload {sha[:12]} unreadable")
            continue
        match = _link_match(ctx, page, sha)
        if match is None:
            ctx.add(
                "source.match_unlinked",
                f"source:{url}",
                evidence={"sha256": sha},
                season=page.match_date.year if page.match_date else None,
            )
            continue
        cov["pages_compared"] += 1
        ctx.count("resources")
        _compare_match(ctx, match, page, sha, url)
        for r in _compare_players(ctx, match, page, sha, url, url_to_player, names_by_id, normalize_name, cov):
            compared_rows.add(r)
    for s in sorted(required - {str(o["content_sha256"]) for o in pages.values()}):
        if ctx.evidence.get(s) is None:
            ctx.add("source.evidence_missing", f"payload:{s}", evidence={"reason": "no match-detail observation"})
            unknown.append(f"required payload {s[:12]} missing")
    cov["player_games_total"] = int(ctx.rows("SELECT count(*) FROM player_games")[0][0])
    cov["player_games_source_fetch"] = int(
        ctx.rows("SELECT count(*) FROM player_games WHERE provenance = 'source_fetch'")[0][0]
    )
    cov["player_games_without_capture"] = cov["player_games_total"] - len(compared_rows)
    if ctx.current_season is not None:
        cur_total = int(ctx.rows("SELECT count(*) FROM player_games WHERE season = ?", [ctx.current_season])[0][0])
        cov["current_season"] = {
            "season": ctx.current_season,
            "player_games": cur_total,
            "compared_with_capture": _in_season(ctx, compared_rows, ctx.current_season),
        }
    ctx.coverage["source_capture"] = cov
    return unknown


def _in_season(ctx: AuditContext, rows: set[tuple[str, str, str]], season: int) -> int:
    in_season = {str(m) for (m,) in ctx.rows("SELECT match_id FROM matches WHERE season = ?", [season])}
    return sum(1 for m, _p, _c in rows if m in in_season)


def _link_match(ctx: AuditContext, page: sp.SourceMatch, sha: str) -> dict[str, Any] | None:
    by_sha = ctx.records("SELECT * FROM matches WHERE source_sha256 = ? ORDER BY match_id", [sha])
    if len(by_sha) == 1:
        return by_sha[0]
    if page.match_date is None or len(page.teams) != 2:
        return None
    names = _club_names(ctx, page.match_date.year)
    home, away = _resolve(names, page.teams[0].name), _resolve(names, page.teams[1].name)
    found = ctx.records(
        "SELECT * FROM matches WHERE match_date = ? AND home_club_id = ? AND away_club_id = ? ORDER BY match_id",
        [page.match_date, home, away],
    )
    return found[0] if len(found) == 1 else None


def _compare_match(ctx: AuditContext, m: dict[str, Any], page: sp.SourceMatch, sha: str, url: str) -> None:
    entity = f"match:{m['match_id']}"
    season = int(m["season"])
    ev = {"source_sha256": sha, "url": url}
    checks: list[tuple[str, Any, Any]] = [
        ("match_date", _iso(page.match_date), _iso(m["match_date"])),
        ("local_start", page.local_start, m["local_start"]),
        ("venue_source_name", page.venue, m["venue_source_name"]),
        ("attendance", page.attendance, m["attendance"]),
        ("stage_label", sp.stage_label_for(page.stage_text or ""), m["stage_label"]),
    ]
    for side, team in zip(("home", "away"), page.teams, strict=False):
        checks.append((f"{side}_source_name", team.name, m[f"{side}_source_name"]))
        for (g, b), q in zip(team.quarters, ("q1", "q2", "q3", "final"), strict=True):
            checks.append((f"{side}_{q}", f"{g}.{b}", f"{m[f'{side}_{q}_goals']}.{m[f'{side}_{q}_behinds']}"))
        checks.append((f"{side}_score", team.points[-1], m[f"{side}_score"]))
    for field, want, have in checks:
        ctx.count("cells")
        if want != have:
            ctx.add(
                "source.match_value_mismatch",
                entity,
                table="matches",
                field=field,
                season=season,
                expected=want,
                actual=have,
                evidence=ev,
            )


def _same(want: Any, have: Any) -> bool:
    if isinstance(want, str) or isinstance(have, str):
        return bool(want == have)
    if want is None or have is None:
        return want is None and have is None
    return float(want) == float(have)


#: counts every player who takes the field records when the page reports them
_UNIVERSAL_CELLS = ("kicks", "marks", "handballs", "disposals", "time_on_ground_pct")


def expected_cells(page: sp.SourceMatch) -> dict[tuple[str, str], dict[str, Any]]:
    """What each cell on a captured match page says, as canonical values.

    AFL Tables prints 0 as a blank. A blank is 0 when the page reports that column (some
    player has a value), the player took the field (some value other than Brownlow votes,
    or the page has none of the statistics every player on the field records), the column is
    not time on ground, and it is not Brownlow votes on a finals page. Otherwise it is null.
    Unparseable text is kept as text so the comparison reports it.
    """
    final = (page.stage_text or "") in sp.FINAL_NAMES
    reported = {s for p in page.players for s, text in p.cells.items() if text.strip()}
    universal = bool(reported & set(_UNIVERSAL_CELLS))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for p in page.players:
        took_field = not universal or any(t.strip() for s, t in p.cells.items() if s != "brownlow_votes")
        row: dict[str, Any] = {}
        for stat, text in p.cells.items():
            value = sp.cell_value(stat, text)
            blank_is_zero = (
                value is None
                and stat in reported
                and took_field
                and stat != "time_on_ground_pct"
                and not (final and stat == "brownlow_votes")
            )
            row[stat] = 0 if blank_is_zero else value
        out[(p.team, p.name)] = row
    return out


def _compare_players(
    ctx: AuditContext,
    m: dict[str, Any],
    page: sp.SourceMatch,
    sha: str,
    url: str,
    url_to_player: dict[str, str],
    names_by_id: dict[str, str],
    normalize: Any,
    cov: dict[str, Any],
) -> list[tuple[str, str, str]]:
    season = int(m["season"])
    ev = {"source_sha256": sha, "url": url}
    meaning = expected_cells(page)
    team_club = {t.name: m[f"{side}_club_id"] for side, t in zip(("home", "away"), page.teams, strict=False)}
    rows = ctx.records("SELECT * FROM player_games WHERE match_id = ? ORDER BY club_id, player_id", [m["match_id"]])
    by_club: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_club[str(r["club_id"])].append(r)
    matched: set[tuple[str, str, str]] = set()
    for team, labels in sorted(page.columns.items()):
        club = team_club.get(team)
        present = {sp.SOURCE_STAT_LABELS[lab] for lab in labels if lab in sp.SOURCE_STAT_LABELS}
        players = [p for p in page.players if p.team == team]
        if club is None:
            continue
        pool = by_club.get(club, [])
        cells_by_stat: dict[str, int] = defaultdict(int)
        for p in players:
            for stat, text in p.cells.items():
                v = sp.cell_value(stat, text)
                if isinstance(v, int | float):
                    cells_by_stat[stat] += int(v)
            target = None
            if p.link:
                pid = url_to_player.get(urljoin(url, p.link))
                cands = [r for r in pool if r["player_id"] == pid] if pid else []
                target = cands[0] if len(cands) == 1 else None
            if target is None:
                want = normalize(sp.person_name(p.name))
                cands = [r for r in pool if normalize(names_by_id.get(r["player_id"], "")) == want]
                if len(cands) > 1:
                    ctx.add(
                        "source.player_ambiguous",
                        f"source_player:{m['match_id']}|{club}|{sp.person_name(p.name)}",
                        season=season,
                        actual=sorted(r["player_id"] for r in cands),
                        evidence=ev,
                    )
                    continue
                target = cands[0] if cands else None
            if target is None:
                ctx.add(
                    "source.player_missing",
                    f"source_player:{m['match_id']}|{club}|{sp.person_name(p.name)}",
                    table="player_games",
                    season=season,
                    evidence={**ev, "link": p.link},
                )
                continue
            key = (str(target["match_id"]), str(target["player_id"]), str(target["club_id"]))
            if key in matched:
                ctx.add("source.player_ambiguous", f"player_game:{'|'.join(key)}", season=season, evidence=ev)
                continue
            matched.add(key)
            entity = "player_game:" + "|".join(key)
            jn = sp.jersey(p.jersey_token)
            if jn is not None and target["jersey_number"] is not None and jn != target["jersey_number"]:
                ctx.add(
                    "source.jersey_mismatch",
                    entity,
                    table="player_games",
                    field="jersey_number",
                    season=season,
                    expected=jn,
                    actual=target["jersey_number"],
                    evidence=ev,
                )
            for stat in PLAYER_STAT_COLUMNS:
                ctx.count("cells")
                cov["cells_compared"] += 1
                have = target[stat]
                if stat in p.cells:
                    want = meaning[(p.team, p.name)][stat]
                    if not _same(want, have):
                        ctx.add(
                            "source.stat_cell_mismatch",
                            entity,
                            table="player_games",
                            field=stat,
                            season=season,
                            expected=want,
                            actual=have,
                            evidence=ev,
                        )
                elif have is not None:
                    ctx.add(
                        "source.value_without_source_column",
                        entity,
                        table="player_games",
                        field=stat,
                        season=season,
                        expected=None,
                        actual=have,
                        evidence=ev,
                    )
            cov["player_games_compared"] += 1
        for r in pool:
            key = (str(r["match_id"]), str(r["player_id"]), str(r["club_id"]))
            if key not in matched:
                ctx.add(
                    "source.player_extra",
                    "player_game:" + "|".join(key),
                    table="player_games",
                    season=season,
                    evidence=ev,
                )
        totals = page.totals.get(team)
        if totals:
            rushed = page.rushed.get(team) or 0  # no Rushed row: none were rushed
            for stat in sorted(present):
                t = sp.cell_value(stat, totals.get(stat, ""))
                if not isinstance(t, int | float) or stat == "time_on_ground_pct":
                    continue
                # the Totals row's behinds include rushed behinds, which no player is credited with
                players_sum = cells_by_stat[stat] + (rushed if stat == "behinds" else 0)
                if int(t) != players_sum:
                    ctx.add(
                        "source.page_totals_inconsistent",
                        f"source:{url}|{team}",
                        field=stat,
                        expected=int(t),
                        actual=players_sum,
                        evidence=ev,
                    )
            line = next((t for t in page.teams if t.name == team), None)
            bh = sp.cell_value("behinds", totals.get("behinds", ""))
            if line is not None and isinstance(bh, int) and bh != line.quarters[-1][1]:
                ctx.add(
                    "source.page_totals_inconsistent",
                    f"source:{url}|{team}",
                    field="final_behinds",
                    expected=line.quarters[-1][1],
                    actual=bh,
                    evidence=ev,
                )
    return sorted(matched)


def check_player_pages(ctx: AuditContext) -> list[str]:
    """Captured AFL Tables player pages vs the player's canonical rows (the B1 repair evidence).

    A page lists every game of a career, so it establishes the player's row MEMBERSHIP, and per
    row the club, opponent and round (the link), the career game counter, result, jumper number
    and every non-blank statistic cell exactly. A blank cell establishes only that the count
    was not positive: whether it is a recorded zero is decided by the match-level blank rule.
    Rows of players without a captured page are outside this check.
    """
    snap = ctx.snapshot
    if snap is None or snap.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no parseable manifest")
    ctx.need("matches", "player_games", "players", "source_observations", "clubs")
    pages: dict[str, dict[str, Any]] = {}
    for o in ctx.records(
        "SELECT url, content_sha256 FROM source_observations WHERE adapter = ? AND outcome = 'PASS' "
        "AND content_sha256 IS NOT NULL ORDER BY url, fetched_at NULLS FIRST, source_ref",
        [PLAYER_ADAPTER],
    ):
        pages[str(o["url"])] = o  # the latest capture of each page
    if not pages:
        raise CheckSkipped(Status.NOT_APPLICABLE, "no captured player pages")
    required = {
        str(s)
        for (s,) in ctx.rows(
            "SELECT DISTINCT source_sha256 FROM player_games WHERE provenance = 'source_fetch' "
            "AND source_sha256 IS NOT NULL"
        )
    }
    url_to_player: dict[str, str] = {}
    for pid, urls in ctx.rows("SELECT player_id, source_urls FROM players WHERE source_urls IS NOT NULL ORDER BY 1"):
        try:
            for u in json.loads(urls):
                url_to_player.setdefault(str(u), str(pid))
        except ValueError:
            continue
    cov: dict[str, Any] = {
        "pages_observed": len(pages),
        "pages_compared": 0,
        "pages_missing_optional": 0,
        "rows_on_pages": 0,
        "rows_compared": 0,
        "cells_exact": 0,
        "cells_blank_not_positive": 0,
        "players": [],
    }
    unknown: list[str] = []
    for url, o in sorted(pages.items()):
        sha = str(o["content_sha256"])
        raw = ctx.evidence.get(sha)
        if raw is None:
            if sha in required:
                ctx.add("source.player_page_missing", f"payload:{sha}", evidence={"url": url})
                unknown.append(f"required player page {sha[:12]} missing")
            else:
                cov["pages_missing_optional"] += 1
            continue
        page = sp.read_player_page(raw)
        if page.problems:
            ctx.add(
                "source.player_page_unreadable", f"payload:{sha}", evidence={"url": url, "problems": page.problems[:5]}
            )
            unknown.append(f"player page {sha[:12]} unreadable")
            continue
        pid = url_to_player.get(url)
        if pid is None:
            ctx.add(
                "source.player_page_unlinked",
                f"source:{url}",
                message="no canonical player records this URL",
                evidence={"sha256": sha},
            )
            continue
        cov["pages_compared"] += 1
        cov["players"].append(pid)
        _compare_player_page(ctx, pid, page, {"source_sha256": sha, "url": url}, cov)
    ctx.coverage["player_pages"] = cov
    return unknown


def _compare_player_page(
    ctx: AuditContext, pid: str, page: sp.SourcePlayerPage, ev: dict[str, Any], cov: dict[str, Any]
) -> None:
    rows = ctx.records(
        "SELECT g.*, m.stage_label AS m_stage_label FROM player_games g JOIN matches m USING (match_id) "
        "WHERE g.player_id = ? ORDER BY g.season, g.career_game_counter, g.match_id",
        [pid],
    )
    names = {s: _club_names(ctx, s) for s in {g.season for g in page.games}}
    linked: set[str] = set()
    cov["rows_on_pages"] += len(page.games)
    for g in page.games:
        club = _resolve(names[g.season], g.team)
        opp = _resolve(names[g.season], g.opponent)
        stage = sp.stage_label_for_round(g.round_token)
        cands = [
            r
            for r in rows
            if r["season"] == g.season
            and r["club_id"] == club
            and r["opponent_club_id"] == opp
            and r["m_stage_label"] == stage
            and r["match_id"] not in linked
        ]
        if len(cands) > 1:
            cands = [r for r in cands if r["career_game_counter"] == g.counter] or cands
        where = f"source_player_game:{pid}|{g.season}|{g.team}|{g.round_token}|{g.opponent}"
        if not cands:
            ctx.add(
                "source.player_page_membership",
                where,
                season=g.season,
                expected="a canonical row",
                actual="absent",
                evidence=ev,
            )
            continue
        if len(cands) > 1:
            ctx.add("source.player_page_unlinked", where, season=g.season, evidence={**ev, "rows": len(cands)})
            continue
        r = cands[0]
        linked.add(r["match_id"])
        cov["rows_compared"] += 1
        entity = f"player_game:{r['match_id']}|{pid}|{r['club_id']}"
        checks: list[tuple[str, Any, Any]] = [
            ("career_game_counter", g.counter, r["career_game_counter"]),
            ("result", g.result or None, r["result"]),
            ("jersey_number", sp.jersey(g.jersey), r["jersey_number"]),
        ]
        for field, want, have in checks:
            ctx.count("cells")
            if want != have:
                ctx.add(
                    "source.player_page_value",
                    entity,
                    table="player_games",
                    field=field,
                    season=g.season,
                    expected=want,
                    actual=have,
                    evidence=ev,
                )
        for stat, text in sorted(g.cells.items()):
            ctx.count("cells")
            have = r.get(stat)
            if text.strip():
                cov["cells_exact"] += 1
                want = sp.cell_value(stat, text)
                if not _same(want, have):
                    ctx.add(
                        "source.player_page_value",
                        entity,
                        table="player_games",
                        field=stat,
                        season=g.season,
                        expected=want,
                        actual=have,
                        evidence=ev,
                    )
            else:
                cov["cells_blank_not_positive"] += 1
                if have is not None and float(have) > 0:
                    ctx.add(
                        "source.player_page_value",
                        entity,
                        table="player_games",
                        field=stat,
                        season=g.season,
                        expected="blank: not a positive count",
                        actual=have,
                        evidence=ev,
                    )
    for r in rows:
        if r["match_id"] not in linked:
            ctx.add(
                "source.player_page_membership",
                f"player_game:{r['match_id']}|{pid}|{r['club_id']}",
                season=int(r["season"]),
                expected="listed on the player's page",
                actual="absent",
                evidence=ev,
            )


CHECKS = [
    CheckSpec(
        "freshness.observations",
        "freshness",
        "fixture freshness claims backed by PASS observations and pins",
        check_observations,
    ),
    CheckSpec("freshness.as_of", "freshness", "timestamps and staleness against the explicit --as-of", check_as_of),
    CheckSpec(
        "freshness.fixture_inventory", "freshness", "pinned season fixture vs accepted matches", check_fixture_inventory
    ),
    CheckSpec(
        "source.match_pages", "source", "captured match pages vs canonical match and player rows", check_match_pages
    ),
    CheckSpec(
        "source.player_pages",
        "source",
        "captured player pages vs every canonical row of that player's career",
        check_player_pages,
    ),
]
