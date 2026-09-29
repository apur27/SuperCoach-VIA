"""Bounded corrections of accepted rows from pinned source captures (offline, with provenance).

A refresh upserts a match only when its status, score or start time changed, so a
fixture field the legacy import got wrong (an attendance of 0, for example) was never
corrected even though the pinned season page states the right value. ``fixture_corrections``
reads the season page the snapshot already pinned (``source_revisions``) from the raw
archive, parses it with the production adapter, and corrects fixture fields of matches
whose identity, status, scores and start time already agree with the page. Every change
is recorded as a resolved ``quality_issues`` row naming the old and new value and the
page's URL and SHA-256. A match whose score disagrees is not patched field by field: that
needs a bounded repair of the whole match.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import TABLES, MatchStatus, SnapshotManifest

ClubResolver = Callable[[str, int], str | None]
VenueResolver = Callable[[str], str | None]
FIELDS = ("attendance", "venue_source_name")


class CorrectionError(RuntimeError):
    """A pinned capture needed for a correction is missing or does not parse."""


def default_resolvers(config_dir: Path) -> tuple[ClubResolver, VenueResolver]:
    from supercoach_via.domain.ids import ClubRegistry, VenueRegistry

    clubs = ClubRegistry.from_csv(config_dir / "team_aliases.csv")
    venues = VenueRegistry.from_csv(config_dir / "venue_aliases.csv")
    return clubs.resolve, venues.resolve


def _issue(
    rule: str, severity: str, status: str, match: dict[str, Any], text: str, url: str, sha: str
) -> dict[str, Any]:
    digest = hashlib.sha256(f"{rule}\x1f{match['match_id']}\x1f{text}\x1f{sha}".encode()).hexdigest()[:24]
    return {
        "issue_id": f"qc:{digest}",
        "severity": severity,
        "status": status,
        "table_name": "matches",
        "row_key": match["match_id"],
        "source_path": url,
        "rule_id": rule,
        "explanation": text,
        "remediation": None,
        "acceptance_basis": f"pinned source {sha}",
        "season": match["season"],
    }


def _quarters(row: dict[str, Any], side: str) -> list[tuple[int, int]]:
    return [(row[f"{side}_{q}_goals"], row[f"{side}_{q}_behinds"]) for q in ("q1", "q2", "q3", "final")]


def fixture_corrections(
    data_root: Path,
    manifest: SnapshotManifest,
    season: int,
    *,
    club_resolver: ClubResolver,
    venue_resolver: VenueResolver,
) -> dict[str, list[dict[str, Any]]]:
    """Upserts correcting ``season``'s fixture fields from its pinned season page (empty if none pinned)."""
    from supercoach_via.ingest import afltables
    from supercoach_via.ingest.http import RawArchive
    from supercoach_via.storage.queries import SnapshotQuery

    out: dict[str, list[dict[str, Any]]] = {"matches": [], "quality_issues": []}
    sha = manifest.source_revisions.get(f"afltables:season:{season}")
    if not sha:
        return out
    page = RawArchive(data_root / "raw").get(sha)
    if page is None:
        raise CorrectionError(f"pinned season page {sha} for {season} is not in {data_root / 'raw'}")
    fixture = afltables.parse_season_page(page, season=season, club_resolver=club_resolver)
    if fixture.outcome.value != "PASS":
        raise CorrectionError(f"pinned season page {sha} does not parse: {fixture.issues[:3]}")
    url = afltables.season_url(season)
    with SnapshotQuery(data_root, manifest, tables={"matches"}, partitions={"matches": {str(season)}}) as q:
        base = {r["match_id"]: r for r in q.arrow("SELECT * FROM matches WHERE season = ?", [season]).to_pylist()}
    patched: dict[str, dict[str, Any]] = {}
    for m in fixture.completed():
        b = base.get(m.match_id)
        if b is None:
            continue  # a missing result is a refresh or repair, not a field correction
        patch: dict[str, Any] = {}
        if m.attendance is not None and m.attendance != b["attendance"]:
            patch["attendance"] = m.attendance
        if m.venue and m.venue != b["venue_source_name"]:
            patch["venue_source_name"] = m.venue
        if not patch:
            continue
        agrees = (
            b["status"] == MatchStatus.COMPLETE.value
            and b["home_score"] == m.home_score
            and b["away_score"] == m.away_score
            and b["local_start"] == m.local_start
            and _quarters(b, "home") == list(m.home_quarters or ())
            and _quarters(b, "away") == list(m.away_quarters or ())
        )
        if not agrees:
            out["quality_issues"].append(
                _issue(
                    "fixture_correction_refused",
                    "warning",
                    "open",
                    b,
                    f"{', '.join(sorted(patch))} differ from pinned season page {sha}, but the score or start time "
                    "differs too; repair the whole match",
                    url,
                    sha,
                )
            )
            continue
        row = patched.setdefault(m.match_id, dict(b))
        for name, value in sorted(patch.items()):
            out["quality_issues"].append(
                _issue(
                    "fixture_field_corrected",
                    "info",
                    "resolved",
                    b,
                    f"{name} {b[name]!r} -> {value!r} from pinned season page {sha}",
                    url,
                    sha,
                )
            )
            row[name] = value
        if "venue_source_name" in patch:
            row["venue_id"] = venue_resolver(str(patch["venue_source_name"]))
    for mid, b in sorted(base.items()):
        row = patched.get(mid, b)
        if row.get("venue_id") is None and row.get("venue_source_name"):
            vid = venue_resolver(str(row["venue_source_name"]))
            if vid is not None:
                row = patched.setdefault(mid, dict(b))
                row["venue_id"] = vid
                out["quality_issues"].append(
                    _issue(
                        "venue_id_resolved",
                        "info",
                        "resolved",
                        b,
                        f"venue_id None -> {vid!r} from venue_aliases.csv for {row['venue_source_name']!r}",
                        url,
                        sha,
                    )
                )
    names = TABLES["matches"].column_names
    out["matches"] = [{c: patched[mid].get(c) for c in names} for mid in sorted(patched)]
    return out
