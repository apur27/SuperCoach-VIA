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
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import TABLES, MatchStatus, SnapshotManifest

ClubResolver = Callable[[str, int], str | None]
VenueResolver = Callable[[str], str | None]
FIELDS = ("attendance", "venue_source_name")


class CorrectionError(RuntimeError):
    """A pinned capture needed for a correction is missing or does not parse."""


def pinned_season_evidence(
    root: Path, snapshot_id: str, seasons: set[int],
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, bytes]]:
    """Read only requested season captures and their observations from an immutable snapshot.

    This supplies evidence for an unpinned legacy import, never donor match/player rows
    or a claim that the imported season is complete. Every required capture must exist,
    hash to its pin and have a successful observation for the season page.
    """
    from supercoach_via.ingest import afltables
    from supercoach_via.ingest.http import RawArchive
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery

    try:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", snapshot_id):
            raise CorrectionError("correction evidence requires an immutable sha256:<id> snapshot")
        manifest = snapshots.load_snapshot(root, snapshot_id, verify=True)
        if "source_observations" not in manifest.tables:
            raise CorrectionError(f"evidence snapshot {snapshot_id} has no source observations")
        with SnapshotQuery(root, manifest, tables={"source_observations"}) as q:
            observed = q.arrow("SELECT * FROM source_observations").to_pylist()
        revisions: dict[str, str] = {}
        observations: list[dict[str, Any]] = []
        payloads: dict[str, bytes] = {}
        for season in sorted(seasons):
            key = f"afltables:season:{season}"
            sha = manifest.source_revisions.get(key)
            if not sha:
                raise CorrectionError(f"evidence snapshot {snapshot_id} has no pinned season page for {season}")
            snapshots.contained_path(root, f"raw/objects/{sha[:2]}/{sha}")
            body = RawArchive(root / "raw").get(sha)
            if body is None:
                raise CorrectionError(
                    f"pinned season page {sha} for {season} is missing or corrupt under {root / 'raw'}"
                )
            matching = [r for r in observed if r["content_sha256"] == sha
                        and r["url"] == afltables.season_url(season)
                        and r["adapter"] == "afltables.season_fixture"
                        and r["http_status"] == 200 and r["outcome"] == "PASS"]
            if not matching:
                raise CorrectionError(f"pinned season page {sha} has no successful season observation for {season}")
            revisions[key] = sha
            observations.extend(matching)
            payloads[sha] = body
        return revisions, observations, payloads
    except (OSError, ValueError, snapshots.IntegrityError) as exc:
        raise CorrectionError(f"cannot read pinned correction evidence: {exc}") from exc


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


# ---------------------------------------------------------------------------
# Player rows of a replayed drawn final (O55-06)
# ---------------------------------------------------------------------------


def replay_link_corrections(data_root: Path, manifest: SnapshotManifest) -> dict[str, list[dict[str, Any]]]:
    """Resolve rows linked to a drawn match whose own result is W/L (they cannot belong to a draw).

    Legacy player files name a stage and opponent, not the occurrence, so a player who only
    played the replay can be linked to the draw. The recorded W/L proves the draw link wrong
    but not the replay link. A row is relinked only when the official team scores prove it:
    moving the suspect rows makes the player goal sums of both the draw and the replay equal
    their teams' goals (and keeps player behinds within team behinds), the result agrees with
    the replay's outcome, and the player has no row on the replay yet. Anything else is
    quarantined with its evidence and an actionable issue.
    """
    import json

    from supercoach_via.storage.queries import SnapshotQuery

    out: dict[str, list[dict[str, Any]]] = {"player_games": [], "deletes": [], "quarantine": [], "quality_issues": []}
    with SnapshotQuery(data_root, manifest, tables={"matches", "player_games"}) as q:
        pairs = q.arrow(
            """SELECT d.match_id AS draw_id, r.match_id AS replay_id FROM matches d JOIN matches r
                 ON r.season = d.season AND r.stage_id = d.stage_id AND r.replay_occurrence = d.replay_occurrence + 1
                AND ((r.home_club_id = d.home_club_id AND r.away_club_id = d.away_club_id)
                  OR (r.home_club_id = d.away_club_id AND r.away_club_id = d.home_club_id))
               WHERE d.status = 'complete' AND r.status = 'complete' AND d.home_score = d.away_score
               ORDER BY 1"""
        ).to_pylist()
        for pair in pairs:
            draw, replay = (
                q.arrow("SELECT * FROM matches WHERE match_id = ?", [mid]).to_pylist()[0]
                for mid in (pair["draw_id"], pair["replay_id"])
            )
            rows = {
                mid: q.arrow("SELECT * FROM player_games WHERE match_id = ? ORDER BY player_id", [mid]).to_pylist()
                for mid in (draw["match_id"], replay["match_id"])
            }
            suspects = [g for g in rows[draw["match_id"]] if g["result"] in ("W", "L")]
            for club in sorted({g["club_id"] for g in suspects}):
                mine = [g for g in suspects if g["club_id"] == club]
                ok, why = _replay_move_proven(draw, replay, rows, club, mine)
                for g in mine:
                    key = f"{draw['match_id']}|{g['player_id']}|{club}"
                    out["deletes"].append(g)
                    if ok:
                        moved = dict(g)
                        moved.update(match_id=replay["match_id"], link_method="score_reconciled")
                        out["player_games"].append(moved)
                        out["quality_issues"].append(_pg_issue(
                            "replay_link_corrected", "info", "resolved", key, g["season"],
                            f"result {g['result']} cannot belong to drawn {draw['match_id']}; moved to "
                            f"{replay['match_id']}: {why}", None))  # fmt: skip
                    else:
                        digest = hashlib.sha256(f"replay\x1f{key}".encode()).hexdigest()[:24]
                        raw = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in g.items()}
                        out["quarantine"].append({
                            "quarantine_id": f"q:{digest}", "table_name": "player_games",
                            "reason": "replayed_draw_link_unresolved",
                            "candidates": json.dumps([draw["match_id"], replay["match_id"]]),
                            "raw": json.dumps(raw, sort_keys=True), "season": g["season"],
                            "provenance": g["provenance"], "source_path": g["source_path"],
                            "source_sha256": g["source_sha256"], "source_row": g["source_row"],
                        })  # fmt: skip
                        out["quality_issues"].append(_pg_issue(
                            "replay_link_quarantined", "warning", "open", key, g["season"],
                            f"result {g['result']} cannot belong to drawn {draw['match_id']}, and the team "
                            f"scores do not prove {replay['match_id']}: {why}",
                            "confirm the game on the player's AFL Tables page and re-import the row with the "
                            "right match"))  # fmt: skip
    return out


def _replay_move_proven(
    draw: dict[str, Any], replay: dict[str, Any], rows: dict[str, list[dict[str, Any]]], club: str,
    suspects: list[dict[str, Any]],
) -> tuple[bool, str]:  # fmt: skip
    def side(m: dict[str, Any]) -> str:
        return "home" if m["home_club_id"] == club else "away"

    def total(stat: str, gs: list[dict[str, Any]]) -> int:
        return sum(int(g[stat] or 0) for g in gs if g["club_id"] == club)

    rd, rr = rows[draw["match_id"]], rows[replay["match_id"]]
    if {g["player_id"] for g in suspects} & {g["player_id"] for g in rr if g["club_id"] == club}:
        return False, "the player already has a row on the replay"
    won = replay[f"{side(replay)}_score"] > replay[f"{'away' if side(replay) == 'home' else 'home'}_score"]
    if any(g["result"] != ("W" if won else "L") for g in suspects):
        return False, "the recorded result does not match the replay's outcome"
    tg_d, tg_r = draw[f"{side(draw)}_final_goals"], replay[f"{side(replay)}_final_goals"]
    s = total("goals", suspects)
    before = (total("goals", rd), total("goals", rr))
    after = (before[0] - s, before[1] + s)
    if tg_d is None or tg_r is None or after != (tg_d, tg_r) or before == (tg_d, tg_r):
        return False, f"player goals draw/replay {before} -> {after}, team goals ({tg_d}, {tg_r})"
    bd, br = draw[f"{side(draw)}_final_behinds"], replay[f"{side(replay)}_final_behinds"]
    sb = total("behinds", suspects)
    if bd is not None and br is not None and (total("behinds", rd) - sb > bd or total("behinds", rr) + sb > br):
        return False, "player behinds would exceed a team's behinds"
    return True, f"player goals draw/replay {before} -> {after} = team goals ({tg_d}, {tg_r})"


def _pg_issue(rule: str, severity: str, status: str, key: str, season: int, text: str,
              remediation: str | None) -> dict[str, Any]:  # fmt: skip
    digest = hashlib.sha256(f"{rule}\x1f{key}".encode()).hexdigest()[:24]
    return {"issue_id": f"qc:{digest}", "severity": severity, "status": status, "table_name": "player_games",
            "row_key": key, "source_path": None, "rule_id": rule, "explanation": text, "remediation": remediation,
            "acceptance_basis": None, "season": season}  # fmt: skip
