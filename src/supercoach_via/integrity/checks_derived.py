"""Family E (release side, continued): every published resource that is not a match, log or
player page, compared with values recomputed from its OWN authoritative input.

Resource inventory (``AUTHORITY``): each public resource type names the input it is derived
from and the check that compares it.

* ``release.derived``: team pages and index, history tables and index (including the
  ``legacy_v1`` rankings, recomputed from the pinned method file), list seasons and index,
  ``overview.json``, ``quality.json``, ``downloads.json`` and the numeric downloads (CSV rows,
  chart values, fan-pack members), all from the verified snapshot.
* ``release.forecast``: prediction sets and index against the prediction artifacts; accuracy
  reports against the evaluation artifacts (or, for the legacy archive, the snapshot's
  ``legacy_predictions``); live resources against the live monitor's accepted captures.
* ``release.content``: articles and assets, provenance only (source path and hash, the
  snapshot a generated note was checked against). Their prose is not audited.
* ``release.coverage``: every public file is claimed by an executed comparison or is
  provenance-only by type. A resource with no executed comparison makes the audit UNKNOWN;
  a missing comparator input is never a PASS.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import Severity
from supercoach_via.integrity import derived_expect as dx
from supercoach_via.integrity import public_compare as pc
from supercoach_via.integrity.capture import strict_json
from supercoach_via.integrity.context import AuditContext, CheckSkipped, CheckSpec, rule
from supercoach_via.integrity.report import Status, canonical_bytes

B = Severity.BLOCKING

#: resource type -> (authoritative input, comparing check); "provenance" = hash/provenance only
AUTHORITY: dict[str, tuple[str, str]] = {
    "release": ("the release build record (bound by release.artifact)", "release.artifact"),
    "match_index": ("snapshot matches", "release.public"),
    "match_detail": ("snapshot matches and player games", "release.public"),
    "player_season_games": ("snapshot player games", "release.public"),
    "player_index": ("snapshot players and player games", "release.public"),
    "player_detail": ("snapshot players and player games", "release.public"),
    "team_index": ("snapshot clubs and matches", "release.derived"),
    "team_season": ("snapshot matches and player games", "release.derived"),
    "history_index": ("snapshot player games", "release.derived"),
    "history_table": ("snapshot player games; legacy_v1 method file for rankings", "release.derived"),
    "lists_index": ("snapshot list imports (drafts, contracts, schools)", "release.derived"),
    "lists_season": ("snapshot list imports (drafts, contracts, schools)", "release.derived"),
    "overview": ("snapshot facts and the release's verified forecast and article index", "release.derived"),
    "quality": ("snapshot manifest and quality issues", "release.derived"),
    "downloads": ("the published download files", "release.derived"),
    "download": ("snapshot facts, legacy_v1 method file, the release's verified resources", "release.derived"),
    "prediction_index": ("prediction artifacts and snapshot fixture", "release.forecast"),
    "prediction_set": ("prediction artifacts and snapshot fixture", "release.forecast"),
    "accuracy_index": ("evaluation artifacts and snapshot legacy predictions", "release.forecast"),
    "accuracy_report": ("evaluation artifacts and snapshot legacy predictions", "release.forecast"),
    "live_index": ("live monitor accepted captures", "release.forecast"),
    "live_snapshot": ("live monitor accepted captures", "release.forecast"),
    "article_index": ("the release's article resources", "release.content"),
    "article": ("curated source documents (provenance only; prose unaudited)", "provenance"),
    "asset": ("curated source assets (bytes)", "release.content"),
    "rendered": ("chart images, Markdown reports and README rendered from compared values", "provenance"),
}
_UNAUDITED = {
    "article": "article prose (curated or generated) is not re-verified; provenance only",
    "rendered": "chart images and Markdown report prose are not parsed; their values are compared in "
    "charts.json and the source resources",
}

RULES = [
    rule(
        "release.derived_resource",
        "release.derived",
        B,
        "a team/history/list/summary/download resource the facts require is missing, unreadable, or not "
        "backed by any fact",
        "rebuild the release from the audited snapshot",
    ),
    rule(
        "release.team_value",
        "release.derived",
        B,
        "a team page or the team index differs from values recomputed from the matches and player games "
        "(ladder, position, form, fixtures, team-game means, leaders, five-year view, heuristic facts)",
        "rebuild the release; a stale or mis-joined team aggregate surfaces here",
    ),
    rule(
        "release.history_value",
        "release.derived",
        B,
        "a history table differs from leaders or legacy_v1 rankings recomputed from the facts (membership, "
        "rank, value, denominators)",
        "rebuild the release; never hand-edit a ranking",
    ),
    rule(
        "release.list_value",
        "release.derived",
        B,
        "a list season differs from the snapshot's draft, contract and school rows",
        "rebuild the release",
    ),
    rule(
        "release.summary_value",
        "release.derived",
        B,
        "overview.json or quality.json states a value the facts or the release's own resources contradict",
        "rebuild the release; a stale summary is shown on the home page",
    ),
    rule(
        "release.download_value",
        "release.derived",
        B,
        "a download (CSV rows, chart values, fan-pack member) differs from values recomputed from the facts",
        "rebuild the release",
    ),
    rule(
        "release.download_metadata",
        "release.derived",
        B,
        "downloads.json disagrees with the download files (bytes, sha256, rows, kind, membership)",
        "rebuild the release",
    ),
    rule(
        "release.forecast_resource",
        "release.forecast",
        B,
        "a prediction, accuracy or live resource is missing, unreadable, or has no pinned input behind it",
        "rebuild the release from the pinned artifacts",
    ),
    rule(
        "release.forecast_value",
        "release.forecast",
        B,
        "a published prediction, accuracy or live value differs from its pinned artifact or capture",
        "rebuild the release; a forecast must be the artifact's own numbers",
    ),
    rule(
        "release.content_resource",
        "release.content",
        B,
        "an article or asset is missing from its index, listed without a resource, or an unreferenced asset",
        "rebuild the release",
    ),
    rule(
        "release.unclassified_resource",
        "release.coverage",
        B,
        "a public file has a type no comparison knows (it could publish anything unchecked)",
        "remove the file or add its type to the resource inventory with a comparator",
    ),
    rule(
        "release.content_provenance",
        "release.content",
        B,
        "an article's provenance (source path and hash, or checked snapshot) does not match its input",
        "rebuild the release from the current curated sources",
    ),
]

_TEAM_PAGE = re.compile(r"^teams/[^/]+/(\d{4})\.json$")


def _kind(rel: str) -> str:
    from supercoach_via.publish.release import model_for_path

    model = model_for_path(rel)
    if model is not None:
        return model
    if rel.startswith("assets/"):
        return "asset"
    if rel.startswith("downloads/") and rel.endswith((".png", ".svg", ".md")):
        return "rendered"
    return "download" if rel.startswith("downloads/") else "other"


def _claim(ctx: AuditContext, paths: Any, how: str) -> None:
    claims = ctx.coverage.setdefault("_claims", {})
    for p in paths:
        claims.setdefault(p, how)


def _subset(inv: dict[str, tuple[str, int]], paths: Any) -> dict[str, tuple[str, int]]:
    return {p: inv.get(p, ("absent", 0)) for p in sorted(set(paths))}


def _digest(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _optional(ctx: AuditContext, *names: str) -> set[str]:
    have = set()
    for n in names:
        if ctx.has(n):
            try:
                ctx.need(n)
                have.add(n)
            except CheckSkipped:
                pass
    return have


def _public_json(ctx: AuditContext, rel: str) -> Any:
    assert ctx.release is not None
    try:
        return strict_json(ctx.release.read_public(rel))
    except (KeyError, ValueError, UnicodeDecodeError):
        return None


def _run_units(ctx: AuditContext, units: list[Any], counters: dict[str, int]) -> set[str]:
    from supercoach_via.integrity.checks_release import _run

    assert ctx.release is not None
    expected: set[str] = set()
    for r in _run(ctx, units, ctx.collector.rules, ctx.release.release_dir):
        ctx.collector.extend(r.findings)
        expected.update(r.expected_paths)
        for k, v in r.examined.items():
            counters[k] += v
            ctx.count(k, v)
    return expected


def _unexpected(ctx: AuditContext, inv: dict[str, Any], expected: set[str], owned: Any, rule_id: str) -> None:
    for rel in sorted(inv):
        if owned(rel) and rel not in expected:
            ctx.add(
                rule_id, f"resource:{rel}", expected="absent", actual="present", message="no fact behind this resource"
            )


# ---------------------------------------------------------------------------
# release.derived
# ---------------------------------------------------------------------------


def check_derived(ctx: AuditContext) -> list[str]:
    from supercoach_via.integrity.checks_data import _coverage

    cap = ctx.release
    assert cap is not None
    if ctx.snapshot is None or ctx.snapshot.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no verifiable snapshot to compare the release with")
    ctx.need("matches", "player_games", "players", "clubs")
    optional = _optional(
        ctx, "seasons", "quality_issues", "draft_events", "contract_observations", "school_observations"
    )
    release_meta = _public_json(ctx, "release.json")
    if not isinstance(release_meta, dict):
        raise CheckSkipped(Status.UNKNOWN, "release.json is unreadable; the release's season is unknown")
    con = ctx.db()
    inv = {rel: (f.sha256, f.bytes) for rel, f in cap.public.items()}
    frag = ctx.snapshot.fragment_digest
    manifest = ctx.snapshot.manifest
    snapshot_id = str(ctx.snapshot.snapshot_id)
    recorded_from = {str(k): int(v) for k, v in _coverage(ctx).coverage.recorded_from.items()}
    cfg = dx.ranking_config(
        ctx.policy.config_dir, captured_bytes=ctx.external.read(ctx.policy.config_dir / "ranking_legacy_v1.toml")
    )
    method = {"ranking_config": cfg.config_hash(), "recorded_from": recorded_from}
    names = {str(c): str(n) for c, n in ctx.rows("SELECT club_id, name FROM clubs ORDER BY 1")}
    base = {t: frag(t) for t in ("clubs", "players")}
    rules = {"value_rule": "", "resource_rule": "release.derived_resource"}
    units: list[Any] = []
    counters: dict[str, int] = defaultdict(int)

    def unit(unit_id: str, value_rule: str, build: Any, deps: Any, paths: Any, season: int | None = None) -> None:
        def payload() -> dict[str, Any]:
            return {**rules, "value_rule": value_rule, "docs": build(), "unit_id": unit_id, "season": season}

        units.append((unit_id, pc.docs_unit, payload, _digest(deps), _subset(inv, paths)))

    # teams
    unit(
        "teams:index",
        "release.team_value",
        lambda: {"teams/index.json": dx.team_index(con)},
        {"matches": frag("matches"), **base},
        ["teams/index.json"],
    )
    seasons = [int(s) for (s,) in ctx.rows("SELECT DISTINCT season FROM matches ORDER BY 1")]
    clubs_by = {
        int(s): [str(c) for c in cs]
        for s, cs in ctx.rows(
            """SELECT season, list(DISTINCT c ORDER BY c) FROM (SELECT season, home_club_id AS c FROM matches
           UNION ALL SELECT season, away_club_id FROM matches) GROUP BY season"""
        )
    }

    def team_docs(season: int) -> dict[str, Any]:
        window = list(range(season - dx.FIVE_YEARS, season + 1))
        matches = ctx.records(
            "SELECT * FROM matches WHERE season BETWEEN ? AND ? ORDER BY match_id", [window[0], window[-1]]
        )
        ladders = {y: dx.ladder([m for m in matches if m["season"] == y], names) for y in window}
        return dx.team_season_docs(con, season, names, ladders, matches)

    for season in seasons:
        window = {str(y) for y in range(season - dx.FIVE_YEARS, season + 1)}
        unit(
            f"teams:{season}",
            "release.team_value",
            lambda s=season: team_docs(s),
            {"matches": frag("matches", window), "player_games": frag("player_games", {str(season)}), **base},
            [f"teams/{c}/{season}.json" for c in clubs_by.get(season, [])],
            season,
        )
    # history (including the legacy_v1 rankings) and the ranking downloads share one computation
    memo: dict[str, Any] = {}

    def ranking() -> dict[str, Any]:
        if "ranking" not in memo:
            memo["ranking"] = dx.rankings(con, cfg, snapshot_id)
        return memo["ranking"]  # type: ignore[no-any-return]

    facts: dict[str, Any] = {t: frag(t) for t in ("matches", "player_games", "players", "clubs")}
    facts["seasons"] = frag("seasons") if "seasons" in optional else None
    history_paths = [p for p in inv if p.startswith("history/")] + ["history/index.json"]
    unit(
        "history",
        "release.history_value",
        lambda: dx.history_docs(con, recorded_from, ranking()),
        {**facts, **method, "snapshot": ctx.snapshot.snapshot_id},
        history_paths,
    )
    # lists
    list_tables = {t for t in ("draft_events", "contract_observations", "school_observations") if t in optional}
    unit(
        "lists",
        "release.list_value",
        lambda: dx.lists_expected(con, list_tables | {"clubs"}),
        {t: frag(t) for t in (*sorted(list_tables), "clubs")},
        [p for p in inv if p.startswith("lists/")] + ["lists/index.json"],
    )
    # overview and quality
    season = int(release_meta.get("season") or 0)
    forecast_index = _public_json(ctx, "predictions/index.json") or {}
    current_rel = forecast_index.get("current") if isinstance(forecast_index, dict) else None
    current_set = _public_json(ctx, current_rel) if isinstance(current_rel, str) else None
    article_index = _public_json(ctx, "articles/index.json") or {}
    forecast = {"status": forecast_index.get("status"), "reason": forecast_index.get("reason")}
    status = manifest.status.value if hasattr(manifest.status, "value") else str(manifest.status)
    counts = {k: int(v.row_count) for k, v in manifest.tables.items()}
    read_by_summaries = [
        "overview.json",
        "quality.json",
        "release.json",
        "predictions/index.json",
        "articles/index.json",
        *([current_rel] if isinstance(current_rel, str) else []),
    ]

    def summaries() -> dict[str, Any]:
        season_matches = ctx.records("SELECT * FROM matches WHERE season = ? ORDER BY match_id", [season])
        return {
            "overview.json": dx.overview_expected(
                con,
                release=release_meta,
                season_matches=season_matches,
                names=names,
                dataset_status=status,
                has_seasons="seasons" in optional,
                forecast=forecast,
                current_set=current_set,
                articles=list(article_index.get("articles") or []) if isinstance(article_index, dict) else [],
            ),
            "quality.json": dx.quality_expected(
                con,
                dataset_status=status,
                demo=bool(release_meta.get("demo")),
                table_counts=counts,
                has_issues="quality_issues" in optional,
                snapshot_id=snapshot_id,
                forecast=forecast,
            ),
        }

    unit(
        "summaries",
        "release.summary_value",
        summaries,
        {
            **facts,
            "quality_issues": frag("quality_issues") if "quality_issues" in optional else None,
            "counts": counts,
            "status": status,
        },
        read_by_summaries,
    )
    # downloads (numeric content); downloads.json metadata is checked below
    download_paths = [p for p in inv if p.startswith("downloads/")]
    read_by_downloads = [
        *download_paths,
        "players/index.json",
        "release.json",
        "predictions/index.json",
        *([current_rel] if isinstance(current_rel, str) else []),
    ]
    unit(
        "downloads",
        "release.download_value",
        lambda: _download_docs(
            ctx, release_meta, names, status, ranking(), recorded_from, current_set, forecast, inv, season
        ),
        {**facts, **method, "snapshot": ctx.snapshot.snapshot_id, "status": status},
        read_by_downloads,
    )
    expected = _run_units(ctx, units, counters)
    _unexpected(
        ctx,
        inv,
        expected,
        lambda r: (
            r.startswith(("teams/", "history/", "lists/"))
            or (
                r.startswith("downloads/")
                and r.endswith((".csv", ".json", ".zip"))
                and r != "downloads/accuracy-rows.csv"
            )
        ),
        "release.derived_resource",
    )
    _download_metadata(ctx, inv, release_meta, season)
    claimed = expected | {"downloads.json"}
    _claim(ctx, [p for p in claimed if p in inv], "release.derived")
    ctx.coverage["derived_compare"] = {k: counters[k] for k in sorted(counters)}
    return []


def _as_of(release_meta: dict[str, Any], matches: list[dict[str, Any]], season: int) -> str:
    through = dx.coverage_through(matches, season)
    return f"{'DEMO ' if release_meta.get('demo') else ''}{through or season}"


def _download_metadata(
    ctx: AuditContext, inv: dict[str, tuple[str, int]], release_meta: dict[str, Any], season: int
) -> None:
    import csv
    import io

    doc = _public_json(ctx, "downloads.json")
    if not isinstance(doc, dict):
        ctx.add("release.derived_resource", "resource:downloads.json", expected="readable downloads.json")
        return
    matches = ctx.records("SELECT * FROM matches WHERE season = ? ORDER BY match_id", [season])
    as_of = _as_of(release_meta, matches, season)
    kinds = {".csv": "csv", ".png": "png", ".svg": "svg", ".zip": "zip", ".json": "json", ".md": "md"}
    if doc.get("release_id") != release_meta.get("release_id"):
        ctx.add(
            "release.download_metadata",
            "resource:downloads.json",
            field="release_id",
            expected=release_meta.get("release_id"),
            actual=doc.get("release_id"),
        )
    items = [i for i in doc.get("items") or [] if isinstance(i, dict)]
    listed = [i.get("path") for i in items]
    for rel in sorted(p for p in inv if p.startswith("downloads/") and p not in listed):
        ctx.add("release.download_metadata", f"resource:{rel}", field="items", expected="listed", actual="absent")
    if len(set(i.get("key") for i in items)) != len(items):
        ctx.add("release.download_metadata", "resource:downloads.json", field="items.key", actual="duplicate keys")
    for n, item in enumerate(items):
        rel = str(item.get("path"))
        where = f"items[{n}]"
        ctx.count("download_items")
        info = inv.get(rel)
        if info is None:
            ctx.add(
                "release.download_metadata",
                "resource:downloads.json",
                field=f"{where}.path",
                expected="a published file",
                actual=rel,
            )
            continue
        want: dict[str, Any] = {
            "bytes": info[1],
            "sha256": info[0],
            "as_of": as_of,
            "kind": kinds.get(Path(rel).suffix, "?"),
        }
        if rel.endswith(".csv"):
            text = ctx.release.read_public(rel).decode("utf-8", "replace")  # type: ignore[union-attr]
            want["rows"] = max(0, len(list(csv.reader(io.StringIO(text)))) - 1)
        else:
            want["rows"] = None
        for k, v in want.items():
            same = item.get(k) is None if v is None else pc._scalar_equal(v, item.get(k))
            if not same:
                ctx.add(
                    "release.download_metadata",
                    "resource:downloads.json",
                    field=f"{where}.{k}",
                    expected=v,
                    actual=item.get(k),
                    evidence={"path": rel},
                )


def _download_docs(
    ctx: AuditContext,
    release_meta: dict[str, Any],
    names: dict[str, str],
    status: str,
    ranking: dict[str, Any],
    recorded_from: dict[str, int],
    current_set: Any,
    forecast: dict[str, Any],
    inv: dict[str, tuple[str, int]],
    season: int,
) -> dict[str, Any]:
    con = ctx.db()
    docs: dict[str, Any] = {}
    index = _public_json(ctx, "players/index.json") or {}
    docs["downloads/players.csv"] = dx.players_csv(con, list(index.get("players") or []))
    docs["downloads/all-time-top-100-scores.csv"] = {
        "header": ["player", "all_time_score"],
        "rows": [[k, s] for _p, k, s in ranking["all_time"]],
        "key": None,
    }
    docs["downloads/all-time-top-100.csv"] = dx.biography_csv(con, ranking["all_time"])
    final = [s for s in ranking["yearly"] if s not in ranking["provisional"]]
    if final:
        s = max(final)
        docs[f"downloads/yearly-top-100-{s}.csv"] = {
            "header": ["player", "score", "percentile_rank", "games_played"],
            "rows": [[e["player_key"], e["score"], e["percentile_rank"], e["games"]] for e in ranking["yearly"][s]],
            "key": None,
        }
    docs["downloads/era-summary.csv"] = dx.era_summary_csv(con, recorded_from)
    proxy = dx.brownlow_proxy_csv(con, season)
    if proxy is not None:
        docs[f"downloads/brownlow-proxy-{season}.csv"] = proxy
    rows = list((current_set or {}).get("rows") or []) if isinstance(current_set, dict) else []
    if rows:
        docs.update(_prediction_downloads(ctx, release_meta, current_set, rows))
    leaders = dx.season_leaders(con, "disposals", season)
    ladder = dx.ladder(ctx.records("SELECT * FROM matches WHERE season = ?", [season]), names)
    top = sorted(rows, key=lambda r: (-r["predicted_disposals"], r["prediction_id"]))[:10]
    charts, missing = [], []
    for key, plotted in (
        ("season_disposal_leaders", [[r["name"], r["value"]] for r in leaders]),
        ("ladder_points", [[r["name"], float(r["premiership_points"])] for r in ladder]),
        ("top_predictions", [[r["player_name"], r["predicted_disposals"]] for r in top]),
    ):
        rel = f"downloads/charts/{key.replace('_', '-')}.png"
        if plotted:
            charts.append(
                {
                    "key": key,
                    "status": "available",
                    "path": rel,
                    "title": dx.skip(),
                    "unit": dx.skip(),
                    "alt_text": dx.skip("rendered description"),
                    "plotted": plotted,
                }
            )
        else:
            missing.append((rel, {"key": key, "status": "unavailable", "reason": dx.skip("build-time reason")}))
    docs["downloads/charts.json"] = {
        "release_id": release_meta.get("release_id"),
        "snapshot_id": release_meta.get("snapshot_id"),
        "charts": charts + [m for _r, m in sorted(missing, key=lambda x: x[0])],
    }
    members = {
        p.removeprefix("downloads/"): inv[p][0]
        for p in inv
        if p.startswith("downloads/") and p != "downloads/fan-pack.zip"
    }
    cutoff = (current_set or {}).get("forecast_cutoff") if rows else None
    docs["downloads/fan-pack.zip"] = {
        "members": members,
        "meta": {
            "kind": "supercoach_via_fan_pack",
            "release_id": release_meta.get("release_id"),
            "snapshot_id": release_meta.get("snapshot_id"),
            "generated_at": release_meta.get("generated_at"),
            "demo": bool(release_meta.get("demo")),
            "dataset_status": "demo" if release_meta.get("demo") else status,
            "forecast_status": forecast["status"],
            "forecast_reason": forecast["reason"],
            "forecast_cutoff": cutoff,
        },
    }
    return docs


def _prediction_downloads(
    ctx: AuditContext, release_meta: dict[str, Any], pset: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """The forecast downloads restate the published current set (itself compared with its artifact)."""
    from supercoach_via.publish.view_models import PredictionRow

    fields = list(PredictionRow.model_fields)
    table = [[";".join(r[f]) if f == "warnings" else r[f] for f in fields] for r in rows]
    people = {p: (f or "", la or "") for p, f, la in ctx.rows("SELECT player_id, first_name, last_name FROM players")}
    legacy, side = [], []
    for i, r in enumerate(sorted(rows, key=lambda r: (-r["predicted_disposals"], r["prediction_id"]))):
        first, last = people.get(r["player_id"], ("", ""))
        name = f"{last} {first}".strip() if first and last else r["player_name"]
        legacy.append([name, r["club_name"], round(r["predicted_disposals"])])
        side.append(
            {
                "line": i + 2,
                "player": name,
                "team": r["club_name"],
                "prediction_id": r["prediction_id"],
                "player_id": r["player_id"],
                "club_id": r["club_id"],
                "match_id": r["match_id"],
                "predicted_disposals": r["predicted_disposals"],
            }
        )
    return {
        "downloads/predictions.csv": {"header": fields, "rows": table, "key": None},
        "downloads/predictions-legacy.csv": {
            "header": ["player", "team", "predicted_disposals"],
            "rows": legacy,
            "key": None,
        },
        "downloads/predictions-legacy.manifest.json": {
            "kind": "legacy_forward_prediction_csv_sidecar",
            "legacy_file": "downloads/predictions-legacy.csv",
            "columns": ["player", "team", "predicted_disposals"],
            "display_rounding": dx.skip(),
            "release_id": release_meta.get("release_id"),
            "snapshot_id": release_meta.get("snapshot_id"),
            "prediction_run_id": rows[0]["prediction_run_id"],
            "model_id": (release_meta.get("forecast") or {}).get("model_id"),
            "season": pset.get("season"),
            "stage_id": pset.get("stage_id"),
            "forecast_cutoff": pset.get("forecast_cutoff"),
            "rows": side,
        },
    }


# ---------------------------------------------------------------------------
# release.forecast: predictions, accuracy and live against their pinned inputs
# ---------------------------------------------------------------------------


def check_forecast(ctx: AuditContext) -> list[str]:
    cap = ctx.release
    assert cap is not None
    if ctx.snapshot is None or ctx.snapshot.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no verifiable snapshot to compare the release with")
    ctx.need("matches", "player_games", "players", "clubs")
    _optional(ctx, "venues", "legacy_predictions")
    inv = {rel: (f.sha256, f.bytes) for rel, f in cap.public.items()}
    unknown: list[str] = []
    compared: set[str] = set()
    compared |= _forecast_predictions(ctx, inv, unknown)
    compared |= _forecast_accuracy(ctx, inv, unknown)
    compared |= _forecast_live(ctx, inv, unknown)
    _claim(ctx, sorted(compared), "release.forecast")
    return unknown


def _diff(ctx: AuditContext, rule_id: str, rel: str, want: Any, have: Any) -> None:
    counts: dict[str, int] = {}
    for path, expected, actual in pc.diff_value(want, have, "", counts):
        ctx.add(
            rule_id,
            f"resource:{rel}",
            field=path or "(document)",
            expected=pc._jsonable(expected),
            actual=pc._jsonable(actual),
        )
    for k, v in counts.items():
        ctx.count(k, v)


def _forecast_predictions(ctx: AuditContext, inv: dict[str, Any], unknown: list[str]) -> set[str]:
    import pyarrow.parquet as pq

    from supercoach_via.integrity.checks_models import _inputs
    from supercoach_via.publish.builder import classify_prediction_targets

    index = _public_json(ctx, "predictions/index.json")
    if not isinstance(index, dict):
        if "predictions/index.json" in inv:
            ctx.add("release.forecast_resource", "resource:predictions/index.json", message="unreadable")
        return set()
    _bundles, preds = _inputs(ctx)
    by_run = {k: v for k, v in preds.items() if v.get("manifest") is not None}
    names = dict(ctx.rows("SELECT club_id, name FROM clubs"))
    players = dict(ctx.rows("SELECT player_id, display_name FROM players"))
    venues = dict(ctx.rows("SELECT venue_id, name FROM venues")) if ctx.has("venues") else {}
    match = {m["match_id"]: m for m in ctx.records("SELECT * FROM matches")}
    compared: set[str] = set()
    sets: list[dict[str, Any]] = []
    for rel in sorted(p for p in inv if p.startswith("predictions/") and p != "predictions/index.json"):
        doc = _public_json(ctx, rel)
        if not isinstance(doc, dict):
            ctx.add("release.forecast_resource", f"resource:{rel}", message="unreadable")
            continue
        runs = sorted({str(r.get("prediction_run_id")) for r in doc.get("rows") or [] if isinstance(r, dict)})
        entry = next((by_run[r] for r in runs if r in by_run), None)
        if entry is None:
            unknown.append(f"{rel}: prediction artifact {', '.join(map(str, runs)) or '?'} not supplied")
            continue
        m = entry["manifest"]
        import pyarrow as pa

        table = pq.read_table(pa.BufferReader(ctx.external.read(entry["dir"] / m.rows_file))).to_pylist()
        targets = [match.get(mid) for mid in m.target_matches]
        state = classify_prediction_targets(
            [
                t["status"]
                if t is not None and int(t["season"]) == int(c["season"]) and t["stage_id"] == c["stage_id"]
                else None
                for t, c in zip(targets, m.target_matches.values(), strict=True)
            ]
        )
        want = {
            "season": m.season,
            "stage_id": m.stage_id,
            "stage_label": m.stage_label,
            "status": state if state is not None else "(rejected: targets absent or re-staged)",
            "reason": "targets_started_or_played" if state == "expired" else m.reason,
            "units": "disposals",
            "generated_at": dx.iso(m.generated_at),
            "forecast_cutoff": dx.iso(m.forecast_cutoff),
            "target_matches": [
                dx.summary(t, names) if t else {"match_id": mid}
                for mid, t in zip(m.target_matches, targets, strict=True)
            ],
            "model": {
                "model_id": m.model_id,
                **{
                    k: dx.skip("bundle description (models.bundles verifies the bundle)")
                    for k in ("kind", "name", "description", "trained_cutoff", "promoted", "promotion_note")
                },
            },
            "interval": dx.skip("bundle interval description (models.bundles verifies the bundle)"),
            "rows": [_prediction_row(r, players, names, venues, match) for r in table],
            "omissions": [{"reason": k, "count": int(v), "detail": None} for k, v in sorted(m.omissions.items())],
        }
        _diff(ctx, "release.forecast_value", rel, want, doc)
        ctx.count("prediction_sets")
        compared.add(rel)
        first = min((str(t["match_date"]) for t in targets if t and t["match_date"]), default="9999-12-31")
        sets.append(
            {
                "rel": rel,
                "season": m.season,
                "first": first,
                "stage_id": m.stage_id,
                "entry": {
                    "season": m.season,
                    "stage_id": m.stage_id,
                    "stage_label": m.stage_label,
                    "status": want["status"],
                    "resource": rel,
                    "rows": len(table),
                },
            }
        )
    if any(p.startswith("predictions/") and p != "predictions/index.json" and p not in compared for p in inv):
        return compared  # an index over sets we could not verify is itself unverifiable
    sets.sort(key=lambda s: (s["season"], s["first"], s["stage_id"]))
    current = next((s["rel"] for s in sets if s["entry"]["status"] == "available"), None)
    reason: Any
    if current is not None:
        status, reason = "available", None
    elif any(s["entry"]["status"] == "expired" for s in sets):
        status, reason = "expired", "prediction_targets_started_or_played"
    else:
        scheduled = ctx.rows("SELECT count(*) FROM matches WHERE status = 'scheduled'")[0][0]
        allowed = ["no_valid_prediction_artifact", "no_prediction_artifact" if scheduled else "no_valid_future_fixture"]
        allowed += sorted({str(v["manifest"].reason) for v in by_run.values() if v["manifest"].reason})
        status, reason = "unavailable", {"$one_of": allowed}
    _diff(
        ctx,
        "release.forecast_value",
        "predictions/index.json",
        {"current": current, "status": status, "reason": reason, "sets": [s["entry"] for s in sets]},
        index,
    )
    return compared | {"predictions/index.json"}


def _prediction_row(
    r: dict[str, Any], players: dict[str, str], clubs: dict[str, str], venues: dict[str, str], match: dict[str, Any]
) -> dict[str, Any]:
    def f(v: Any) -> Any:
        return None if v is None or (isinstance(v, float) and v != v) else float(v)

    local = (match.get(r["match_id"]) or {}).get("local_start")
    opp = r.get("opponent_club_id")
    return {
        "prediction_id": r["prediction_id"],
        "prediction_run_id": r["prediction_run_id"],
        "snapshot_id": r["snapshot_id"],
        "model_id": r["model_id"],
        "player_id": r["player_id"],
        "player_name": players.get(r["player_id"], r["player_id"]),
        "club_id": r["club_id"],
        "club_name": clubs.get(r["club_id"], r["club_id"]),
        "opponent_club_id": opp,
        "opponent_name": clubs.get(opp) if opp else None,
        "match_id": r["match_id"],
        "season": int(r["season"]),
        "stage_id": r["stage_id"],
        "stage_label": r["stage_label"],
        "scheduled_at": dx.iso(r.get("scheduled_at")),
        "scheduled_local": local if local not in (None, "None") else None,
        "venue": venues.get(r["venue_id"]) if r.get("venue_id") else None,
        "forecast_cutoff": dx.iso(r["forecast_cutoff"]),
        "origin": r["origin"],
        "generated_at": dx.iso(r["generated_at"]),
        "selection_status": r["selection_status"],
        "eligibility_basis": r["eligibility_basis"],
        "history_games": int(r["history_games"]),
        "recent_mean_5": f(r.get("recent_mean_5")),
        "predicted_disposals": float(r["predicted_disposals"]),
        "interval_low": f(r.get("interval_low")),
        "interval_high": f(r.get("interval_high")),
        "interval_level": f(r.get("interval_level")),
        "interval_method": r.get("interval_method") if isinstance(r.get("interval_method"), str) else None,
        "warnings": list(r.get("warnings") or []),
    }


def _metrics(pred: list[float], actual: list[float]) -> dict[str, Any] | None:
    """Pooled error metrics recomputed from scored rows (the checker's own arithmetic)."""
    import math

    n = len(pred)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "median_ae": None, "within_5": None, "within_10": None}
    err = [p - a for p, a in zip(pred, actual, strict=True)]
    ae = sorted(abs(e) for e in err)
    mid = n // 2
    median = ae[mid] if n % 2 else (ae[mid - 1] + ae[mid]) / 2
    return {
        "n": n,
        "mae": sum(ae) / n,
        "rmse": math.sqrt(sum(e * e for e in err) / n),
        "bias": sum(err) / n,
        "median_ae": median,
        "within_5": sum(1 for x in ae if x <= 5) / n,
        "within_10": sum(1 for x in ae if x <= 10) / n,
    }


def _forecast_accuracy(ctx: AuditContext, inv: dict[str, Any], unknown: list[str]) -> set[str]:
    import pyarrow.parquet as pq

    reports = sorted(p for p in inv if p.startswith("accuracy/") and p != "accuracy/index.json")
    compared: set[str] = set()
    evaluations = {}
    for d in ctx.evaluation_dirs:
        try:
            import pyarrow as pa

            summary = strict_json(ctx.external.read(d / "evaluation.json"))
            rows = pq.read_table(pa.BufferReader(ctx.external.read(d / "scored_rows.parquet"))).to_pylist()
        except (OSError, ValueError) as exc:
            unknown.append(f"evaluation {d.name} unreadable: {exc}"[:200])
            continue
        evaluations[summary["evaluation_id"]] = (summary, rows)
    acc_rows: list[list[Any]] = []
    for ev_id, (s, rows) in sorted(evaluations.items()):
        seasons = sorted({int(r["season"]) for r in rows})
        season = seasons[0] if len(seasons) == 1 else None
        rel = f"accuracy/{s['model_id']}/{season if season is not None else 'all'}-{s['origin']}.json"
        doc = _public_json(ctx, rel)
        if doc is None:
            ctx.add(
                "release.forecast_resource",
                f"resource:{rel}",
                expected="present",
                actual="absent",
                evidence={"evaluation_id": ev_id},
            )
            continue
        pops = s["populations"]
        want = {
            "model_id": s["model_id"],
            "baseline_id": s["baseline_model_id"],
            "season": season,
            "origin": s["origin"],
            "label": s["label"],
            "headline": _metrics([float(r["predicted_disposals"]) for r in rows], [float(r["actual"]) for r in rows]),
            "baseline_headline": s["baseline_headline"],
            "mean_of_rounds_mae": s["mean_of_rounds_mae"],
            "populations": {
                **{k: int(pops[k]) for k in ("intended", "predicted", "joined", "played", "missing", "excluded")},
                "exclusion_reasons": {k: int(v) for k, v in pops["exclusion_reasons"].items()},
            },
            "cohorts": [
                {
                    "dimension": c["dimension"] + (" (post-hoc)" if c["post_hoc"] else ""),
                    "cohort": c["cohort"],
                    "model": c["model"],
                    "baseline": c["baseline"],
                    "sufficient": c["sufficient"],
                }
                for c in s["cohorts"]
            ],
            "interval": dx.skip("bundle interval description"),
            "interval_coverage": s["interval"].get("coverage"),
            "interval_median_width": s["interval"].get("median_width"),
            "promotion": dx.skip("promotion note"),
            "notes": s["notes"],
            "rows_resource": "downloads/accuracy-rows.csv" if rows else None,
        }
        _diff(ctx, "release.forecast_value", rel, want, doc)
        compared.add(rel)
        for r in rows:
            p, a = float(r["predicted_disposals"]), float(r["actual"])
            bp = r.get("baseline_prediction")
            acc_rows.append(
                [
                    ev_id,
                    s["origin"],
                    s["model_id"],
                    r["prediction_id"],
                    r["prediction_run_id"],
                    r["player_id"],
                    r["club_id"],
                    r["match_id"],
                    int(r["season"]),
                    r["stage_id"],
                    p,
                    a,
                    p - a,
                    abs(p - a),
                    None if bp is None or bp != bp else float(bp),
                ]
            )
    legacy = "accuracy/legacy_unknown/all-legacy_unknown.json"
    if legacy in inv:
        if ctx.has("legacy_predictions"):
            _diff(ctx, "release.forecast_value", legacy, _legacy_accuracy(ctx), _public_json(ctx, legacy))
            compared.add(legacy)
        else:
            unknown.append(f"{legacy}: the snapshot has no legacy_predictions table")
    for rel in reports:
        if rel not in compared and rel != legacy:
            unknown.append(f"{rel}: no evaluation artifact supplied (--evaluation)")
    if "downloads/accuracy-rows.csv" in inv:
        if acc_rows and not [r for r in reports if r not in compared]:
            want_csv = {
                "header": [
                    "evaluation_id",
                    "origin",
                    "model_id",
                    "prediction_id",
                    "prediction_run_id",
                    "player_id",
                    "club_id",
                    "match_id",
                    "season",
                    "stage_id",
                    "predicted_disposals",
                    "actual",
                    "error",
                    "abs_error",
                    "baseline_prediction",
                ],
                "rows": acc_rows,
                "key": None,
            }
            counts: dict[str, int] = {}
            data = ctx.release.read_public("downloads/accuracy-rows.csv")  # type: ignore[union-attr]
            for path, e, a in pc.diff_csv(want_csv, data, counts):
                ctx.add(
                    "release.forecast_value", "resource:downloads/accuracy-rows.csv", field=path, expected=e, actual=a
                )
            compared.add("downloads/accuracy-rows.csv")
        else:
            unknown.append("downloads/accuracy-rows.csv: its evaluation artifacts were not all supplied")
    index = _public_json(ctx, "accuracy/index.json")
    if isinstance(index, dict) and not [r for r in reports if r not in compared]:
        release_meta = _public_json(ctx, "release.json") or {}
        entries = []
        for rel in reports:
            rep = _public_json(ctx, rel) or {}
            entries.append(
                {
                    "model_id": rep.get("model_id"),
                    "season": rep.get("season"),
                    "origin": rep.get("origin"),
                    "label": rep.get("label"),
                    "resource": rel,
                }
            )
        want_index = {
            "champion_model_id": (release_meta.get("forecast") or {}).get("model_id")
            or dx.skip("promoted bundle chosen at build time"),
            "baseline_model_id": dx.skip("baseline bundle chosen at build time"),
            "reports": entries,
            "model_card": dx.skip("model card prose from the bundle manifest"),
        }
        _diff(ctx, "release.forecast_value", "accuracy/index.json", want_index, index)
        compared.add("accuracy/index.json")
    return compared


def _legacy_accuracy(ctx: AuditContext) -> dict[str, Any]:
    """The imported legacy forecast archive, re-joined to the facts by its documented rules."""
    lp = ctx.records("SELECT * FROM legacy_predictions")
    sides: dict[tuple[int, str, Any], list[str]] = defaultdict(list)
    for mid, season, label, home, away in ctx.rows(
        "SELECT match_id, season, stage_label, home_club_id, away_club_id FROM matches WHERE status = 'complete'"
    ):
        sides[(int(season), str(label), home)].append(mid)
        sides[(int(season), str(label), away)].append(mid)
    actual = {(m, p): d for m, p, d in ctx.rows("SELECT match_id, player_id, disposals FROM player_games")}
    reasons: dict[str, int] = defaultdict(int)
    pred, act = [], []
    for r in lp:
        if r.get("player_id") is None:
            reasons["unresolved_identity"] += 1
            continue
        season = r.get("claimed_season")
        if season is None:
            ts = str(r.get("claimed_timestamp") or "")
            if not ts[:4].isdigit():
                reasons["no_claimed_season"] += 1
                continue
            season = int(ts[:4])
        cands = sides.get((int(season), str(r.get("claimed_round_label")), r.get("club_id")))
        if not cands or len(cands) != 1:
            reasons["claimed_round_unmatched_or_ambiguous"] += 1
            continue
        key = (cands[0], r["player_id"])
        if key not in actual:
            reasons["did_not_play_claimed_round"] += 1
            continue
        if actual[key] is None:
            reasons["unknown_actual"] += 1
            continue
        pred.append(float(r["predicted_value"]))
        act.append(float(actual[key]))
    n = len(lp)
    return {
        "model_id": "legacy_unknown",
        "baseline_id": None,
        "season": None,
        "origin": "legacy_unknown",
        "label": dx.skip(),
        "headline": _metrics(pred, act),
        "baseline_headline": None,
        "mean_of_rounds_mae": None,
        "populations": {
            "intended": n,
            "predicted": n,
            "joined": len(pred),
            "played": len(pred),
            "missing": n - len(pred),
            "excluded": 0,
            "exclusion_reasons": dict(reasons),
        },
        "cohorts": [],
        "interval": dx.skip("no interval for the archive"),
        "interval_coverage": None,
        "interval_median_width": None,
        "promotion": dx.skip(),
        "notes": dx.skip(),
        "rows_resource": None,
    }


def _forecast_live(ctx: AuditContext, inv: dict[str, Any], unknown: list[str]) -> set[str]:
    from supercoach_via.publish.view_models import LiveSnapshot

    published = sorted(p for p in inv if p.startswith("live/") and p != "live/index.json")
    index = _public_json(ctx, "live/index.json")
    if ctx.live_root is None:
        if published:
            unknown.append(
                f"{len(published)} live snapshot(s) published but no live capture root supplied (--live-root)"
            )
            return set()
        if isinstance(index, dict) and not index.get("matches"):
            return {"live/index.json"}  # an empty live index states nothing to compare
        unknown.append("live/index.json: no live capture root supplied (--live-root)")
        return set()
    entries, compared = [], {"live/index.json"}
    for state_path in ctx.external.select(ctx.live_root, "*/state.json"):
        try:
            state = strict_json(ctx.external.read(state_path))
        except ValueError as exc:
            raise CheckSkipped(Status.UNKNOWN, f"live state unavailable: {exc}") from exc
        if not state.get("accepted"):
            continue
        rel = f"live/{state['source_game_id']}/latest.json"
        raw = ctx.external.read(state_path.parent / "snapshots" / f"{state['accepted'][-1]}.json")
        snap = LiveSnapshot.model_validate_json(raw).model_dump(mode="json")
        stamps = state.get("accepted_at") or []
        entries.append(
            {
                "source_game_id": state["source_game_id"],
                "match_id": state.get("match_id"),
                "label": state.get("label") or state["source_game_id"],
                "last_fetched_at": dx.iso(datetime.fromisoformat(stamps[-1])) if stamps else snap["fetched_at"],
                "final": bool(state.get("final") or snap["final"]),
                "resource": rel,
            }
        )
        doc = _public_json(ctx, rel)
        if doc is None:
            ctx.add("release.forecast_resource", f"resource:{rel}", expected="present", actual="absent")
            continue
        _diff(ctx, "release.forecast_value", rel, snap, doc)
        compared.add(rel)
    for rel in published:
        if rel not in compared:
            ctx.add(
                "release.forecast_resource",
                f"resource:{rel}",
                expected="absent",
                actual="present",
                message="no accepted live capture behind this resource",
            )
    _diff(
        ctx,
        "release.forecast_value",
        "live/index.json",
        {"delivery": dx.skip("delivery statement"), "matches": entries},
        index,
    )
    return compared


# ---------------------------------------------------------------------------
# release.content: articles and assets, provenance only
# ---------------------------------------------------------------------------

_IMPORTED = re.compile(r"from (\S+) \(sha256 ([0-9a-f]{64})\)")


def _slug(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(path).stem.lower()).strip("-")[:120]


def check_content(ctx: AuditContext) -> list[str]:
    cap = ctx.release
    assert cap is not None
    inv = {rel: (f.sha256, f.bytes) for rel, f in cap.public.items()}
    unknown: list[str] = []
    index = _public_json(ctx, "articles/index.json")
    articles = sorted(p for p in inv if p.startswith("articles/") and p != "articles/index.json")
    listed = [a for a in (index or {}).get("articles") or [] if isinstance(a, dict)] if isinstance(index, dict) else []
    by_resource: dict[Any, dict[str, Any]] = {a.get("resource"): a for a in listed}
    for rel in articles:
        if rel not in by_resource:
            ctx.add(
                "release.content_resource", f"resource:{rel}", expected="listed in articles/index.json", actual="absent"
            )
    for rel in sorted(set(by_resource) - set(articles), key=str):
        ctx.add("release.content_resource", f"resource:{rel}", expected="present", actual="absent")
    order = sorted(listed, key=lambda s: (-(_ordinal(s.get("published"))), str(s.get("slug"))))
    if [a.get("slug") for a in order] != [a.get("slug") for a in listed]:
        ctx.add("release.content_resource", "resource:articles/index.json", field="articles.order")
    manifest: dict[str, dict[str, Any]] | None = None
    if ctx.content_root is not None and ctx.content_manifest is not None:
        try:
            raw = tomllib.loads(ctx.external.read(ctx.content_manifest).decode("utf-8"))
        except ValueError as exc:
            raise CheckSkipped(Status.UNKNOWN, f"content manifest unavailable: {exc}") from exc
        manifest = {str(a["path"]): a for a in raw.get("article", [])}
    referenced: set[str] = set()
    curated = 0
    for rel in articles:
        doc = _public_json(ctx, rel)
        if not isinstance(doc, dict):
            ctx.add("release.content_resource", f"resource:{rel}", message="unreadable")
            continue
        ctx.count("articles")
        summary = doc.get("summary") or {}
        if by_resource.get(rel) is not None:
            _diff(ctx, "release.content_resource", "articles/index.json", summary, by_resource[rel])
        referenced |= {m for m in re.findall(r"assets/[A-Za-z0-9_./-]+", str(doc.get("html") or ""))}
        prov = str(doc.get("provenance") or "")
        if summary.get("editorial_state") == "generated":
            if ctx.snapshot is None or f"snapshot {ctx.snapshot.snapshot_id};" not in prov:
                ctx.add(
                    "release.content_provenance",
                    f"resource:{rel}",
                    field="provenance",
                    expected=f"checked against snapshot {ctx.snapshot.snapshot_id if ctx.snapshot else '?'}",
                    actual=prov[:300],
                )
            continue
        curated += 1
        m = _IMPORTED.search(prov)
        path = summary.get("original_path")
        if m is None or m.group(1) != path:
            ctx.add(
                "release.content_provenance",
                f"resource:{rel}",
                field="provenance",
                expected=f"names {path} and its sha256",
                actual=prov[:300],
            )
            continue
        if manifest is None:
            continue
        entry = manifest.get(str(path))
        source = ctx.content_root / str(path) if ctx.content_root is not None else None
        digest = ctx.external.digest(source) if source is not None else None
        want = {
            "listed": True,
            "sha256": m.group(2),
            "slug": _slug(str(path)),
            "category": entry.get("category") if entry else None,
            "scope": entry.get("scope") if entry else None,
        }
        have = {
            "listed": entry is not None,
            "sha256": digest,
            "slug": summary.get("slug"),
            "category": summary.get("category"),
            "scope": summary.get("scope"),
        }
        _diff(ctx, "release.content_provenance", rel, want, have)
    if curated and manifest is None:
        unknown.append(f"{curated} curated article(s): no content source supplied (--content-root/--content-manifest)")
    assets = sorted(p for p in inv if p.startswith("assets/"))
    for rel in assets:
        if rel not in referenced:
            ctx.add("release.content_resource", f"resource:{rel}", message="asset referenced by no article")
        elif ctx.content_root is not None:
            src = ctx.content_root / rel
            if ctx.external.digest(src) != inv[rel][0]:
                ctx.add(
                    "release.content_provenance",
                    f"resource:{rel}",
                    expected="the curated source bytes",
                    actual="different or missing source",
                )
    if assets and ctx.content_root is None:
        unknown.append(f"{len(assets)} asset(s): no content source supplied (--content-root)")
    if not unknown:
        _claim(ctx, [*assets, "articles/index.json"], "release.content")
        _claim(ctx, articles, "provenance")
    return unknown


def _ordinal(v: Any) -> int:
    try:
        return datetime.fromisoformat(str(v)).toordinal()
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# release.coverage
# ---------------------------------------------------------------------------


def check_coverage(ctx: AuditContext) -> list[str]:
    cap = ctx.release
    assert cap is not None
    claims: dict[str, str] = ctx.coverage.get("_claims", {})
    # a comparison counts only if it executed (PASS or FAIL), never when skipped or UNKNOWN
    ran = {c for c, st in ctx.coverage.get("_executed", []) if st in (Status.PASS.value, Status.FAIL.value)}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"resources": 0, "compared": 0, "provenance_only": 0})
    uncompared: dict[str, int] = defaultdict(int)
    for rel in cap.public:
        kind = _kind(rel)
        row = by_type[kind]
        row["resources"] += 1
        how = claims.get(rel)
        if how is None:
            _src, owner = AUTHORITY.get(kind, ("", ""))
            if owner in ("release.public", "release.artifact") and owner in ran:
                how = owner  # compared by the match/player comparison or bound by the artifact check
            elif owner == "provenance":
                how = "provenance"
        if kind == "other":
            ctx.add("release.unclassified_resource", f"resource:{rel}", actual=cap.public[rel].sha256)
        if how == "provenance":
            row["provenance_only"] += 1
        elif how is not None:
            row["compared"] += 1
        else:
            uncompared[kind] += 1
    unaudited = [_UNAUDITED[k] for k in sorted(_UNAUDITED) if by_type.get(k, {}).get("resources")]
    ctx.coverage["semantic"] = {
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "uncompared": dict(sorted(uncompared.items())),
        "unaudited": unaudited,
        "authority": {k: {"input": a, "check": c} for k, (a, c) in sorted(AUTHORITY.items())},
    }
    if uncompared:
        return [f"{n} {k} resource(s) have no executed semantic comparison" for k, n in sorted(uncompared.items())]
    return []


CHECKS = [
    CheckSpec(
        "release.derived",
        "release",
        "team, history, list, summary and download resources recomputed from the facts",
        check_derived,
        needs=("release", "snapshot"),
    ),
    CheckSpec(
        "release.forecast",
        "release",
        "prediction, accuracy and live resources against their pinned artifacts and captures",
        check_forecast,
        needs=("release", "snapshot"),
    ),
    CheckSpec(
        "release.content",
        "release",
        "article and asset provenance (prose is not audited)",
        check_content,
        needs=("release",),
    ),
    CheckSpec(
        "release.coverage",
        "release",
        "every published resource has an executed semantic comparison or is provenance-only by type",
        check_coverage,
        needs=("release",),
    ),
]
