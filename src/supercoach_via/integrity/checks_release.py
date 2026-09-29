"""Family E (release side): artifact integrity and binding, and public resources vs canonical facts.

- ``release.artifact`` works on the captured inventory: closure against ``checksums.json``,
  the seal against the measured ``site/`` tree, the embedded ``site/data/<id>/`` copy, and
  that ``release.json`` and ``validation.json`` name this snapshot and these exact bytes.
  Every differing file is its own finding (exact counts).
- ``release.validate`` re-runs ``validate_release(write=False)`` for the rules it owns
  (allowlist, private content, JSON, schemas, manifest refs, references).
- ``release.public`` recomputes what every match index, match detail, player log, player
  index entry and player detail must say from the canonical rows and compares cell by cell.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS, Severity
from supercoach_via.integrity import public_compare as pc
from supercoach_via.integrity.capture import DriftError, strict_json
from supercoach_via.integrity.context import AuditContext, CheckSkipped, CheckSpec, rule
from supercoach_via.integrity.report import Status, canonical_bytes

B, E, W = Severity.BLOCKING, Severity.ERROR, Severity.WARNING
_DETAIL_GROUP = 600
_SEASON_PATH = re.compile(
    r"^(matches/\d{4}/index\.json|matches/detail/[^/]+\.json|player-games/[^/]+/\d{4}\.json|players/[^/]+\.json)$"
)

RULES = [
    rule(
        "release.metadata_invalid",
        "release.artifact",
        B,
        "checksums.json, seal.json or validation.json is missing or invalid",
        "rebuild or restore the release; never edit its metadata",
    ),
    rule(
        "release.tree_entry_refused",
        "release.artifact",
        B,
        "a symlink or non-regular file is inside public/ or site/",
        "remove it and rebuild; releases hold regular files only",
    ),
    rule(
        "release.public_extra",
        "release.artifact",
        B,
        "a public file is not listed in checksums.json",
        "rebuild the release; unlisted files are never published",
    ),
    rule(
        "release.public_missing",
        "release.artifact",
        B,
        "a file listed in checksums.json is missing",
        "restore the release from its archive",
    ),
    rule(
        "release.public_hash",
        "release.artifact",
        B,
        "a public file's bytes differ from checksums.json (truncated or substituted)",
        "restore the release from its archive",
    ),
    rule(
        "release.release_id",
        "release.artifact",
        B,
        "release.json, checksums.json or seal.json names another release",
        "restore the release directory from its archive",
    ),
    rule(
        "release.snapshot_mismatch",
        "release.artifact",
        B,
        "the release was built from a different snapshot than the one audited",
        "audit the release against its own snapshot, or rebuild the release from the audited snapshot",
    ),
    rule(
        "release.manifest_ref",
        "release.artifact",
        B,
        "a release.json resource reference disagrees with checksums.json",
        "rebuild the release",
    ),
    rule(
        "release.validation_outcome",
        "release.artifact",
        B,
        "validation.json does not record PASS",
        "run scvia validate-release and fix what it reports",
    ),
    rule(
        "release.validation_binding",
        "release.artifact",
        B,
        "validation.json names other checksums, another seal or another release than these bytes",
        "re-validate this exact release; a validation record for other bytes authorizes nothing",
    ),
    rule(
        "release.seal_mismatch",
        "release.artifact",
        B,
        "site/ differs from seal.json, or the seal's self-hash is wrong",
        "rebuild and re-seal the site; never copy over a sealed site",
    ),
    rule(
        "release.embedded_mismatch",
        "release.artifact",
        B,
        "site/data/<release>/ is not the validated public tree",
        "rebuild the site against this release's public tree and re-seal",
    ),
    rule(
        "release.unsafe_path",
        "release.validate",
        B,
        "a public path is outside the allowlist",
        "rename or drop the file and rebuild",
    ),
    rule(
        "release.private_content",
        "release.validate",
        B,
        "a published file contains a local path or secret marker",
        "remove the content at its source and rebuild",
    ),
    rule("release.json_invalid", "release.validate", B, "a public JSON file does not parse", "rebuild the release"),
    rule(
        "release.schema_invalid",
        "release.validate",
        B,
        "a public JSON file fails its view-model schema",
        "rebuild the release",
    ),
    rule(
        "release.reference_missing",
        "release.validate",
        B,
        "an index references a resource the release lacks",
        "rebuild the release",
    ),
    rule(
        "release.validate_other",
        "release.validate",
        B,
        "validate_release reported another failure",
        "run scvia validate-release",
    ),
    rule(
        "release.changed_during_audit",
        "release.public",
        B,
        "a public file changed between inventory and comparison",
        "re-run the audit on a stable copy",
    ),
    rule(
        "release.json_duplicate_key",
        "release.public",
        B,
        "a public JSON object repeats a key (consumers may read either value)",
        "rebuild the release; the serializer must never emit duplicate keys",
    ),
    rule(
        "release.resource_missing",
        "release.public",
        B,
        "a resource the canonical rows require is absent from the release",
        "rebuild the release from the audited snapshot",
    ),
    rule(
        "release.unexpected_resource",
        "release.public",
        B,
        "a match/player resource has no canonical row behind it",
        "rebuild the release; stale or foreign resources must not ship",
    ),
    rule(
        "release.index_membership",
        "release.public",
        B,
        "an index lists a match/player the facts lack, or omits one they hold",
        "rebuild the release from the audited snapshot",
    ),
    rule("release.index_order", "release.public", E, "an index is not in its contract order", "rebuild the release"),
    rule(
        "release.match_summary_mismatch",
        "release.public",
        B,
        "a published match fact differs from the canonical match row",
        "rebuild the release; shared match facts must equal the match row",
    ),
    rule(
        "release.source_label",
        "release.public",
        B,
        "a match's source label disagrees with its recorded provenance",
        "rebuild; provenance labels are claims to readers",
    ),
    rule(
        "release.stat_columns",
        "release.public",
        B,
        "stat_columns are not the canonical stats observed in the file, in canonical order "
        "(an all-null column must be omitted)",
        "rebuild the release",
    ),
    rule(
        "release.compact_row_width",
        "release.public",
        B,
        "a compact stats row is not as wide as its stat_columns",
        "rebuild the release; a short row shifts values onto the wrong statistic",
    ),
    rule(
        "release.cell_mismatch",
        "release.public",
        B,
        "a published statistic differs from the canonical cell (null stays null)",
        "rebuild the release; a value on the wrong player or stat surfaces here",
    ),
    rule(
        "release.box_membership",
        "release.public",
        B,
        "a match box score lists the wrong players or names",
        "rebuild the release",
    ),
    rule("release.box_order", "release.public", E, "a box score is not in contract order", "rebuild the release"),
    rule(
        "release.log_membership",
        "release.public",
        B,
        "a player season log lists the wrong games",
        "rebuild the release",
    ),
    rule(
        "release.log_order",
        "release.public",
        E,
        "a player season log is not in chronological order",
        "rebuild the release",
    ),
    rule(
        "release.log_value_mismatch",
        "release.public",
        B,
        "a player log field differs from the canonical row",
        "rebuild the release",
    ),
    rule(
        "release.shared_facts",
        "release.public",
        B,
        "a player log's shared match facts point at the wrong index, duplicate it, or name a match it lacks",
        "rebuild the release",
    ),
    rule(
        "release.player_index_value",
        "release.public",
        B,
        "a player index entry (seasons played, games, clubs, search) differs from the facts",
        "rebuild the release; season membership must be exact, not a first-last span",
    ),
    rule(
        "release.aggregate_mismatch",
        "release.public",
        B,
        "a player detail aggregate differs from totals recomputed from the facts (observed denominators)",
        "rebuild the release; a stale or mis-joined aggregate surfaces here",
    ),
]

_VALIDATE_MAP = {
    "allowlist": "release.unsafe_path",
    "private_content": "release.private_content",
    "json": "release.json_invalid",
    "schema": "release.schema_invalid",
    "manifest": "release.manifest_ref",
    "references": "release.reference_missing",
    "checksums": "release.metadata_invalid",
}
#: validate_release checks the artifact check covers file by file from the captured inventory
_OWN_CHECKS = frozenset({"closure", "hashes", "seal", "embedded_data"})


def _sha(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def check_artifact(ctx: AuditContext) -> list[str]:
    from supercoach_via.publish.release import SEAL_CHECKER
    from supercoach_via.publish.web_data import canonical_json_bytes

    cap = ctx.release
    assert cap is not None
    rid = cap.release_id
    for _attr, name, why in cap.meta_problems:
        ctx.add("release.metadata_invalid", f"file:{name}", message=why)
    for tree, rel, why in cap.tree_problems:
        ctx.add("release.tree_entry_refused", f"{tree}:{rel}", message=why)
    ctx.count("resources", len(cap.public) + len(cap.site))
    sums = (cap.checksums or {}).get("files")
    if cap.checksums is not None and (cap.checksums.get("release_id") != rid or not isinstance(sums, dict)):
        ctx.add("release.release_id", "file:checksums.json", expected=rid, actual=cap.checksums.get("release_id"))
    if isinstance(sums, dict):
        for rel in sorted(set(cap.public) - set(sums)):
            ctx.add("release.public_extra", f"resource:{rel}", actual=cap.public[rel].sha256)
        for rel in sorted(set(sums) - set(cap.public)):
            ctx.add("release.public_missing", f"resource:{rel}", expected=(sums[rel] or {}).get("sha256"))
        for rel in sorted(set(sums) & set(cap.public)):
            info, got = sums[rel] or {}, cap.public[rel]
            if info.get("sha256") != got.sha256 or info.get("bytes") != got.bytes:
                ctx.add(
                    "release.public_hash",
                    f"resource:{rel}",
                    expected={"sha256": info.get("sha256"), "bytes": info.get("bytes")},
                    actual={"sha256": got.sha256, "bytes": got.bytes},
                )
    # release.json names the release and the snapshot
    manifest: dict[str, Any] | None = None
    if "release.json" in cap.public:
        try:
            doc = strict_json(cap.read_public("release.json"))
            manifest = doc if isinstance(doc, dict) else None
        except (ValueError, DriftError) as exc:
            ctx.add("release.metadata_invalid", "resource:release.json", message=str(exc)[:300])
    if manifest is not None:
        if manifest.get("release_id") != rid:
            ctx.add("release.release_id", "resource:release.json", expected=rid, actual=manifest.get("release_id"))
        snap_id = ctx.snapshot.snapshot_id if ctx.snapshot is not None else None
        if snap_id is not None and manifest.get("snapshot_id") != snap_id:
            ctx.add(
                "release.snapshot_mismatch",
                f"release:{rid}",
                field="snapshot_id",
                expected=snap_id,
                actual=manifest.get("snapshot_id"),
            )
        for key, ref in sorted((manifest.get("resources") or {}).items()):
            info = (sums or {}).get(ref.get("path")) if isinstance(ref, dict) else None
            if (
                not isinstance(ref, dict)
                or info is None
                or info.get("sha256") != ref.get("sha256")
                or info.get("bytes") != ref.get("bytes")
            ):
                ctx.add(
                    "release.manifest_ref", f"resource:{ref.get('path') if isinstance(ref, dict) else key}", field=key
                )
    # validation.json must name these exact bytes
    val = cap.validation
    seal_hash = (cap.seal or {}).get("seal_sha256") if cap.has_site else None
    if val is not None:
        if val.get("outcome") != "PASS":
            ctx.add("release.validation_outcome", "file:validation.json", expected="PASS", actual=val.get("outcome"))
        if val.get("release_id") != rid:
            ctx.add(
                "release.validation_binding",
                "file:validation.json",
                field="release_id",
                expected=rid,
                actual=val.get("release_id"),
            )
        if val.get("checksums_sha256") != _sha(cap.checksums_raw):
            ctx.add(
                "release.validation_binding",
                "file:validation.json",
                field="checksums_sha256",
                expected=_sha(cap.checksums_raw),
                actual=val.get("checksums_sha256"),
            )
        if val.get("seal_sha256") != seal_hash:
            ctx.add(
                "release.validation_binding",
                "file:validation.json",
                field="seal_sha256",
                expected=seal_hash,
                actual=val.get("seal_sha256"),
            )
    # the seal covers exactly the measured site tree
    if cap.has_site:
        seal = cap.seal
        if seal is None:
            ctx.add("release.seal_mismatch", "file:seal.json", message="sealed site has no readable seal.json")
        else:
            body = {
                "release_id": seal.get("release_id"),
                "checker": seal.get("checker"),
                "build_inputs": seal.get("build_inputs") or {},
                "files": seal.get("files") or {},
                "seal_sha256": "",
            }
            if seal.get("seal_sha256") != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
                ctx.add(
                    "release.seal_mismatch", "file:seal.json", field="seal_sha256", message="seal self-hash mismatch"
                )
            if seal.get("release_id") != rid:
                ctx.add("release.release_id", "file:seal.json", expected=rid, actual=seal.get("release_id"))
            if seal.get("checker") != SEAL_CHECKER:
                ctx.add(
                    "release.seal_mismatch",
                    "file:seal.json",
                    field="checker",
                    expected=SEAL_CHECKER,
                    actual=seal.get("checker"),
                )
            declared = seal.get("files") or {}
            for rel in sorted(set(cap.site) - set(declared)):
                ctx.add("release.seal_mismatch", f"site:{rel}", expected="sealed", actual="unlisted")
            for rel in sorted(set(declared) - set(cap.site)):
                ctx.add("release.seal_mismatch", f"site:{rel}", expected="present", actual="missing")
            for rel in sorted(set(declared) & set(cap.site)):
                info, got = declared[rel] or {}, cap.site[rel]
                if info.get("sha256") != got.sha256 or info.get("bytes") != got.bytes:
                    ctx.add("release.seal_mismatch", f"site:{rel}", expected=info.get("sha256"), actual=got.sha256)
        prefix = f"data/{rid}/"
        embedded = {k[len(prefix) :]: v for k, v in cap.site.items() if k.startswith(prefix)}
        for rel in sorted(k for k in cap.site if k.startswith("data/") and not k.startswith(prefix)):
            ctx.add("release.embedded_mismatch", f"site:{rel}", expected="absent", actual="present")
        for rel in sorted(set(cap.public) | set(embedded)):
            a, b = cap.public.get(rel), embedded.get(rel)
            if a is None or b is None or a.sha256 != b.sha256:
                ctx.add(
                    "release.embedded_mismatch",
                    f"site:{prefix}{rel}",
                    expected=a.sha256 if a else None,
                    actual=b.sha256 if b else None,
                )
    return []


def check_validate(ctx: AuditContext) -> list[str]:
    from supercoach_via.publish.release import validate_release

    cap = ctx.release
    assert cap is not None
    key = None
    if ctx.cache is not None:
        from supercoach_via.integrity.cache import unit_key

        meta = hashlib.sha256(
            b"\0".join((cap.checksums_raw or b"-", cap.seal_raw or b"-", cap.validation_raw or b"-"))
        ).hexdigest()
        key = unit_key(ctx.cache.salt, "release.validate", meta + cap.identity()["inventory_sha256"], {})
        hit = ctx.cache.get(key)
        if hit is not None:
            ctx.collector.extend(hit.findings)
            for k, v in hit.examined.items():
                ctx.count(k, v)
            return []
    from supercoach_via.integrity.public_compare import UnitResult, _Out

    out = _Out(ctx.collector.rules, ctx.current_season, "release.validate")
    report = validate_release(cap.release_dir, write=False)
    out.count("resources", int(report.counts.get("files", 0)))
    for issue in report.issues:
        check = str(issue.get("check"))
        if check in _OWN_CHECKS:
            continue
        rid = _VALIDATE_MAP.get(check, "release.validate_other")
        path = str(issue.get("path"))
        out.add(
            rid,
            f"resource:{path}" if path != "site" else f"site:{issue.get('why')}"[:200],
            field=check,
            message=str(issue.get("why"))[:300],
        )
    result: UnitResult = out.done()
    if ctx.cache is not None and key is not None:
        ctx.cache.put(key, result)
    ctx.collector.extend(result.findings)
    for k, v in result.examined.items():
        ctx.count(k, v)
    return []


# ---------------------------------------------------------------------------
# release.public
# ---------------------------------------------------------------------------

_PG_COLUMNS = (
    "match_id",
    "player_id",
    "club_id",
    "season",
    "match_date",
    "date_quality",
    "result",
    "career_game_counter",
    *PLAYER_STAT_COLUMNS,
)


def check_public(ctx: AuditContext) -> list[str]:
    from supercoach_via.integrity.checks_data import _coverage
    from supercoach_via.publish.release import model_for_path

    cap = ctx.release
    assert cap is not None
    if ctx.snapshot is None or ctx.snapshot.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no verifiable snapshot to compare the release with")
    ctx.need("matches", "player_games", "players", "clubs")
    rules = ctx.collector.rules
    inventory = {rel: (f.sha256, f.bytes) for rel, f in cap.public.items()}
    club_names = {str(c): str(n) for c, n in ctx.rows("SELECT club_id, name FROM clubs ORDER BY 1")}
    names = {str(p): str(n) for p, n in ctx.rows("SELECT player_id, display_name FROM players ORDER BY 1")}
    seasons = [int(s) for (s,) in ctx.rows("SELECT DISTINCT season FROM matches ORDER BY 1")]
    con = ctx.db()
    frag = ctx.snapshot.fragment_digest
    units: list[tuple[str, Any, dict[str, Any], str, dict[str, tuple[str, int]]]] = []
    pg_sel = ", ".join(f'"{c}"' for c in _PG_COLUMNS)
    for season in seasons:
        matches = con.execute("SELECT * FROM matches WHERE season = ? ORDER BY match_id", [season]).to_arrow_table()
        games = con.execute(
            f"SELECT {pg_sel} FROM player_games WHERE season = ? ORDER BY match_id, player_id, club_id",  # noqa: S608 - column names are PLAYER_STAT_COLUMNS constants
            [season],
        ).to_arrow_table()
        # Arrow, not Python rows: all seasons' payloads exist at once before the units run
        match_ids = [str(x) for x in matches.column("match_id").to_pylist()]
        used = sorted({str(x) for x in games.column("player_id").to_pylist()})
        payload = {
            "season": season,
            "matches": matches,
            "player_games": games,
            "club_names": club_names,
            "player_names": {p: names.get(p, "") for p in used},
        }
        paths = [
            f"matches/{season}/index.json",
            *(f"matches/detail/{pc.public_key(m)}.json" for m in match_ids),
            *(f"player-games/{pc.public_key(p)}/{season}.json" for p in used),
        ]
        deps = hashlib.sha256(
            canonical_bytes(
                {
                    "matches": frag("matches", {str(season)}),
                    "player_games": frag("player_games", {str(season)}),
                    "names": [[p, names.get(p, "")] for p in used],
                    "clubs": sorted(club_names.items()),
                }
            )
        ).hexdigest()
        units.append((f"season:{season}", pc.season_unit, payload, deps, _subset(inventory, paths)))
    # players: aggregates recomputed from the facts
    cov = _coverage(ctx).coverage
    agg_sql = ", ".join(
        f'sum("{s}")::{"DOUBLE" if s == "time_on_ground_pct" else "BIGINT"} AS "{s}__t", count("{s}") AS "{s}__o", '
        + (f"count(*) FILTER (WHERE season >= {int(cov.recorded_from[s])})" if s in cov.recorded_from else "count(*)")
        + f' AS "{s}__e"'
        for s in PLAYER_STAT_COLUMNS
    )
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE agg_career AS SELECT player_id, count(*) AS rows_n, "  # noqa: S608
        f"max(career_game_counter) AS counter_max, min(season) AS first_season, max(season) AS last_season, {agg_sql} "
        "FROM player_games GROUP BY player_id"
    )
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE agg_season AS SELECT player_id, season, count(*) AS games, {agg_sql} "  # noqa: S608
        "FROM player_games GROUP BY player_id, season"
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE season_clubs AS "
        "SELECT player_id, season, list(club_id ORDER BY first_seen, club_id) "
        "AS clubs FROM (SELECT player_id, season, club_id, min(coalesce(match_date, DATE '9999-12-31')) AS first_seen "
        "FROM player_games GROUP BY ALL) GROUP BY ALL"
    )
    people = ctx.records(
        "SELECT player_id, display_name, identity_status, birth_date, birth_date_quality FROM players "
        "WHERE identity_status = 'canonical' ORDER BY player_id"
    )
    last_by_player = dict(ctx.rows("SELECT player_id, last_season FROM agg_career"))
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for p in people:
        groups[last_by_player.get(p["player_id"])].append(p)
    for last, members in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        for start in range(0, len(members), _DETAIL_GROUP):
            chunk = members[start : start + _DETAIL_GROUP]
            ids = [p["player_id"] for p in chunk]
            players = _detail_payload(ctx, chunk, ids)
            unit_id = f"players:{last}:{start // _DETAIL_GROUP}"
            deps = hashlib.sha256(canonical_bytes(_jsonable(players))).hexdigest()
            paths = [f"players/{pc.public_key(pid)}.json" for pid in ids]
            units.append(
                (unit_id, pc.detail_unit, {"players": players, "unit_id": unit_id}, deps, _subset(inventory, paths))
            )
    # player index
    latest = ctx.rows("SELECT max(season) FROM player_games")[0][0]
    idx_rows = ctx.records(
        """WITH s AS (SELECT player_id, list(DISTINCT season ORDER BY season) AS seasons FROM player_games GROUP BY 1),
                c AS (SELECT g.player_id, list(DISTINCT c.name ORDER BY c.name) AS club_names
                      FROM player_games g JOIN clubs c USING (club_id) GROUP BY 1)
           SELECT p.player_id, p.display_name, a.first_season, a.last_season, a.rows_n AS games, s.seasons, c.club_names
           FROM players p LEFT JOIN agg_career a USING (player_id) LEFT JOIN s USING (player_id)
           LEFT JOIN c USING (player_id)
           WHERE p.identity_status = 'canonical' ORDER BY p.display_name, p.player_id"""
    )
    index_players = [
        (
            r["player_id"],
            r["display_name"],
            {
                "first_season": r["first_season"],
                "last_season": r["last_season"],
                "games": r["games"] or 0,
                "seasons": r["seasons"] or [],
                "club_names": r["club_names"] or [],
            },
        )
        for r in idx_rows
    ]
    deps = hashlib.sha256(canonical_bytes(_jsonable([index_players, latest]))).hexdigest()
    units.append(
        (
            "players:index",
            pc.index_unit,
            {"players": index_players, "latest": latest},
            deps,
            _subset(inventory, ["players/index.json"]),
        )
    )
    results = _run(ctx, units, rules, cap.release_dir)
    expected: set[str] = set()
    counters: dict[str, int] = defaultdict(int)
    for r in results:
        ctx.collector.extend(r.findings)
        expected.update(r.expected_paths)
        for k, v in r.examined.items():
            counters[k] += v
            ctx.count(k, v)
    for rel in sorted(inventory):
        if _SEASON_PATH.match(rel) and rel not in expected and rel != "players/index.json":
            ctx.add("release.unexpected_resource", f"resource:{rel}", actual=inventory[rel][0])
    not_compared: dict[str, int] = defaultdict(int)
    for rel in inventory:
        if rel not in expected:
            not_compared[model_for_path(rel) or ("download" if rel.startswith("downloads/") else "other")] += 1
    ctx.coverage["public_compare"] = {
        "match_indexes": sum(1 for p in expected if p.startswith("matches/") and p.endswith("/index.json")),
        "match_details": counters.get("match_details", 0),
        "player_logs": counters.get("player_logs", 0),
        "player_details": counters.get("player_details", 0),
        "player_index_entries": counters.get("player_index_entries", 0),
        "resources_compared": counters.get("resources", 0),
        "not_compared_by_model": dict(sorted(not_compared.items())),
    }
    return []


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _subset(inventory: dict[str, tuple[str, int]], paths: list[str]) -> dict[str, tuple[str, int]]:
    return {p: inventory.get(p, ("absent", 0)) for p in paths}


def _detail_payload(ctx: AuditContext, chunk: list[dict[str, Any]], ids: list[str]) -> list[Any]:
    con = ctx.db()
    career = {
        r["player_id"]: r
        for r in con.execute("SELECT * FROM agg_career WHERE list_contains(?, player_id)", [ids])
        .to_arrow_table()
        .to_pylist()
    }
    per_season: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in (
        con.execute("SELECT * FROM agg_season WHERE list_contains(?, player_id)", [ids]).to_arrow_table().to_pylist()
    ):
        per_season[r["player_id"]][int(r["season"])] = r
    clubs: dict[tuple[str, int], list[str]] = {}
    for pid, season, cl in con.execute(
        "SELECT player_id, season, clubs FROM season_clubs WHERE list_contains(?, player_id)", [ids]
    ).fetchall():
        clubs[(pid, int(season))] = list(cl)

    def stats(r: dict[str, Any] | None) -> dict[str, tuple[Any, int, int]]:
        if r is None:
            return {}
        return {s: (r[f"{s}__t"], int(r[f"{s}__o"]), int(r[f"{s}__e"])) for s in PLAYER_STAT_COLUMNS}

    out = []
    for p in chunk:
        pid = p["player_id"]
        c = career.get(pid)
        rows = int(c["rows_n"]) if c else 0
        counter = c["counter_max"] if c else None
        career_d = {"career_games": max(rows, counter or 0), "counter_max": counter, "stats": stats(c)}
        seasons = {
            s: {"games": int(r["games"]), "clubs": clubs.get((pid, s), []), "stats": stats(r)}
            for s, r in sorted(per_season.get(pid, {}).items())
        }
        out.append((pid, p, career_d, seasons))
    return out


def _run(ctx: AuditContext, units: list[Any], rules: dict[str, Any], release_dir: Any) -> list[pc.UnitResult]:
    from supercoach_via.integrity.cache import unit_key

    results: dict[str, pc.UnitResult] = {}
    pending: list[tuple[str, Any, dict[str, Any], str | None]] = []
    for unit_id, fn, payload, deps, inv in units:
        key = unit_key(ctx.cache.salt, unit_id, deps, inv) if ctx.cache is not None else None
        if key is not None:
            hit = ctx.cache.get(key)
            if hit is not None:
                results[unit_id] = hit
                continue
        full = {
            **payload,
            "rules": rules,
            "current_season": ctx.current_season,
            "release_dir": release_dir,
            "inventory": {k: v for k, v in inv.items() if v[0] != "absent"},
        }
        pending.append((unit_id, fn, full, key))
    by_fn: dict[Any, list[tuple[str, dict[str, Any], str | None]]] = defaultdict(list)
    for unit_id, fn, full, key in pending:
        by_fn[fn].append((unit_id, full, key))
    for fn, items in by_fn.items():
        for (unit_id, _full, key), res in zip(items, pc.run_units(fn, [i[1] for i in items], ctx.workers), strict=True):
            results[unit_id] = res
            if ctx.cache is not None and key is not None:
                ctx.cache.put(key, res)
    return [results[u[0]] for u in units]


CHECKS = [
    CheckSpec(
        "release.artifact",
        "release",
        "closure, seal, embedded data and binding to the snapshot and validation record",
        check_artifact,
        needs=("release",),
    ),
    CheckSpec(
        "release.validate",
        "release",
        "validate_release (allowlist, private content, schemas, references)",
        check_validate,
        needs=("release",),
    ),
    CheckSpec(
        "release.public",
        "release",
        "every match, log and player resource recomputed from the facts",
        check_public,
        needs=("release", "snapshot"),
    ),
]
