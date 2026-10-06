"""``propose-corrections``: source-backed changes for both local layers from a completed audit (offline).

Binds to the report it reads: the report's plan, snapshot and capture must be the ones this run loads, so a
proposal can never be computed against inputs other than those the findings describe. Writes ``changes.jsonl``
(``corrections.Change`` per line, stable order) and ``summary.json``; never edits an input.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from supercoach_via.reconciliation import compare as CP
from supercoach_via.reconciliation import corrections as CO
from supercoach_via.reconciliation import idfix as IF

_DERIVED = frozenset({"AGGREGATE_MISMATCH"})


class ProposeError(RuntimeError):
    """The report does not describe the inputs this run loaded, or it is incomplete (exit 2)."""


def _snapshot_current(audit: CP.Audit) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if audit.snap is None:
        return out
    for r in audit.snap._rows("players"):
        bd = r.get("birth_date")
        out[r["player_id"]] = {
            "display_name": r.get("display_name"),
            "first_name": r.get("first_name"),
            "last_name": r.get("last_name"),
            "birth_date": bd.isoformat() if bd is not None else None,
            "source_urls": r.get("source_urls"),
            "identity_status": r.get("identity_status"),
            "canonical_player_id": r.get("canonical_player_id"),
        }
    return out


def _legacy_current(root: Path, slugs: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slug in sorted(slugs):
        path = root / "data" / "player_data" / f"{slug}_personal_details.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) == 1:
            out[slug] = {k: (rows[0].get(k) or "") for k in ("first_name", "last_name", "born_date")}
    return out


def propose(opts: CP.CompareOptions, report_dir: Path, out_dir: Path) -> dict[str, Any]:
    report = json.loads((report_dir / "report.json").read_bytes())
    if not (report_dir / "output-manifest.json").is_file():
        raise ProposeError(f"{report_dir} is not a completed report (no output-manifest.json)")
    audit = CP.Audit(opts)
    audit.keep_identity = True  # type: ignore[attr-defined]
    try:
        audit.load()
        if report["plan_id"] != audit.plan.plan_id or report["snapshot_id"] != audit.plan.inputs.snapshot.snapshot_id:
            raise ProposeError("the report was produced under a different plan or snapshot than this run loads")
        manifest_sha = CP.hashlib_sha(opts.capture_manifest.read_bytes())
        if report["capture_manifest_sha256"] != manifest_sha:
            raise ProposeError("the report was produced from a different capture manifest")
        audit.parse()
        audit.inventory()
        audit.identity_snapshot()
        audit.identity_legacy()
        audit._profile_index()  # profile URL -> (body sha, name), for rebuilding rows from the captured pages
        with (report_dir / "findings.jsonl").open(encoding="utf-8") as fh:
            findings = [
                json.loads(line) for line in fh if '"severity":"fail"' in line or "LOCAL_UNSUPPORTED_NUMERIC" in line
            ]
        changes, unsupported = CO.changes_from_findings(findings)
        changes = _keep_invented_zeros_only(audit, changes, unsupported)
        id_counts: Counter[str] = Counter()
        id_urls: set[str] = set()
        for layer in ("snapshot", "legacy_csv"):
            st = audit.layers.get(layer)
            if st is None or st.identity is None or st.identity_inputs is None:
                continue
            profiles, recs = st.identity_inputs
            props = IF.identity_proposals(profiles, recs, st.identity)
            id_counts.update(f"{layer}:{p.kind}" for p in props)
            id_urls |= {p.url for p in props if p.kind in ("repair", "duplicate")}
            if layer == "snapshot":
                current = _snapshot_current(audit)
            else:
                assert audit.legacy_root is not None
                current = _legacy_current(audit.legacy_root, {p.key.removeprefix("legacy:") for p in props})
            changes.extend(CO.identity_changes(props, layer=layer, current=current))
        rows, row_unsupported = row_changes(audit, findings, skip_profiles=id_urls)
        changes.extend(rows)
        for k, n in row_unsupported.items():
            unsupported[k] = n
        for k in ("APPEARANCE_QUARANTINED", "APPEARANCE_MISSING_LOCAL", "MATCH_MISSING_LOCAL", "MATCH_EXTRA_LOCAL",
                  "PLAYER_MISSING_LOCAL"):  # fmt: skip
            if k not in row_unsupported:
                unsupported.pop(k, None)
    finally:
        CP.cleanup(audit)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = CO.write_changes(out_dir / "changes.jsonl", changes)
    by_kind = Counter(f"{c.layer}:{c.rule_id}" for c in changes)
    summary = {
        "kind": "afltables-reconciliation-corrections",
        "report_sha256": CP.hashlib_sha((report_dir / "report.json").read_bytes()),
        "plan_id": report["plan_id"],
        "snapshot_id": report["snapshot_id"],
        "capture_manifest_sha256": report["capture_manifest_sha256"],
        "changes": len(changes),
        "changes_sha256": digest,
        "by_layer_rule": dict(sorted(by_kind.items())),
        "identity_proposals": dict(sorted(id_counts.items())),
        "unsupported_fail_findings": dict(sorted((k, v) for k, v in unsupported.items() if k not in _DERIVED)),
        # totals that differ only because rows or cells differ: they are re-judged, never edited, after the rest
        "derived_fail_findings": dict(sorted((k, v) for k, v in unsupported.items() if k in _DERIVED)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _keep_invented_zeros_only(audit: CP.Audit, changes: list[CO.Change], unsupported: Counter[str]) -> list[CO.Change]:
    """A cell the source proves unrecorded is nulled only when the snapshot stores 0 there (an invented zero); a
    non-zero stored value is a real disagreement and is reported, never erased."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    wanted = [c for c in changes if c.layer == "snapshot" and c.old == 0 and c.new is None]
    if not wanted or audit.snap is None:
        return changes
    frags = {f.ref.sha256[:16]: f for f in audit.snap.cap.fragments if f.table == "player_games"}
    tables: dict[str, Any] = {}
    drop: set[int] = set()
    for c in wanted:
        sha16, _, row = c.target.partition("#")
        f = frags.get(sha16)
        if f is None or f.data is None:
            drop.add(id(c))
            continue
        if sha16 not in tables:
            tables[sha16] = pq.read_table(pa.BufferReader(f.data))
        if tables[sha16].column(c.field)[int(row)].as_py() != 0:
            drop.add(id(c))
    if drop:
        unsupported["LOCAL_UNSUPPORTED_NUMERIC:nonzero"] += len(drop)
    return [c for c in changes if id(c) not in drop]


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def row_changes(
    audit: CP.Audit, findings: list[dict[str, Any]], *, skip_profiles: set[str]
) -> tuple[list[CO.Change], Counter[str]]:
    """Row-level corrections: a quarantined snapshot row relinked to the match the source proves, a missing legacy
    player-game row (or a whole missing legacy player), and a missing or corrupted legacy match row. A profile an
    identity proposal repairs is skipped: its appearances are matched once the identity is corrected."""
    from supercoach_via.reconciliation import facts as FX
    from supercoach_via.reconciliation import legacy_rows as LR
    from supercoach_via.reconciliation.cache import FactsCache

    cache = FactsCache(audit.cache_root, audit.phash)
    out: list[CO.Change] = []
    left: Counter[str] = Counter()
    snap = audit.layers.get("snapshot")
    to_local = snap.match_result.mapping.source_to_local if snap and snap.match_result else {}
    legacy_root = audit.legacy_root

    def match_facts(url: str) -> Any:
        res = audit.res_by_url.get(url)
        return FX.load_match(cache, res.sha256) if res and res.sha256 else None

    new_players: dict[str, dict[str, Any]] = {}
    extras: dict[tuple[int, str], dict[str, Any]] = {}
    for f in findings:
        if f["layer"] == "legacy_csv" and f["category"] == "MATCH_EXTRA_LOCAL":
            origin = f["local"].get("origin", "")
            rel, _, n = origin.rpartition("#")
            if legacy_root is not None and n.isdigit():
                rows = _csv_rows(legacy_root / rel)
                hdr = rows[0]
                row = rows[int(n)]
                extras[(f["season"], row[hdr.index("date")][:10])] = {"origin": origin, "row": row, "id": f["id"]}
    for f in findings:
        cat, layer = f["category"], f["layer"]
        ev = f.get("evidence") or {}
        url, sha = ev.get("source_url"), ev.get("body_sha256")
        purl = (f.get("player") or {}).get("source_url")
        if cat == "APPEARANCE_QUARANTINED" and layer == "snapshot":
            target = to_local.get(f["match"].get("source_url") or "")
            qid = (f.get("local") or {}).get("origin", "")
            if target and qid.startswith("q:"):
                out.append(CO.Change(layer, f"quarantine:{qid}", "relink", None, json.dumps({"match_id": target}),
                                     f["rule_id"], f["id"], url, sha))  # fmt: skip
            else:
                left[cat] += 1
        elif cat == "APPEARANCE_MISSING_LOCAL" and layer == "legacy_csv" and legacy_root is not None:
            if purl in skip_profiles:
                continue
            murl = f["match"].get("source_url") or ""
            info = audit.profile_info.get(purl or "")
            games = FX.load_profile_games(cache, info[0], f["season"]) if info else None
            g = next((x for x in games or [] if x.match_url == murl), None)
            m = match_facts(murl)
            if g is None or m is None:
                left[cat] += 1
                continue
            slug = (f["player"].get("local_id") or "").split(",")[0]
            if slug:
                rel = f"data/player_data/{slug}_performance_details.csv"
                hdr = _csv_rows(legacy_root / rel)[0]
                row = LR.player_row(hdr, g, m.match_date)
                out.append(CO.Change(layer, f"{rel}#insert", "row", None, json.dumps(row), f["rule_id"], f["id"],
                                     url, sha))  # fmt: skip
            else:
                core = FX.load_core(cache, info[0]) if info else None
                if core is None or not core.born or not core.h1:
                    left[cat] += 1
                    continue
                first, _, last = core.h1.partition(" ")
                y, mo, d = core.born.split("-")
                slug = f"{last.lower().replace(' ', '_').replace(chr(39), '')}_{first.lower()}_{d}{mo}{y}"
                spec = new_players.setdefault(slug, {"personal": {"first_name": first, "last_name": last,
                                              "born_date": f"{d}-{mo}-{y}"}, "rows": [], "url": purl})  # fmt: skip
                spec["rows"].append(LR.player_row(LR.PERF_HEADER, g, m.match_date))
        elif cat == "MATCH_MISSING_LOCAL" and layer == "legacy_csv" and legacy_root is not None:
            murl = f["match"].get("source_url") or ""
            m = match_facts(murl)
            if m is None or not m.match_date:
                left[cat] += 1
                continue
            rel = f"data/matches/matches_{f['season']}.csv"
            hdr = _csv_rows(legacy_root / rel)[0]
            row = LR.match_row(hdr, m)
            extra = extras.pop((f["season"], m.match_date), None)
            if extra is not None:  # the local row for this match exists but holds corrupted values
                out.append(CO.Change(layer, extra["origin"], "row", json.dumps(extra["row"]), json.dumps(row),
                                     f["rule_id"], f["id"], murl, ev.get("body_sha256")))  # fmt: skip
            else:
                out.append(CO.Change(layer, f"{rel}#insert", "row", None, json.dumps(row), f["rule_id"], f["id"],
                                     murl, ev.get("body_sha256")))  # fmt: skip
        elif cat == "PLAYER_MISSING_LOCAL" and purl in skip_profiles:
            continue
    for slug, spec in sorted(new_players.items()):

        def counter(r: list[str]) -> int:
            return int(r[2]) if r[2].isdigit() else 0

        spec["rows"].sort(key=counter)
        if spec["rows"]:
            dates = [r[-1] for r in spec["rows"] if r[-1]]
            if dates:
                y, mo, d = min(dates).split("-")
                spec["personal"]["debut_date"] = f"{d}-{mo}-{y}"
        body = {"personal": spec["personal"], "rows": spec["rows"]}
        out.append(CO.Change("legacy_csv", f"data/player_data/{slug}", "create_player", None, json.dumps(body),
                             "R-PLAYER-MISSING", "", spec["url"], None))  # fmt: skip
    if extras:
        left["MATCH_EXTRA_LOCAL"] = len(extras)  # a local match row the source does not show, with no missing pair
    return out, left
