#!/usr/bin/env python3
"""Offline, explicitly provisional operator tables; never promotes a snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from supercoach_via.analytics.players import player_stats_bundle
from supercoach_via.analytics.rankings import RankingConfig, run_legacy_v1
from supercoach_via.domain.metrics import CoverageEras
from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.settings import default_config_dir
from supercoach_via.storage.queries import SnapshotQuery
from supercoach_via.storage.snapshots import contained_path, load_snapshot, sha256_file

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs/hall-of-fame/provisional"
STATS = tuple(s for s in PLAYER_STAT_COLUMNS if s != "time_on_ground_pct")


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def read_audit(path, expected_hash, snapshot_id):
    if sha256_file(path) != expected_hash:
        raise ValueError("audit hash mismatch")
    report = json.loads(path.read_bytes())
    if report["snapshot_id"] != snapshot_id:
        raise ValueError("audit snapshot mismatch")
    return report


def output_path(root, relative):
    return contained_path(root, relative)


def leader_rows(frame, names, n=100):
    """Rank observed totals with competition ties and identity-based stable order."""
    import pandas as pd

    keys = ["total", "player_id"] + (["season"] if "season" in frame else [])
    frame = (
        frame[frame.total.notna()]
        .sort_values(keys, ascending=[False] + [True] * (len(keys) - 1), kind="stable")
        .head(n)
    )
    rows = []
    last, rank = None, 0
    for i, row in enumerate(frame.to_dict("records"), 1):
        if row["total"] != last:
            rank = i
        last = row["total"]
        clean = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        rows.append(dict(rank=rank, name=names.get(row["player_id"], row["player_id"]), **clean))
    return rows


def render_table(title, rows, banner):
    if not rows:
        return f"# PROVISIONAL — {title}\n\n{banner}\n\nNo recorded values.\n"
    wanted = (
        [
            "rank",
            "name",
            "player_id",
            "all_time_score",
            "recorded_appearances",
            "career_games",
            "source_counter_max",
            "best_season",
        ]
        if "all_time_score" in rows[0]
        else list(rows[0])
    )
    columns = [c for c in wanted if c in rows[0]]
    labels = {
        "player_id": "Player ID",
        "total": "Total",
        "observed_games": "Observed games",
        "mean": "Mean over observed games",
        "career_games": "Career games (max rows/counter)",
        "recorded_appearances": "Recorded appearances",
        "source_counter_max": "Source counter maximum",
        "all_time_score": "legacy_v1 score",
        "games": "Recorded season games",
    }

    def cell(value):
        if value is None:
            return "not recorded"
        if isinstance(value, (int, float)):
            return f"{value:g} **[data]**"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        f"# PROVISIONAL — {title}",
        "",
        banner,
        "",
        "| " + " | ".join(labels.get(c, c.replace("_", " ").title()) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(cell(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def generate(data_root, snapshot_id, audit_path, audit_hash, as_of):
    date.fromisoformat(as_of)
    audit = read_audit(audit_path, audit_hash, snapshot_id)
    manifest = load_snapshot(data_root, snapshot_id, verify=True)
    manifest_path = data_root / "snapshots" / (snapshot_id.removeprefix("sha256:") + ".json")
    config = RankingConfig.load()
    with SnapshotQuery(data_root, manifest) as q:
        names = dict(q.rows("SELECT player_id, display_name FROM players ORDER BY player_id"))
        bundle = player_stats_bundle(q, CoverageEras.load(default_config_dir() / "stat_coverage_eras.yaml"), STATS)
        ranking = run_legacy_v1(q, config)
    games = bundle.games.set_index("player_id")
    top = []
    for entry in ranking.all_time:
        row = asdict(entry)
        row["name"] = names.get(entry.player_id, entry.player_id)
        row["recorded_appearances"] = int(games.loc[entry.player_id, "row_games"])
        row["source_counter_max"] = (
            None
            if pd.isna(games.loc[entry.player_id, "counter_max"])
            else int(games.loc[entry.player_id, "counter_max"])
        )
        top.append(row)
    tables = {"top100": ("legacy_v1 top100 — score is a method output", top)}
    for stat in STATS:
        for scope, frame in [("career", bundle.careers), ("single-season", bundle.seasons)]:
            rows = leader_rows(
                frame[frame.stat == stat].drop(columns=["stat"]), names, n=20 if scope == "career" else 10
            )
            tables[f"{scope}/{stat}"] = (f"{scope} {stat} — recorded totals", rows)
    recorded = bundle.games.rename(columns={"row_games": "total"})
    tables["career/recorded-appearances"] = ("Recorded appearances", leader_rows(recorded, names, n=20))
    provenance = dict(
        status="PROVISIONAL",
        as_of=as_of,
        snapshot_id=snapshot_id,
        manifest_sha256=sha256_file(manifest_path),
        source_manifest=str(manifest_path.relative_to(REPO)),
        source_audit=str(audit_path.relative_to(REPO)),
        audit_sha256=audit_hash,
        audit_verdicts={k: v["verdict"] for k, v in audit["layers"].items()},
        ranking_config_sha256=config.config_hash(),
        stats=STATS,
    )
    manifest_local = manifest_path.relative_to(REPO)
    base_banner = (
        f"**PROVISIONAL · as of {as_of}.** Snapshot `{snapshot_id}`. "
        f"Audit snapshot **{provenance['audit_verdicts']['snapshot']}**, "
        f"legacy **{provenance['audit_verdicts']['legacy_csv']}**; SHA-256 `{audit_hash}`. "
        "These are data-based ranks, not official AFL Hall of Fame selections.\n\n"
        "**Historical Brownlow totals are incomplete, especially before 1984.** "
        "Known audit failures include missing appearances, source-proved zero values left null locally, "
        "aggregate mismatches and unresolved identities. "
        "Totals sum recorded values; all-missing totals are absent. "
        "Means divide by stat observed_games, and are observed-game means. "
        "Recorded appearances count rows; career_games uses max(rows, source counter). "
        "legacy_v1 alone retains its pinned historical blank-as-zero scoring imputation and era adjustments."
    )
    outputs = {"provenance.json": encode_json(provenance)}
    for path, (title, rows) in tables.items():
        depth = path.count("/")
        prefix = "../" * depth
        banner = (
            base_banner + f"\n\nSource: queried immutable snapshot manifest at `{manifest_local}`. "
            f"[Verified provenance]({prefix}provenance.json). Raw snapshot and audit inputs are retained locally; "
            f"rerun instructions are in [the report index]({prefix}README.md)."
        )
        outputs[path + ".md"] = render_table(title, rows, banner)
        outputs[path + ".json"] = encode_json(rows)
        buffer = io.StringIO(newline="")
        if rows:
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        outputs[path + ".csv"] = buffer.getvalue()
    command = (
        f".venv/bin/python scripts/refresh_provisional_hof.py --data-root {data_root.relative_to(REPO)} "
        f"--snapshot {snapshot_id} --audit-report {audit_path.relative_to(REPO)} "
        f"--audit-sha256 {audit_hash} --as-of {as_of}"
    )
    outputs["README.md"] = (
        "# PROVISIONAL — Hall of Fame operator tables\n\n"
        + base_banner
        + f"\n\nSource: queried immutable snapshot manifest at `{manifest_local}`. "
        + f"Audit report: `{audit_path.relative_to(REPO)}`. "
        + "Raw inputs are retained locally. [Verified provenance](provenance.json).\n\n"
        + f"Ranking method hash: `{config.config_hash()}`. "
        + "All additive current stats are included; time-on-ground percentage is not additive.\n\n"
        + "These separate operator reports do not promote the candidate. "
        + "Redo them after the audit corrections are verified.\n\n"
        + f"Repeat offline from repository root:\n\n```bash\n{command}\n```\n\n"
        + "Add `--check` to verify existing output bytes without writing. "
        + "All manifest fragments and the audit report hash are verified before querying.\n\n"
        + "\n".join(
            f"- [{title}]({path}.md) ([CSV]({path}.csv), [JSON]({path}.json))" for path, (title, _) in tables.items()
        )
        + "\n"
    )
    outputs["checksums.json"] = encode_json(
        {path: hashlib.sha256(content.encode()).hexdigest() for path, content in sorted(outputs.items())}
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data_root, audit_path = args.data_root.resolve(), args.audit_report.resolve()
    if (
        OUTPUT.resolve().is_relative_to(data_root)
        or data_root.is_relative_to(OUTPUT.resolve())
        or audit_path.is_relative_to(OUTPUT.resolve())
    ):
        raise ValueError("outputs must be separate from inputs")
    outputs = generate(data_root, args.snapshot, audit_path, args.audit_sha256, args.as_of)
    for relative, content in sorted(outputs.items()):
        target = output_path(OUTPUT, relative)
        if args.check:
            if not target.is_file() or target.read_bytes() != content.encode():
                raise ValueError(f"output differs: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode())
    print(f"{'Verified' if args.check else 'Generated'} {len(outputs)} provisional files for {args.snapshot}")


if __name__ == "__main__":
    main()
