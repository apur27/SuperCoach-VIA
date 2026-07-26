#!/usr/bin/env python3
"""Completion accounting for backtest artifacts.

WHY THIS EXISTS
---------------
On 2026-07-20 the weekly cycle wrote backtest artifacts at 15:57, then FATALed at
16:04 when the phantom-row gate caught a mass-duplicated player corpus. Those
artifacts stayed on disk. The retry run selected the newest
`backtest_summary_*.csv`, concluded "last complete backtest: round 20", and
skipped re-scoring. Figures computed on a corpus known to be bad were committed
and published.

The old detector asked "does an artifact for round N exist?" That question cannot
distinguish a finished cycle from an aborted one. This module asks the correct
question instead: "did a cycle explicitly declare this run complete?"

Absence of a mark means INCOMPLETE. That is the fail-closed direction: the worst
case is re-scoring a round we already had, which is cheap on the --from-csv path;
the alternative is publishing numbers from a corpus we rejected.

Quarantining moves orphans OUT of the directory, which matters because two
independent consumers glob it and both were fooled by the same file:
  - refresh_and_rank.sh          picks the newest summary by FILENAME
  - update_team_analysis.py /
    update_eval_surface.sh       pick the newest summary by MTIME
Neither can select a file that is no longer there, so one mechanic closes both
paths without touching either consumer's logic.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

MANIFEST_NAME = "completed_runs.json"
QUARANTINE_DIR = "quarantine"

# A single backtest invocation stamps one timestamp across this whole family.
ARTIFACT_PREFIXES = (
    "backtest_summary_",
    "backtest_by_team_",
    "backtest_by_position_",
    "backtest_run_",
    "prediction_vs_actual_round_",
)

_TS_RE = re.compile(r"(\d{8}_\d{6})")
_ROUND_RE = re.compile(r"prediction_vs_actual_round_(\d+)_(\d{4})_(\d{8}_\d{6})\.csv$")


def _artifact_files(bt_dir: Path):
    for p in Path(bt_dir).iterdir():
        if p.is_file() and p.name.startswith(ARTIFACT_PREFIXES):
            yield p


def timestamps_on_disk(bt_dir: Path) -> set[str]:
    """Every run timestamp with at least one artifact present."""
    found = set()
    for p in _artifact_files(bt_dir):
        m = _TS_RE.search(p.name)
        if m:
            found.add(m.group(1))
    return found


def summary_backed_timestamps(bt_dir: Path) -> set[str]:
    """Timestamps that produced a `backtest_summary_*.csv`.

    A run that wrote no summary scored no round, so this is the only on-disk
    evidence of completion available when bootstrapping a directory that predates
    the manifest.
    """
    found = set()
    for p in Path(bt_dir).glob("backtest_summary_*.csv"):
        m = _TS_RE.search(p.name)
        if m:
            found.add(m.group(1))
    return found


def completed_timestamps(bt_dir: Path) -> set[str]:
    """Timestamps a cycle marked complete.

    A missing or unreadable manifest yields the empty set — never a blanket
    approval. Fail closed.
    """
    manifest = Path(bt_dir) / MANIFEST_NAME
    try:
        payload = json.loads(manifest.read_text())
        return set(payload["completed"])
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def incomplete_timestamps(bt_dir: Path) -> set[str]:
    return timestamps_on_disk(bt_dir) - completed_timestamps(bt_dir)


def mark_complete(bt_dir: Path, timestamps) -> set[str]:
    """Record timestamps as belonging to a cycle that ran to completion.

    Additive and idempotent: call it after a cycle's final push succeeds.
    """
    bt_dir = Path(bt_dir)
    bt_dir.mkdir(parents=True, exist_ok=True)
    merged = completed_timestamps(bt_dir) | set(timestamps)
    (bt_dir / MANIFEST_NAME).write_text(
        json.dumps({"completed": sorted(merged)}, indent=2) + "\n"
    )
    return merged


def last_complete_round(bt_dir: Path, year: int):
    """Highest round scored by a COMPLETE run for `year`, or None.

    Rounds scored only by unmarked (aborted-cycle) runs are invisible here, so
    the caller re-scores them against the current corpus.
    """
    complete = completed_timestamps(bt_dir)
    rounds = []
    for p in Path(bt_dir).glob("prediction_vs_actual_round_*.csv"):
        m = _ROUND_RE.search(p.name)
        if m and int(m.group(2)) == int(year) and m.group(3) in complete:
            rounds.append(int(m.group(1)))
    return max(rounds) if rounds else None


def quarantine_incomplete(bt_dir: Path) -> list[str]:
    """Move every artifact of every unmarked run into `quarantine/`.

    Returns the moved file names. Nothing is deleted — the artifacts remain
    available for forensics, they just stop being selectable by the pipeline.
    """
    bt_dir = Path(bt_dir)

    # Bootstrap: no manifest at all means this directory was never initialised
    # (e.g. a fresh clone of the committed backtest history), NOT that every run
    # is an orphan. Quarantining on that reading would move the entire committed
    # history aside and reset the pipeline to round 1.
    #
    # Adopt only SUMMARY-BACKED runs, though. The first version adopted every
    # timestamp on disk, which grandfathered in two orphan round-1 vintages from
    # aborted runs — and since their mtimes were newer than the authoritative
    # file, a consumer resolving by latest timestamp was publishing the orphan's
    # figures. The manifest ended up endorsing precisely what it exists to catch.
    # Nothing is moved on the bootstrap pass itself: a fresh clone must not have
    # tracked files pulled out from under it. The next sweep handles the orphans.
    if not (bt_dir / MANIFEST_NAME).exists():
        mark_complete(bt_dir, summary_backed_timestamps(bt_dir))
        return []

    orphans = incomplete_timestamps(bt_dir)
    if not orphans:
        return []
    qdir = bt_dir / QUARANTINE_DIR
    qdir.mkdir(exist_ok=True)
    moved = []
    for p in sorted(_artifact_files(bt_dir)):
        m = _TS_RE.search(p.name)
        if m and m.group(1) in orphans:
            dest = qdir / p.name
            if dest.exists():
                dest.unlink()  # a prior sweep already captured this name
            shutil.move(str(p), str(dest))
            moved.append(p.name)
    return moved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/prediction/backtest",
                    help="backtest artifact directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_last = sub.add_parser("last-round",
                            help="print the last COMPLETE round (empty if none)")
    p_last.add_argument("--year", type=int, required=True)

    sub.add_parser("mark", help="mark every on-disk run complete (call after a successful cycle)")
    sub.add_parser("sweep", help="quarantine artifacts of runs never marked complete")
    sub.add_parser("status", help="report complete vs orphaned runs")

    args = ap.parse_args(argv)
    bt_dir = Path(args.dir)
    if not bt_dir.exists():
        print(f"backtest_completeness: no such directory {bt_dir}", file=sys.stderr)
        return 0 if args.cmd == "last-round" else 1

    if args.cmd == "last-round":
        r = last_complete_round(bt_dir, args.year)
        print("" if r is None else r)
        return 0

    if args.cmd == "mark":
        merged = mark_complete(bt_dir, timestamps_on_disk(bt_dir))
        print(f"backtest_completeness: {len(merged)} run(s) marked complete")
        return 0

    if args.cmd == "sweep":
        moved = quarantine_incomplete(bt_dir)
        if moved:
            print(f"backtest_completeness: quarantined {len(moved)} orphaned artifact(s) "
                  f"from an incomplete cycle -> {bt_dir / QUARANTINE_DIR}")
            for name in moved:
                print(f"  {name}")
        else:
            print("backtest_completeness: no orphaned artifacts")
        return 0

    if args.cmd == "status":
        complete = completed_timestamps(bt_dir)
        orphans = incomplete_timestamps(bt_dir)
        print(f"complete runs : {len(complete)}")
        print(f"orphaned runs : {len(orphans)}")
        for ts in sorted(orphans):
            print(f"  ORPHAN {ts}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
