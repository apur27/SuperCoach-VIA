#!/usr/bin/env python3
"""Run a command and record wall time and peak process-tree RSS (sampled from /proc).

Usage: measure_process_tree.py OUT.json -- COMMAND [ARGS...]

Sums the resident set of the command and every descendant at each sample (default every 0.25 s), so
worker processes count together. Linux only. The command's exit status is preserved. Used by the
AFL Tables reconciliation audit to measure compare runs; it never changes what the command does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")


def children_map() -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for path in Path("/proc").iterdir():
        entry = path.name
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        kids.setdefault(ppid, []).append(int(entry))
    return kids


def tree_rss(root: int) -> tuple[int, int]:
    """(resident bytes, process count) of ``root`` and all descendants."""
    kids = children_map()
    stack, seen = [root], []
    while stack:
        pid = stack.pop()
        seen.append(pid)
        stack.extend(kids.get(pid, []))
    total = 0
    for pid in seen:
        try:
            total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * PAGE
        except OSError:
            continue
    return total, len(seen)


def main(argv: list[str]) -> int:
    if "--" not in argv or argv.index("--") != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[1])
    cmd = argv[3:]
    start = time.perf_counter()
    proc = subprocess.Popen(cmd)  # noqa: S603 - the operator supplies the command
    peak, peak_procs, samples = 0, 0, 0
    while proc.poll() is None:
        rss, n = tree_rss(proc.pid)
        samples += 1
        if rss > peak:
            peak, peak_procs = rss, n
        time.sleep(0.25)
    out.write_text(
        json.dumps(
            {
                "command": cmd,
                "exit_code": proc.returncode,
                "wall_seconds": round(time.perf_counter() - start, 3),
                "peak_tree_rss_mib": round(peak / 2**20, 1),
                "processes_at_peak": peak_procs,
                "samples": samples,
            },
            indent=1,
        )
        + "\n"
    )
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
