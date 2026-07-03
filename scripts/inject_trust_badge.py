#!/usr/bin/env python3
"""inject_trust_badge.py — stamp the product trust badge onto a council doc.

The badge is the product's differentiator: a visible, honest verification line on
every published council doc.

  ✓ All N stats verified against source data · council-pipeline-gated · <date>

N is the number of file-backed [data] claims in the doc (the stats DataSentinel
checks against source CSVs). The line MUST carry the token `council-pipeline-gated`
so scripts/council-content-hash.sh strips it — injecting or refreshing the badge
therefore never changes the doc's canonical content hash, so it cannot invalidate
the DataSentinel audit record that backs the provenance stamp. Injection is
idempotent (an existing badge is replaced, never duplicated).

Usage:
  scripts/inject_trust_badge.py <doc.md> [<doc.md> ...] [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List

# Single source of truth for the tag vocabulary — the badge's N must never diverge
# from what DataSentinel/Skeptic recognise, so we delegate counting to the shared
# module rather than keeping a private regex here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tag_vocabulary import count_tags  # noqa: E402

MARK = "council-pipeline-gated"


def count_verified_stats(text: str) -> int:
    """Number of genuine source-data-verified [data] tags (all three forms).

    Delegates to tag_vocabulary.count_tags (kinds=data by default), which excludes
    [historical record]/[unverified], plain unbold [data], and stamp-line tokens.
    """
    return count_tags(text)


def make_badge(n: int, date: str) -> str:
    noun = "stat" if n == 1 else "stats"
    return f"> ✓ All {n} {noun} verified against source data · {MARK} · {date}"


def _strip_badge(lines: List[str]) -> List[str]:
    return [l for l in lines if MARK not in l]


def inject_badge(text: str, n: int, date: str) -> str:
    """Insert (or replace) the trust badge directly after the first H1. Idempotent."""
    trailing_nl = text.endswith("\n")
    lines = _strip_badge(text.splitlines())
    badge = make_badge(n, date)

    out: List[str] = []
    inserted = False
    for l in lines:
        out.append(l)
        if not inserted and l.startswith("# "):
            out.append(badge)
            inserted = True
    if not inserted:
        out = [badge] + lines

    result = "\n".join(out)
    return result + "\n" if trailing_nl else result


def inject_badge_or_strip(text: str, date: str) -> str:
    """Badge the doc with its true stat count; if it has zero tagged stats, add no
    badge (an "All 0 stats verified" line would be a false claim) and strip any
    stale badge that is there."""
    n = count_verified_stats(text)
    if n == 0:
        trailing_nl = text.endswith("\n")
        stripped = "\n".join(_strip_badge(text.splitlines()))
        return stripped + "\n" if trailing_nl else stripped
    return inject_badge(text, n, date)


def process_file(path: Path, date: str) -> int:
    text = path.read_text(encoding="utf-8")
    n = count_verified_stats(text)
    path.write_text(inject_badge_or_strip(text, date), encoding="utf-8")
    return n


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inject the council trust badge.")
    ap.add_argument("docs", nargs="+", help="markdown docs to badge")
    ap.add_argument("--date", default=_dt.date.today().isoformat(),
                    help="verification date (default: today)")
    args = ap.parse_args(argv)
    for d in args.docs:
        p = Path(d)
        if not p.is_file():
            print(f"skip (not a file): {d}")
            continue
        n = process_file(p, args.date)
        print(f"badged {d} (N={n})" if n else f"skipped {d} (no tagged stats — badge removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
