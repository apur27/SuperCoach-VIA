#!/usr/bin/env python3
"""skeptic_sample_tags.py — deterministic tag sampler for the Skeptic gate.

Skeptic spot-probes a fixed-size subset of a draft's verifiable tags (it does not
re-verify every tag — that is DataSentinel's job). Which tags it probes must be a
DETERMINISTIC function of the document path, not an ad-hoc LLM choice, so the same
document always probes the same tags and any prompt change is reproducibly gated.

Selection = a hash-based sample WITHOUT replacement, seeded on the doc path:
  rank each tag ordinal i (1..N) by sha256(f"{doc_path}:{i}"), take the n
  lowest-ranked, return them sorted ascending. This is stable, distinct, and
  in-range by construction.

Usage:
  scripts/skeptic_sample_tags.py <doc.md> [--n 3]
  -> prints the selected tags as "index<TAB>line<TAB>tag<TAB>text", one per line.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Dict

# Single source of truth for the tag vocabulary (all three [data] forms + historical
# record). Importing it — rather than re-declaring a private regex — closes the F2
# failure class where the old bare-only matcher missed **[data: spec]** tags and the
# sampler silently probed nothing on real briefs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tag_vocabulary import extract_tags  # noqa: E402


def select_tag_indices(doc_path: str, num_tags: int, n: int = 3) -> List[int]:
    """Deterministically pick up to n distinct 1-based tag ordinals for doc_path."""
    if num_tags <= 0:
        return []
    if num_tags <= n:
        return list(range(1, num_tags + 1))

    def rank(i: int) -> str:
        return hashlib.sha256(f"{doc_path}:{i}".encode()).hexdigest()

    ordinals = sorted(range(1, num_tags + 1), key=rank)[:n]
    return sorted(ordinals)


def sample(doc_path: str, n: int = 3) -> List[Dict]:
    """Read the doc, extract tags, return the deterministic n-tag sample."""
    with open(doc_path, encoding="utf-8") as fh:
        tags = extract_tags(fh.read())
    if not tags:
        return []
    idx = select_tag_indices(doc_path, len(tags), n=n)
    chosen = [{"index": i, **tags[i - 1]} for i in idx]
    if not chosen:
        # A doc that HAS verifiable tags must never yield an empty sample — that
        # would mean Skeptic's smoke test silently checks nothing (the F2 failure
        # class). Fail loudly instead of passing a vacuous gate.
        raise ValueError(
            f"{doc_path}: {len(tags)} verifiable tag(s) present but the sample is "
            "empty — tag vocabulary or selection is broken; refusing a vacuous check."
        )
    return chosen


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic Skeptic tag sampler.")
    ap.add_argument("doc", help="path to the draft markdown document")
    ap.add_argument("--n", type=int, default=3, help="sample size (default 3)")
    args = ap.parse_args(argv)

    chosen = sample(args.doc, n=args.n)
    if not chosen:
        print(f"# no verifiable [data]/[historical record] tags in {args.doc}", file=sys.stderr)
        return 0
    for s in chosen:
        print(f"{s['index']}\t{s['line']}\t{s['tag']}\t{s['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
