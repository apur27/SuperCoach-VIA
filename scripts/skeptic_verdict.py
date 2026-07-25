#!/usr/bin/env python3
"""Interpret a Skeptic verdict from agent output. Exit 0 to ship, non-zero to halt.

docs/afl-insights.md used to carry a "weekly recap exemption" from Skeptic: it was
gated by DataSentinel only. DataSentinel answers "is this number true?" — it does
not answer "does the prose around this number claim more than the number
supports?" The user has required the adversarial pass, so this reads Skeptic's
three-way verdict deterministically.

  PASS               -> 0   ship
  PASS_WITH_CONCERNS -> 0   ship; concerns printed for the retro
  BLOCK              -> 1   halt

Anything else — no verdict, empty file, prose instead of JSON, a crashed agent, a
mistyped token — is ALSO non-zero. A gate that could not run has not passed; that
distinction is exactly what let a FATALed cycle's artifacts ship earlier this
month. Matching is exact-token and case-sensitive, consistent with
record-sentinel-verdict.sh (F07), and BLOCK dominates so an incidental "PASS"
inside a reason string cannot clear the gate.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CLEARING = ("PASS", "PASS_WITH_CONCERNS")
HALTING = ("BLOCK",)
KNOWN = CLEARING + HALTING

_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"([A-Z_]+)"')


def _extract_verdict(text: str):
    """Prefer real JSON; fall back to a regex for agents that wrap JSON in prose."""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("verdict"), str):
            return payload["verdict"], payload
    except ValueError:
        pass
    found = _VERDICT_RE.findall(text)
    if not found:
        return None, {}
    # BLOCK dominates if several verdict tokens appear.
    for v in found:
        if v in HALTING:
            return v, {}
    return found[0], {}


def _reason(text: str, payload: dict) -> str:
    for key in ("reason", "concerns", "notes"):
        if payload.get(key):
            val = payload[key]
            return "; ".join(val) if isinstance(val, list) else str(val)
    m = re.search(r'"(?:reason|concerns)"\s*:\s*(\[[^\]]*\]|"[^"]*")', text)
    return m.group(1) if m else ""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("skeptic_verdict: usage: skeptic_verdict.py <agent-output-file>",
              file=sys.stderr)
        return 2

    path = Path(argv[0])
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        print(f"skeptic_verdict: HALT — cannot read Skeptic output ({exc}). "
              f"A gate that cannot run has not passed.", file=sys.stderr)
        return 2

    verdict, payload = _extract_verdict(text)
    reason = _reason(text, payload)

    if verdict is None:
        print("skeptic_verdict: HALT — no verdict found in Skeptic output "
              "(agent crashed, timed out, or emitted prose). Fail-closed.",
              file=sys.stderr)
        return 3

    if verdict in HALTING:
        print(f"skeptic_verdict: BLOCK — {reason or 'no reason given'}",
              file=sys.stderr)
        return 1

    if verdict in CLEARING:
        if verdict == "PASS_WITH_CONCERNS":
            print(f"skeptic_verdict: {verdict} — proceeding; log in retro: "
                  f"{reason or 'no detail given'}")
        else:
            print(f"skeptic_verdict: {verdict}")
        return 0

    print(f"skeptic_verdict: HALT — unrecognised verdict {verdict!r} "
          f"(expected one of {', '.join(KNOWN)}). Fail-closed.", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
