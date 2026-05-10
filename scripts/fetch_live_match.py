#!/usr/bin/env python
"""
Fetch and parse a FanFooty live match snapshot.

Usage:
    fetch_live_match.py <gameid>

Pulls https://www.fanfooty.com.au/live/<gameid>.txt, parses the structured
text feed, and writes two artifacts under data/live_snapshots/:

  - <gameid>_<YYYYMMDD_HHMM>_<status>.json   (full structured snapshot)
  - <gameid>_<YYYYMMDD_HHMM>_players.csv     (player table only)

The .txt format is:

  line 1: match header (home, away, round, score, status)
  line 2: match meta   (date, year, time, venue, ...)
  line 3: m0nty commentary stream + injury notes (HTML inline)
  line 4: coach chat stream (HTML inline)
  line 5+: one player row per line, 65 comma-separated columns

Column schema for player rows is decoded by inspection of historical feeds.
Fields not confidently identified are kept under their numeric position.

A column-shift sentry asserts:
  - every player row has exactly 65 columns
  - per-quarter AF totals (cols 46/48/50/52) sum to col 5 (round AF)
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "data" / "live_snapshots"

EXPECTED_COLS = 65
URL_TEMPLATE = "https://www.fanfooty.com.au/live/{gameid}.txt"

# Player column schema. Indexes are 0-based positions in the comma-split row.
# Fields decoded with high confidence are named; uncertain ones are flagged
# `colNN_unknown` so the JSON snapshot stays self-describing.
PLAYER_COLUMNS = [
    "player_id",        # 0
    "first_name",       # 1
    "surname",          # 2
    "team",             # 3 - 2-letter team code (RI, AD, ...)
    "col4_unknown",     # 4 - small int, possibly recent-quarter activity
    "af",               # 5 - AFL Fantasy total (= sum of cols 46/48/50/52)
    "sc",               # 6 - SuperCoach total
    "proj_af",          # 7 - projected AF (high)
    "proj_low",         # 8 - projected AF (low)
    "proj_sc",          # 9 - projected SuperCoach
    "kicks",            # 10
    "handballs",        # 11
    "marks",            # 12
    "tackles",          # 13
    "hitouts",          # 14
    "goals",            # 15 - WARNING: does NOT equal goals-in-this-game; sum across team != team's actual goals; may be season total or score contributions; treat as unreliable for individual game goal counts
    "behinds",          # 16
    "frees_for",        # 17
    "frees_against",    # 18
    "status",           # 19 - quarter / Half Time / Full Time / etc.
    "tag_primary",      # 20 - hot/cold/job/wing/...
    "blurb_template",   # 21 - "%P including %K..." raw template
    "tag_secondary",    # 22 - secondary role tag
    "matchup_note",     # 23 - "Starting on a HBF on Neal-Bullen"
    "col24_unknown",    # 24
    "col25_unknown",    # 25
    "col26_unknown",    # 26
    "col27_unknown",    # 27
    "position",         # 28 - Forward/Back/Midfielder/Ruck (may be blank)
    "jumper",           # 29
    "col30_unknown",    # 30
    "col31_unknown",    # 31
    "col32_unknown",    # 32
    "col33_pct",        # 33 - decimal (ownership / salary / ?)
    "col34_pct",        # 34
    "col35_pct",        # 35
    "col36_unknown",    # 36
    "col37_unknown",    # 37
    "col38_unknown",    # 38
    "clangers",         # 39 - inferred
    "col40_unknown",    # 40
    "col41_unknown",    # 41
    "de_pct",           # 42 - disposal efficiency %
    "tog_pct",          # 43 - time on ground %
    "col44_unknown",    # 44 - large int (price-cycle marker?)
    "col45_unknown",    # 45
    "af_q1",            # 46
    "sc_q1",            # 47
    "af_q2",            # 48
    "sc_q2",            # 49
    "af_q3",            # 50
    "sc_q3",            # 51
    "af_q4",            # 52
    "sc_q4",            # 53
    "col54_unknown",    # 54
    "col55_unknown",    # 55
    "col56_unknown",    # 56
    "col57_unknown",    # 57
    "col58_unknown",    # 58
    "col59_unknown",    # 59
    "col60_unknown",    # 60
    "col61_unknown",    # 61
    "col62_unknown",    # 62
    "col63_unknown",    # 63
    "col64_unknown",    # 64
]
assert len(PLAYER_COLUMNS) == EXPECTED_COLS

# Numeric coercion targets. Fields not in this set stay as strings.
INT_FIELDS = {
    "col4_unknown", "af", "sc", "proj_af", "proj_low", "proj_sc",
    "kicks", "handballs", "marks", "tackles", "hitouts", "goals", "behinds",
    "frees_for", "frees_against", "jumper", "clangers",
    "de_pct", "tog_pct", "col44_unknown", "col45_unknown",
    "af_q1", "sc_q1", "af_q2", "sc_q2", "af_q3", "sc_q3", "af_q4", "sc_q4",
    "col24_unknown", "col25_unknown", "col26_unknown", "col27_unknown",
    "col30_unknown", "col31_unknown", "col32_unknown",
    "col36_unknown", "col37_unknown", "col38_unknown",
    "col40_unknown", "col41_unknown",
    "col54_unknown", "col55_unknown", "col56_unknown", "col57_unknown",
    "col58_unknown", "col59_unknown", "col60_unknown", "col61_unknown",
    "col62_unknown", "col63_unknown", "col64_unknown",
}
FLOAT_FIELDS = {"col33_pct", "col34_pct", "col35_pct"}


def fetch_live_text(gameid: str) -> str:
    """Download the live .txt feed for a given game id."""
    url = URL_TEMPLATE.format(gameid=gameid)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SuperCoach-VIA/fetch_live_match (research)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return raw


def parse_header(line: str) -> dict:
    """Line 1: home_full,home_short,away_full,away_short,round,home_score,away_score,status"""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        raise ValueError(f"header has {len(parts)} fields, expected >=8: {line!r}")
    return {
        "home_team_full": parts[0],
        "home_team_short": parts[1],
        "away_team_full": parts[2],
        "away_team_short": parts[3],
        "round": parts[4],
        "home_score": parts[5],
        "away_score": parts[6],
        "status": parts[7],
        "raw": line,
    }


def parse_meta(line: str) -> dict:
    """Line 2: date, year, time, venue, ..."""
    parts = [p.strip() for p in line.split(",")]
    return {
        "date": parts[0] if len(parts) > 0 else "",
        "year": parts[1] if len(parts) > 1 else "",
        "time": parts[2] if len(parts) > 2 else "",
        "venue": parts[3] if len(parts) > 3 else "",
        "weather": parts[9] if len(parts) > 9 else "",
        "temp_c": parts[10] if len(parts) > 10 else "",
        "raw": line,
    }


_TIME_RE = re.compile(r"\((Q[1-4])\s+([0-9]+:[0-9]{2})\)")
_TAG_RE = re.compile(r"<[^>]+>")


def parse_commentary(line: str) -> list[dict]:
    """Line 3 holds the m0nty commentary stream. Returns a list of
    {quarter, time, text} entries plus any 'note' entries (injury blurbs)."""
    if not line:
        return []
    # Strip the leading '#####' marker if present.
    body = line.lstrip("#")
    chunks = re.split(r"<br\s*/?>", body)
    events: list[dict] = []
    for chunk in chunks:
        text = _TAG_RE.sub("", chunk)
        text = unescape(text).replace("&#044;", ",").replace("&#039;", "'").strip()
        if not text:
            continue
        # m0nty commentary entries have a (Qx mm:ss) trailing tag
        m = _TIME_RE.search(text)
        if m:
            quarter, t = m.group(1), m.group(2)
            event_text = _TIME_RE.sub("", text).strip()
            event_text = re.sub(r"^m0nty:\s*", "", event_text)
            events.append({"quarter": quarter, "time": t, "text": event_text})
        else:
            cleaned = re.sub(r"^m0nty:\s*", "", text)
            if cleaned:
                events.append({"quarter": None, "time": None, "text": cleaned})
    return events


def parse_chat(line: str) -> list[dict]:
    """Line 4 holds the coach chat stream. Returns {user, text} entries."""
    if not line:
        return []
    chunks = re.split(r"<br\s*/?>", line)
    out = []
    for chunk in chunks:
        # Pull the username out of the <em ...>NAME</em> wrapper.
        m = re.search(r"<em[^>]*>([^<]*?)</em>:\s*(.*)$", chunk)
        if not m:
            continue
        user = unescape(_TAG_RE.sub("", m.group(1))).strip()
        text = unescape(_TAG_RE.sub("", m.group(2))).replace("&#044;", ",").strip()
        if user and text:
            out.append({"user": user, "text": text})
    return out


def _coerce(field: str, value: str):
    v = value.strip()
    if v == "":
        return None
    if field in INT_FIELDS:
        try:
            return int(v)
        except ValueError:
            try:
                return int(float(v))
            except ValueError:
                return v
    if field in FLOAT_FIELDS:
        try:
            return float(v)
        except ValueError:
            return v
    return v


def parse_player_row(line: str) -> dict:
    parts = line.split(",")
    if len(parts) != EXPECTED_COLS:
        raise AssertionError(
            f"COLUMN-SHIFT SENTRY: row has {len(parts)} cols, expected {EXPECTED_COLS}.\n"
            f"row: {line[:200]!r}..."
        )
    return {name: _coerce(name, parts[i]) for i, name in enumerate(PLAYER_COLUMNS)}


def assert_quarter_sum(players: list[dict]) -> None:
    """Hard sentry: AF Q1+Q2+Q3+Q4 must equal the round AF (col 5) for every player.
    A failure means the column schema has shifted and downstream parsing is unreliable."""
    bad: list[str] = []
    for p in players:
        af = p.get("af") or 0
        q_sum = sum((p.get(k) or 0) for k in ("af_q1", "af_q2", "af_q3", "af_q4"))
        if af != q_sum:
            bad.append(
                f"  {p.get('first_name')} {p.get('surname')} ({p.get('team')}): "
                f"AF={af} but Q1+Q2+Q3+Q4={q_sum}"
            )
    if bad:
        raise AssertionError(
            "COLUMN-SHIFT SENTRY: per-quarter AF does not sum to round AF for "
            f"{len(bad)} player(s):\n" + "\n".join(bad)
        )


def parse_snapshot(raw: str, gameid: str) -> dict:
    lines = raw.splitlines()
    if len(lines) < 5:
        raise ValueError(f"feed has only {len(lines)} lines, expected at least 5")

    header = parse_header(lines[0])
    meta = parse_meta(lines[1])
    commentary = parse_commentary(lines[2]) if len(lines) > 2 else []
    chat = parse_chat(lines[3]) if len(lines) > 3 else []

    players: list[dict] = []
    for ln in lines[4:]:
        if not ln.strip():
            continue
        # Ignore any trailing non-player section markers
        if not ln.split(",", 1)[0].strip().isdigit():
            continue
        players.append(parse_player_row(ln))

    assert_quarter_sum(players)

    return {
        "gameid": gameid,
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "header": header,
        "meta": meta,
        "commentary": commentary,
        "chat": chat,
        "players": players,
        "schema": {
            "expected_columns": EXPECTED_COLS,
            "column_order": PLAYER_COLUMNS,
            "quarter_sentry": "passed",
        },
    }


def slug_status(status: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", status.strip().lower()).strip("-")
    return s or "unknown"


def write_outputs(snapshot: dict) -> tuple[Path, Path]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    status_slug = slug_status(snapshot["header"].get("status", ""))
    gameid = snapshot["gameid"]

    json_path = SNAPSHOT_DIR / f"{gameid}_{ts}_{status_slug}.json"
    csv_path = SNAPSHOT_DIR / f"{gameid}_{ts}_players.csv"

    with json_path.open("w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAYER_COLUMNS)
        w.writeheader()
        for p in snapshot["players"]:
            w.writerow({k: ("" if p.get(k) is None else p.get(k)) for k in PLAYER_COLUMNS})

    return json_path, csv_path


def print_summary(snapshot: dict) -> None:
    h = snapshot["header"]
    print()
    print(f"Game {snapshot['gameid']}: {h['home_team_full']} {h['home_score']} "
          f"vs {h['away_team_full']} {h['away_score']}  [{h['status']}]")
    print(f"Round: {h['round']}  Venue: {snapshot['meta'].get('venue', '?')}")
    print(f"Players parsed: {len(snapshot['players'])}  "
          f"Commentary entries: {len(snapshot['commentary'])}  "
          f"Schema sentry: {snapshot['schema']['quarter_sentry']}")
    print()

    # Player rows carry a 2-letter team code (RI, AD, ...). Pick the codes that
    # actually appear in this snapshot's player rows, ordered to match the header.
    seen: list[str] = []
    for p in snapshot["players"]:
        code = p.get("team")
        if code and code not in seen:
            seen.append(code)
    if len(seen) >= 2:
        home_code, away_code = seen[0], seen[1]
    else:
        home_code = seen[0] if seen else ""
        away_code = ""

    for code, label in ((home_code, h["home_team_full"]),
                        (away_code, h["away_team_full"])):
        if not code:
            continue
        side = [p for p in snapshot["players"] if p.get("team") == code]
        for p in side:
            p["_disposals"] = (p.get("kicks") or 0) + (p.get("handballs") or 0)
        side.sort(key=lambda p: p["_disposals"], reverse=True)
        print(f"Top 5 disposal getters - {label} ({code}):")
        for p in side[:5]:
            print(f"  {p['_disposals']:3d} disp  ({p.get('kicks') or 0}k + "
                  f"{p.get('handballs') or 0}h)  AF {p.get('af')}  "
                  f"{p.get('first_name')} {p.get('surname')}")
        print()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    gameid = argv[1].strip()
    if not gameid.isdigit():
        print(f"gameid must be numeric, got {gameid!r}", file=sys.stderr)
        return 2

    raw = fetch_live_text(gameid)
    snapshot = parse_snapshot(raw, gameid)
    json_path, csv_path = write_outputs(snapshot)

    print_summary(snapshot)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
