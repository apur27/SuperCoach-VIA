#!/usr/bin/env python3
"""
Live match monitor - polls FanFooty every 90 seconds and updates the live read doc.

Usage:
    python scripts/live_match_monitor.py <gameid> <output_doc>

Example:
    python scripts/live_match_monitor.py 9781 \
        docs/coaches-strategy-corner/richmond-vs-adelaide-round-9-2026-q4-live.md

Stops automatically when match status is "Full Time" and writes final summary.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FETCH_SCRIPT = REPO / "scripts" / "fetch_live_match.py"
PYTHON = "/home/abhi/sourceCode/python/coding/.venv/bin/python"
POLL_SECONDS = 90


def run_fetcher(gameid: str) -> dict | None:
    result = subprocess.run(
        [PYTHON, str(FETCH_SCRIPT), gameid],
        capture_output=True, text=True, cwd=REPO
    )
    if result.returncode != 0:
        print(f"  [fetch error] {result.stderr[:200]}", flush=True)
        return None
    # Load the freshest snapshot JSON
    snaps = sorted((REPO / "data" / "live_snapshots").glob(f"{gameid}_*.json"))
    if not snaps:
        return None
    return json.loads(snaps[-1].read_text())


def disp(p: dict) -> int:
    return (p.get("kicks") or 0) + (p.get("handballs") or 0)


def team_totals(players: list, code: str) -> dict:
    side = [p for p in players if p.get("team") == code]
    return {
        "disposals": sum(disp(p) for p in side),
        "marks":     sum(p.get("marks") or 0 for p in side),
        "tackles":   sum(p.get("tackles") or 0 for p in side),
        "hitouts":   sum(p.get("hitouts") or 0 for p in side),
        "q4_af":     sum(p.get("af_q4") or 0 for p in side),
        "q3_af":     sum(p.get("af_q3") or 0 for p in side),
        "q2_af":     sum(p.get("af_q2") or 0 for p in side),
        "q1_af":     sum(p.get("af_q1") or 0 for p in side),
        "total_af":  sum(p.get("af") or 0 for p in side),
    }


def top_players(players: list, code: str, key: str, n: int = 5) -> list:
    side = [p for p in players if p.get("team") == code]
    for p in side:
        p["_disp"] = disp(p)
    return sorted(side, key=lambda p: p.get(key) or p.get("_disp") or 0, reverse=True)[:n]


def q_events(commentary: list, quarter: str) -> list:
    return [e for e in commentary if e.get("quarter") == quarter]


def format_snapshot(snap: dict, quarter: str) -> str:
    h = snap["header"]
    players = snap["players"]
    ts = snap["fetched_at_utc"]

    home_code = next((p["team"] for p in players), "RI")
    away_code = next((p["team"] for p in players if p["team"] != home_code), "AD")

    ht = team_totals(players, home_code)
    at = team_totals(players, away_code)

    # Quarter AF label
    q_af_key = {"Q4": "q4_af", "Q3": "q3_af"}.get(quarter, "q4_af")
    q_label = quarter

    # Top disposal leaders
    home_top = top_players(players, home_code, "_disp", 5)
    away_top = top_players(players, away_code, "_disp", 5)

    # Top quarter scorers
    home_q = top_players(players, home_code, q_af_key, 4)
    away_q = top_players(players, away_code, q_af_key, 4)

    # Recent commentary (current quarter only)
    q_commentary = q_events(snap.get("commentary", []), quarter)

    # --- Richmond win-condition analysis ---
    ri_players = [p for p in players if p.get("team") == home_code]
    ad_players = [p for p in players if p.get("team") == away_code]

    ri_q4    = ht.get(q_af_key, 0)
    ad_q4    = at.get(q_af_key, 0)
    ri_tk    = ht["tackles"]
    ad_tk    = at["tackles"]
    ri_ho    = ht["hitouts"]
    ad_ho    = at["hitouts"]

    # Lynch contested output
    lynch = next((p for p in ri_players if "Lynch" in (p.get("surname") or "")), None)
    lynch_line = (
        f"Lynch {disp(lynch)} disp / {lynch.get('goals') or 0} goals / "
        f"{lynch.get('marks') or 0} marks"
        if lynch else "Lynch not found in data"
    )
    # Taranto clearance load
    taranto = next((p for p in ri_players if "Taranto" in (p.get("surname") or "")), None)
    taranto_line = (
        f"Taranto {disp(taranto)} disp / {taranto.get('af') or 0} AF"
        if taranto else "Taranto not found"
    )
    # Rankine suppression check (is he still dominating?)
    rankine = next((p for p in ad_players if "Rankine" in (p.get("surname") or "")), None)
    rankine_line = (
        f"Rankine {disp(rankine)} disp / {rankine.get('goals') or 0} goals / "
        f"{rankine.get('sc') or 0} SC"
        if rankine else "Rankine not found"
    )

    # Score margin
    def score_pts(s: str) -> int:
        m = re.search(r"(\d+)$", s.strip())
        return int(m.group(1)) if m else 0

    ri_pts = score_pts(h.get("home_score", "0"))
    ad_pts = score_pts(h.get("away_score", "0"))
    margin = ri_pts - ad_pts
    margin_str = (
        f"Richmond +{margin}" if margin > 0 else
        f"Adelaide +{-margin}" if margin < 0 else "Level"
    )

    # Win condition assessment
    q4_momentum = (
        "Richmond winning Q4" if ri_q4 > ad_q4 else
        "Adelaide winning Q4" if ad_q4 > ri_q4 else "Even Q4"
    )
    tackle_edge = "Richmond" if ri_tk > ad_tk else "Adelaide" if ad_tk > ri_tk else "Level"

    win_conditions = [
        f"**Margin:** {margin_str}",
        f"**Q4 momentum:** {q4_momentum} ({ri_q4} vs {ad_q4} AF)",
        f"**Lynch (Richmond's forward key):** {lynch_line}",
        f"**Taranto (Richmond's engine):** {taranto_line}",
        f"**Rankine (must contain):** {rankine_line}",
        f"**Tackle battle:** {tackle_edge} ({ri_tk} vs {ad_tk})",
        f"**Hit-outs:** Adelaide +{ad_ho - ri_ho} ({ad_ho} vs {ri_ho})",
    ]

    # Richmond's path: need Q4 burst, Lynch goals, stop Rankine, win clearances
    if margin < 0:
        deficit = -margin
        goals_needed = (deficit // 6) + 1
        path = (
            f"Richmond trail by {deficit} pts. Need {goals_needed}+ unanswered goals. "
            f"{'Q4 burst is still live.' if deficit <= 18 else 'Steep climb - need a run.'}"
        )
    elif margin > 0:
        path = f"Richmond lead by {margin} pts. Hold the tackle pressure and protect Q4."
    else:
        path = "Level game. First goal of Q4 is decisive."

    lines = [
        f"### Snapshot - {h['status']} | {ts}",
        f"**Score:** {h['home_team_full']} {h['home_score']} - {h['away_team_full']} {h['away_score']}",
        "",
        "**Richmond win-condition check:**",
        *win_conditions,
        "",
        f"*Path to Richmond win: {path}*",
        "",
        "**Team stats:**",
        f"| | {h['home_team_full']} | {h['away_team_full']} |",
        "|---|---|---|",
        f"| Disposals | {ht['disposals']} | {at['disposals']} |",
        f"| Marks | {ht['marks']} | {at['marks']} |",
        f"| Tackles | {ht['tackles']} | {at['tackles']} |",
        f"| Hit-outs | {ht['hitouts']} | {at['hitouts']} |",
        f"| Q1 AF | {ht['q1_af']} | {at['q1_af']} |",
        f"| Q2 AF | {ht['q2_af']} | {at['q2_af']} |",
        f"| Q3 AF | {ht['q3_af']} | {at['q3_af']} |",
        f"| {q_label} AF | {ht[q_af_key]} | {at[q_af_key]} |",
        "",
        f"**Top disposal leaders:**",
        f"| Player | Team | Disp | Goals | SC | Tackles |",
        "|---|---|---|---|---|---|",
    ]
    for p in (home_top + away_top):
        team_label = h['home_team_full'] if p['team'] == home_code else h['away_team_full']
        lines.append(
            f"| {p.get('first_name')} {p.get('surname')} | {team_label} | "
            f"{disp(p)} | {p.get('goals') or 0} | {p.get('sc') or 0} | {p.get('tackles') or 0} |"
        )

    lines += [
        "",
        f"**{q_label} AF leaders (hot right now):**",
        f"| Player | Team | {q_label} AF | Total AF |",
        "|---|---|---|---|",
    ]
    for p in (home_q + away_q):
        if (p.get(q_af_key) or 0) > 0:
            team_label = h['home_team_full'] if p['team'] == home_code else h['away_team_full']
            lines.append(
                f"| {p.get('first_name')} {p.get('surname')} | {team_label} | "
                f"{p.get(q_af_key) or 0} | {p.get('af') or 0} |"
            )

    if q_commentary:
        lines += ["", f"**{q_label} events:**"]
        for e in q_commentary[-5:]:  # last 5 events
            lines.append(f"- *({e['quarter']} {e['time']})* {e['text'][:120]}")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def write_doc(doc_path: Path, header_written: bool, snap: dict, quarter: str) -> bool:
    content = format_snapshot(snap, quarter)
    if not header_written:
        h = snap["header"]
        players = snap["players"]
        home_code = next((p["team"] for p in players), "RI")
        away_code = next((p["team"] for p in players if p["team"] != home_code), "AD")
        preamble = f"""# {h['home_team_full']} vs {h['away_team_full']} - Q4 Live Read (Round 9, 2026)

> Pre-match brief: [Executive summary](richmond-vs-adelaide-round-9-2026-executive-summary.md) | [Tactical brief](richmond-vs-adelaide-round-9-2026.md)
> In-game reads: [Half-time](richmond-vs-adelaide-round-9-2026-half-time-live.md) · [Q3](richmond-vs-adelaide-round-9-2026-q3-live.md)
>
> Auto-updated every 90 seconds from FanFooty live feed (game {snap['gameid']}).
> Script: `scripts/live_match_monitor.py`

Live snapshots are appended below in reverse-chronological order (newest at top).

---

"""
        doc_path.write_text(preamble + content)
    else:
        existing = doc_path.read_text()
        # Insert new snapshot after the header block (after the first ---)
        insert_after = "---\n\n"
        idx = existing.find(insert_after)
        if idx >= 0:
            insert_at = idx + len(insert_after)
            doc_path.write_text(existing[:insert_at] + content + existing[insert_at:])
        else:
            doc_path.write_text(existing + "\n" + content)
    return True


def git_commit_push(doc_path: Path, status: str) -> None:
    rel = doc_path.relative_to(REPO)
    snaps = list((REPO / "data" / "live_snapshots").glob("*.json")) + \
            list((REPO / "data" / "live_snapshots").glob("*.csv"))
    all_files = [str(rel)] + [str(s.relative_to(REPO)) for s in sorted(snaps)[-2:]]
    subprocess.run(["git", "add"] + all_files, cwd=REPO, capture_output=True)
    msg = f"Live update: {status} | auto-monitor\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO, capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=REPO, capture_output=True)
    print(f"  [pushed] {status}", flush=True)


def main(argv: list) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2

    gameid = argv[1]
    doc_path = REPO / argv[2]
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Monitoring game {gameid} -> {doc_path}", flush=True)
    print(f"Polling every {POLL_SECONDS}s. Stops on Full Time.", flush=True)

    header_written = False
    last_status = ""
    iteration = 0

    # Detect current quarter from first fetch
    current_quarter = "Q4"

    while True:
        iteration += 1
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Poll #{iteration}", flush=True)

        snap = run_fetcher(gameid)
        if snap is None:
            print("  Fetch failed, retrying next cycle.", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        status = snap["header"].get("status", "").strip()
        print(f"  Status: {status} | Score: {snap['header']['home_score']} vs {snap['header']['away_score']}", flush=True)

        # Detect quarter from status string
        if "Q4" in status or "q4" in status.lower():
            current_quarter = "Q4"
        elif "Full Time" in status or "FT" in status:
            current_quarter = "Q4"  # use Q4 data for final summary

        header_written = write_doc(doc_path, header_written, snap, current_quarter)

        if status != last_status or iteration % 3 == 0:
            git_commit_push(doc_path, status)
            last_status = status

        if "Full Time" in status or status.lower() == "ft":
            print("\nFull Time detected. Writing final summary and stopping.", flush=True)
            # One final push to make sure latest is captured
            git_commit_push(doc_path, "Full Time - FINAL")
            break

        time.sleep(POLL_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
