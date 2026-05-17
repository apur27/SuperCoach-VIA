#!/usr/bin/env python3
"""
Live analysis pipeline for Richmond vs St Kilda Round 11 2026 (FanFooty game 9789).

Replaces live_match_monitor.py with:
- Structured stats block (top disposal leaders, team metrics, key-player tracking).
- Plain-English tactical READ generated from conditional rules.
- Automatic quarter routing: writes to the correct quarter doc based on status,
  and emits a QUARTER BREAK summary when the status transitions.
- Commits and pushes every cycle (every 90s).

Usage:
    python scripts/live_analysis_pipeline.py <gameid>

Stops automatically when status is "Full Time".

Data reliability notes (see Scientist memory snapshot_data_quality.md):
- Goals / behinds / clangers per-player columns are misindexed. We avoid them.
- Header home_score / away_score is the authoritative scoreline.
- Inside-50 / contested-poss / clearances are NOT in the per-player snapshot
  schema. We compute the best available proxies and label them honestly.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FETCH_SCRIPT = REPO / "scripts" / "fetch_live_match.py"
SNAP_DIR = REPO / "data" / "live_snapshots"
DOC_DIR = REPO / "docs" / "coaches-strategy-corner"
PYTHON = "/home/abhi/sourceCode/python/coding/.venv/bin/python"
POLL_SECONDS = 90

# Pre-match predictions for key players (from R11 pre-match brief).
KEY_PLAYERS = {
    # surname -> (team_code, predicted_af_or_disp, role_note)
    "Short":    ("RI", 23, "Rebound launcher"),
    "Sinclair": ("SK", 27, "Rebound general"),
    "Hill":     ("SK", 22, "Milera cover - Back"),
}

# Doc routing for each game state.
DOC_BASE = "richmond-vs-stkilda-round-11-2026"
DOC_FOR_QUARTER = {
    "Q1":      f"{DOC_BASE}-q1-live.md",
    "QT":      f"{DOC_BASE}-q1-live.md",        # end-of-Q1 summary goes on Q1 doc
    "Q2":      f"{DOC_BASE}-q2-live.md",
    "HT":      f"{DOC_BASE}-half-time-live.md",
    "Q3":      f"{DOC_BASE}-q3-live.md",
    "3QT":     f"{DOC_BASE}-q3-live.md",
    "Q4":      f"{DOC_BASE}-q4-live.md",
    "FT":      f"{DOC_BASE}-full-time-verdict.md",
}

# Sequence used to detect transitions.
QUARTER_ORDER = ["Q1", "QT", "Q2", "HT", "Q3", "3QT", "Q4", "FT"]


# ---------------------------------------------------------------------------
# Fetch & parse
# ---------------------------------------------------------------------------

def run_fetcher(gameid: str) -> dict | None:
    """Run the fetch_live_match.py helper and load the newest snapshot."""
    result = subprocess.run(
        [PYTHON, str(FETCH_SCRIPT), gameid],
        capture_output=True, text=True, cwd=REPO,
    )
    if result.returncode != 0:
        print(f"  [fetch error] {result.stderr[:200]}", flush=True)
        return None
    snaps = sorted(SNAP_DIR.glob(f"{gameid}_*.json"))
    if not snaps:
        return None
    try:
        return json.loads(snaps[-1].read_text())
    except json.JSONDecodeError as e:
        print(f"  [json error] {e}", flush=True)
        return None


def disp(p: dict) -> int:
    return (p.get("kicks") or 0) + (p.get("handballs") or 0)


def parse_score_pts(score_str: str) -> int:
    """'1.1.7' -> 7 (total points). Returns 0 on failure."""
    if not score_str:
        return 0
    m = re.search(r"(\d+)\s*$", score_str.strip())
    return int(m.group(1)) if m else 0


def classify_status(status: str) -> str:
    """Map FanFooty status string -> normalised quarter code in QUARTER_ORDER."""
    s = (status or "").strip().lower()
    if "full time" in s or s == "ft":
        return "FT"
    if "three quarter time" in s or "3qt" in s or "3 qtr" in s:
        return "3QT"
    if "q4" in s:
        return "Q4"
    if "half time" in s or s == "ht":
        return "HT"
    if "q3" in s:
        return "Q3"
    if "quarter time" in s or s == "qt":
        return "QT"
    if "q2" in s:
        return "Q2"
    if "q1" in s:
        return "Q1"
    return "Q1"


# ---------------------------------------------------------------------------
# Stat aggregation
# ---------------------------------------------------------------------------

def team_totals(players: list, code: str) -> dict:
    side = [p for p in players if p.get("team") == code]
    return {
        "disposals": sum(disp(p) for p in side),
        "kicks":     sum(p.get("kicks") or 0 for p in side),
        "handballs": sum(p.get("handballs") or 0 for p in side),
        "marks":     sum(p.get("marks") or 0 for p in side),
        "tackles":   sum(p.get("tackles") or 0 for p in side),
        "hitouts":   sum(p.get("hitouts") or 0 for p in side),
        "frees_for": sum(p.get("frees_for") or 0 for p in side),
        "total_af":  sum(p.get("af") or 0 for p in side),
        "q1_af":     sum(p.get("af_q1") or 0 for p in side),
        "q2_af":     sum(p.get("af_q2") or 0 for p in side),
        "q3_af":     sum(p.get("af_q3") or 0 for p in side),
        "q4_af":     sum(p.get("af_q4") or 0 for p in side),
    }


def top_disposers(players: list, code: str, n: int = 3) -> list:
    side = [p for p in players if p.get("team") == code]
    return sorted(side, key=disp, reverse=True)[:n]


def find_player(players: list, surname: str, team_code: str | None = None) -> dict | None:
    for p in players:
        if p.get("surname", "").lower() == surname.lower():
            if team_code is None or p.get("team") == team_code:
                return p
    return None


# ---------------------------------------------------------------------------
# Trend tracking (compare to previous snapshot in memory)
# ---------------------------------------------------------------------------

class TrendCache:
    """Keep the previous cycle's per-player disposals so we can show arrows."""
    def __init__(self) -> None:
        self.last: dict[str, int] = {}

    def arrow(self, player_id: str, current: int) -> str:
        prev = self.last.get(player_id)
        self.last[player_id] = current
        if prev is None:
            return "="
        if current > prev:
            return "up"
        if current < prev:
            return "down"
        return "="


# ---------------------------------------------------------------------------
# Tactical read generator
# ---------------------------------------------------------------------------

def generate_read(
    home_code: str,
    home_full: str,
    away_code: str,
    away_full: str,
    ht: dict,
    at: dict,
    players: list,
    home_pts: int,
    away_pts: int,
    status_code: str,
) -> str:
    """Compose 2-3 sentence tactical read from rule conditions.

    NOTE: home is St Kilda (home_team_full from the snapshot header).
    The user-facing tripwire framing is *Richmond perspective* - they are away.
    """
    # We need a Richmond-centric read because the strategy work is RIC-side.
    if home_code == "RI":
        ric = ht; stk = at; ric_pts = home_pts; stk_pts = away_pts
    else:
        ric = at; stk = ht; ric_pts = away_pts; stk_pts = home_pts

    margin = ric_pts - stk_pts
    sentences: list[str] = []

    # 1. Possession / tempo
    disp_gap = ric["disposals"] - stk["disposals"]
    if abs(disp_gap) <= 5:
        sentences.append(
            f"Possession is close ({ric['disposals']}-{stk['disposals']}), "
            f"neither side has yet seized control of the tempo."
        )
    elif disp_gap > 0:
        sentences.append(
            f"Richmond out-possessing St Kilda {ric['disposals']}-{stk['disposals']} "
            f"({disp_gap:+d}) - the Tigers are dictating ball movement."
        )
    else:
        sentences.append(
            f"St Kilda dominating possession {stk['disposals']}-{ric['disposals']} "
            f"({-disp_gap:+d}) - Richmond chasing without the ball."
        )

    # 2. Pressure / tackle proxy + ruck
    tackle_gap = ric["tackles"] - stk["tackles"]
    ho_gap = stk["hitouts"] - ric["hitouts"]
    pressure_bits = []
    if abs(tackle_gap) <= 2:
        pressure_bits.append(f"tackle pressure even ({ric['tackles']}-{stk['tackles']})")
    elif tackle_gap > 0:
        pressure_bits.append(
            f"Richmond winning the tackle count ({ric['tackles']}-{stk['tackles']})"
        )
    else:
        pressure_bits.append(
            f"St Kilda applying more pressure ({stk['tackles']}-{ric['tackles']} tackles)"
        )
    if ho_gap >= 5:
        pressure_bits.append(f"De Koning +{ho_gap} in the ruck giving STK first use")
    elif ho_gap <= -5:
        pressure_bits.append(f"Richmond ruck +{-ho_gap} - rare territory edge")
    sentences.append(("; ".join(pressure_bits) + ".").capitalize())

    # 3. Key-player check (Short, Sinclair, Hill)
    short = find_player(players, "Short", "RI")
    sinclair = find_player(players, "Sinclair", "SK")
    hill = find_player(players, "Hill", "SK")

    key_bits = []
    if short is not None:
        d = disp(short); p = KEY_PLAYERS["Short"][1]
        if d >= p * 0.6:
            key_bits.append(f"Short running hot ({d} disp vs pred {p}) - Richmond rebound plan firing")
        elif d <= p * 0.3 and stk["disposals"] >= 60:
            key_bits.append(f"Short quiet ({d} vs pred {p}) - rebound source missing")
    if sinclair is not None:
        d = disp(sinclair); p = KEY_PLAYERS["Sinclair"][1]
        if d >= p * 0.7:
            key_bits.append(f"Sinclair on track ({d} vs pred {p})")
        elif d <= p * 0.3 and stk["disposals"] >= 60:
            key_bits.append(f"Sinclair contained ({d} vs pred {p}) - STK rebound general muted")
    if hill is not None:
        d = disp(hill); p = KEY_PLAYERS["Hill"][1]
        if d <= max(4, p * 0.4) and stk["disposals"] >= 60:
            key_bits.append(f"Hill not filling Milera role ({d} vs pred {p}) - gap exposed")
        elif d >= p * 0.7:
            key_bits.append(f"Hill stepping up in Milera's absence ({d} vs pred {p})")
    if key_bits:
        sentences.append(" ".join(b + "." for b in key_bits[:2]))

    # 4. Scoreboard / closing line
    if status_code == "FT":
        if margin > 0:
            sentences.append(f"Final: Richmond win by {margin}.")
        elif margin < 0:
            sentences.append(f"Final: Richmond lose by {-margin}.")
        else:
            sentences.append("Final: draw.")
    elif abs(margin) <= 12 and status_code in {"Q3", "3QT", "Q4"}:
        sentences.append(f"Margin {margin:+d} - contested battle, game still alive for Richmond.")
    elif margin <= -25:
        sentences.append(f"Richmond {-margin} down - tripwire territory, structural change needed.")
    elif margin >= 25:
        sentences.append(f"Richmond {margin} up - hold the pressure profile.")

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Block formatting
# ---------------------------------------------------------------------------

def fmt_player_oneliner(p: dict, trend: TrendCache | None = None) -> str:
    d = disp(p)
    k = p.get("kicks") or 0
    hb = p.get("handballs") or 0
    tk = p.get("tackles") or 0
    arrow = ""
    if trend is not None:
        a = trend.arrow(p.get("player_id", ""), d)
        arrow_map = {"up": " up", "down": " dn", "=": ""}
        arrow = arrow_map.get(a, "")
    return f"{p.get('first_name', '')[:1]}. {p.get('surname')} {d}({k}/{hb}) {tk}t{arrow}"


def format_analysis_block(
    snap: dict,
    status_code: str,
    trend_cache: TrendCache,
) -> str:
    h = snap["header"]
    players = snap["players"]
    ts = snap["fetched_at_utc"]

    home_code = h.get("home_team_short")  # "Saints" - need the per-player team code
    # Per-player team codes are different (SK / RI). Derive from players list directly.
    team_codes = list({p.get("team") for p in players if p.get("team")})
    # home is the team listed first in header. Cross-reference team_full -> team_code.
    home_full = h.get("home_team_full", "Home")
    away_full = h.get("away_team_full", "Away")
    # We know from inspection: SK = St Kilda, RI = Richmond.
    if home_full == "St Kilda":
        home_pc, away_pc = "SK", "RI"
    elif home_full == "Richmond":
        home_pc, away_pc = "RI", "SK"
    else:
        # Fallback - pick whichever team has more players, treat as home.
        home_pc = team_codes[0] if team_codes else "SK"
        away_pc = team_codes[1] if len(team_codes) > 1 else "RI"

    ht = team_totals(players, home_pc)
    at = team_totals(players, away_pc)

    home_pts = parse_score_pts(h.get("home_score", ""))
    away_pts = parse_score_pts(h.get("away_score", ""))

    # Always present Richmond first in the per-side tables (Richmond-side analysis).
    ric_pc, stk_pc = "RI", "SK"
    ric_t = team_totals(players, ric_pc)
    stk_t = team_totals(players, stk_pc)

    ric_top = top_disposers(players, ric_pc, 3)
    stk_top = top_disposers(players, stk_pc, 3)

    # Key player tracking
    short = find_player(players, "Short", "RI")
    sinclair = find_player(players, "Sinclair", "SK")
    hill = find_player(players, "Hill", "SK")

    def kp_line(p, label, pred):
        if p is None:
            return f"- {label} (pred {pred}): not on field / data not present"
        d = disp(p)
        arrow_raw = trend_cache.arrow(p.get("player_id", ""), d)
        arrow_map = {"up": "up", "down": "dn", "=": "="}
        arrow = arrow_map.get(arrow_raw, "=")
        extra = ""
        if label.startswith("Hill"):
            extra = f", {p.get('marks') or 0}m"
        ratio_warn = ""
        if pred > 0:
            ratio = d / pred
            if ratio >= 0.7:
                ratio_warn = " (running ahead of rate)"
            elif ratio <= 0.3 and stk_t["disposals"] >= 60:
                ratio_warn = " (well below rate)"
        return f"- {label} (pred {pred}): {d} disp{extra} [{arrow}]{ratio_warn}"

    # Tripwire (Inside 50s not available - use disposal-share proxy and label it).
    # Best proxy for territory in the absence of I50: kicks share.
    kicks_ric = ric_t["kicks"]
    kicks_stk = stk_t["kicks"]
    tripwire_label = (
        "CALL HOLDS"
        if kicks_ric >= kicks_stk
        else "TRIPWIRE TRIGGERED (St Kilda controlling kick-territory proxy)"
    )

    # Quarter-AF mini-row (the currently-active quarter only)
    q_key = {"Q1": "q1_af", "QT": "q1_af", "Q2": "q2_af", "HT": "q2_af",
             "Q3": "q3_af", "3QT": "q3_af", "Q4": "q4_af", "FT": "q4_af"}[status_code]
    q_label = {"Q1": "Q1", "QT": "Q1 (end)", "Q2": "Q2", "HT": "Q2 (end)",
               "Q3": "Q3", "3QT": "Q3 (end)", "Q4": "Q4", "FT": "Q4 (final)"}[status_code]

    read = generate_read(
        home_pc, home_full, away_pc, away_full, ht, at, players,
        home_pts, away_pts, status_code,
    )

    lines = [
        "---",
        f"### {h.get('status', status_code)} - {home_full} {h.get('home_score')} vs {away_full} {h.get('away_score')} - {ts}",
        "",
        "**Disposal leaders - Richmond:** "
        + " | ".join(fmt_player_oneliner(p, trend_cache) for p in ric_top),
        "**Disposal leaders - St Kilda:** "
        + " | ".join(fmt_player_oneliner(p, trend_cache) for p in stk_top),
        "",
        "| Metric | RIC | STK |",
        "|--------|-----|-----|",
        f"| Disposals (K+HB) | {ric_t['disposals']} ({ric_t['kicks']}/{ric_t['handballs']}) | {stk_t['disposals']} ({stk_t['kicks']}/{stk_t['handballs']}) |",
        f"| Marks | {ric_t['marks']} | {stk_t['marks']} |",
        f"| Tackles | {ric_t['tackles']} | {stk_t['tackles']} |",
        f"| Hit-outs | {ric_t['hitouts']} | {stk_t['hitouts']} |",
        f"| Frees for | {ric_t['frees_for']} | {stk_t['frees_for']} |",
        f"| Total AF | {ric_t['total_af']} | {stk_t['total_af']} |",
        f"| {q_label} AF | {ric_t[q_key]} | {stk_t[q_key]} |",
        "",
        "*Inside 50s / contested poss / clearances are not in the FanFooty per-player snapshot schema. Kick-share used as a proxy below.*",
        "",
        f"**Tripwire (kick-share proxy):** RIC {kicks_ric} - STK {kicks_stk} -> {tripwire_label}",
        "",
        "**Key player tracking:**",
        kp_line(short, "Short", KEY_PLAYERS["Short"][1]),
        kp_line(sinclair, "Sinclair", KEY_PLAYERS["Sinclair"][1]),
        kp_line(hill, "Hill, Milera cover", KEY_PLAYERS["Hill"][1]),
        "",
        f"**Read:** {read}",
        "",
        f"*[data] - FanFooty snapshot {snap['gameid']}, {ts}*",
        "",
    ]
    return "\n".join(lines)


def format_quarter_break(prev_code: str, snap: dict) -> str:
    """Compact summary written when status transitions across a break."""
    h = snap["header"]
    players = snap["players"]
    ric_t = team_totals(players, "RI")
    stk_t = team_totals(players, "SK")
    q_key = {"Q1": "q1_af", "Q2": "q2_af", "Q3": "q3_af", "Q4": "q4_af"}.get(prev_code, "q1_af")
    home_full = h.get("home_team_full", "Home")
    away_full = h.get("away_team_full", "Away")
    return "\n".join([
        "",
        "---",
        f"### QUARTER BREAK: end of {prev_code} - {home_full} {h.get('home_score')} vs {away_full} {h.get('away_score')}",
        "",
        f"**{prev_code} AF:** RIC {ric_t[q_key]} - STK {stk_t[q_key]}",
        f"**Cumulative disposals:** RIC {ric_t['disposals']} - STK {stk_t['disposals']}",
        f"**Cumulative tackles:** RIC {ric_t['tackles']} - STK {stk_t['tackles']}",
        f"**Cumulative hit-outs:** RIC {ric_t['hitouts']} - STK {stk_t['hitouts']}",
        "",
        f"*Routing forward to the next quarter doc.*",
        "",
    ])


# ---------------------------------------------------------------------------
# Doc I/O
# ---------------------------------------------------------------------------

def doc_path_for(status_code: str) -> Path:
    name = DOC_FOR_QUARTER.get(status_code, DOC_FOR_QUARTER["Q1"])
    return DOC_DIR / name


def ensure_header(path: Path, snap: dict, status_code: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    h = snap["header"]
    home_full = h.get("home_team_full", "Home")
    away_full = h.get("away_team_full", "Away")
    title = {
        "Q1":  "Q1 live read",
        "Q2":  "Q2 live read",
        "HT":  "Half-time live read",
        "Q3":  "Q3 live read",
        "Q4":  "Q4 live read",
        "FT":  "Full-time verdict",
    }.get(status_code, f"{status_code} live read")
    header = (
        f"# {home_full} vs {away_full} - {title}, Round 11, 2026\n"
        f"\n"
        f"> [Back to strategy corner](README.md) | "
        f"[Pre-match brief]({DOC_BASE}.md)\n"
        f"\n"
        f"Auto-updated every {POLL_SECONDS}s from FanFooty live feed (game {snap['gameid']}). "
        f"Newest snapshot at top.\n"
        f"\n"
    )
    path.write_text(header)


AUTO_MARKER = "<!-- LIVE_ANALYSIS_AUTO_BLOCKS_BELOW -->"


def insert_block(path: Path, block: str) -> None:
    """Insert block under the auto marker, so newest is first.

    If the marker isn't present yet (e.g. doc was hand-authored before the
    pipeline started), append a labelled auto-blocks section to the bottom
    of the existing doc and put the marker there.
    """
    if not path.exists():
        path.write_text(block)
        return
    existing = path.read_text()
    if AUTO_MARKER not in existing:
        existing = (
            existing.rstrip()
            + "\n\n---\n\n"
            + "## Auto-updated live analysis (newest first)\n\n"
            + AUTO_MARKER + "\n\n"
        )
    idx = existing.find(AUTO_MARKER)
    insert_at = idx + len(AUTO_MARKER) + 1  # past the trailing newline
    path.write_text(existing[:insert_at] + "\n" + block + existing[insert_at:])


def git_commit_push(paths: list[Path], status: str) -> None:
    rel_files = [str(p.relative_to(REPO)) for p in paths if p.exists()]
    snaps = sorted(SNAP_DIR.glob("*.json"))[-2:] + sorted(SNAP_DIR.glob("*.csv"))[-2:]
    snap_rel = [str(s.relative_to(REPO)) for s in snaps]
    add_list = rel_files + snap_rel
    if not add_list:
        return
    subprocess.run(["git", "add"] + add_list, cwd=REPO, capture_output=True)
    msg = (
        f"Live analysis 9789: {status}\n\n"
        f"Auto-pushed by live_analysis_pipeline.py\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    r = subprocess.run(["git", "commit", "-m", msg], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
        print(f"  [commit issue] {r.stdout[-120:]} {r.stderr[-120:]}", flush=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=REPO, capture_output=True)
    print(f"  [pushed] {status}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    gameid = argv[1]
    print(f"Starting live analysis pipeline for game {gameid}", flush=True)
    print(f"Poll interval: {POLL_SECONDS}s. Stops on Full Time.", flush=True)

    trend_cache = TrendCache()
    last_status_code: str | None = None
    iteration = 0

    while True:
        iteration += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] poll #{iteration}", flush=True)

        snap = run_fetcher(gameid)
        if snap is None:
            print("  fetch failed, will retry next cycle", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        h = snap["header"]
        status_raw = (h.get("status") or "").strip()
        status_code = classify_status(status_raw)
        print(
            f"  status='{status_raw}' code={status_code} "
            f"score: {h.get('home_team_full')} {h.get('home_score')} - "
            f"{h.get('away_team_full')} {h.get('away_score')}",
            flush=True,
        )

        # On quarter transition, write a QUARTER BREAK summary to the OLD doc first.
        paths_touched: list[Path] = []
        if (
            last_status_code is not None
            and last_status_code != status_code
            and last_status_code in {"Q1", "Q2", "Q3", "Q4"}
            and status_code in {"QT", "HT", "3QT", "FT", "Q2", "Q3", "Q4"}
        ):
            old_path = doc_path_for(last_status_code)
            ensure_header(old_path, snap, last_status_code)
            insert_block(old_path, format_quarter_break(last_status_code, snap))
            paths_touched.append(old_path)
            print(f"  [break written] end of {last_status_code} -> {old_path.name}", flush=True)

        # Write the live analysis block to the current quarter's doc.
        current_path = doc_path_for(status_code)
        ensure_header(current_path, snap, status_code)
        block = format_analysis_block(snap, status_code, trend_cache)
        insert_block(current_path, block)
        paths_touched.append(current_path)
        print(f"  [block written] {status_code} -> {current_path.name}", flush=True)

        git_commit_push(paths_touched, f"{status_raw} -> {current_path.name}")
        last_status_code = status_code

        if status_code == "FT":
            print("\nFull Time reached. Final block written. Exiting.", flush=True)
            break

        time.sleep(POLL_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
