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
    """Map FanFooty status string -> normalised quarter code in QUARTER_ORDER.

    Confirmed FanFooty status strings observed in R10/R11 2026 snapshots:
        "Q1 6:42", "Qtr Time", "Q2 14:23", "Half Time", "Q3 19:11",
        "3 Qtr Time", "Q4 28:36", "Final Siren", "Full Time".

    Order of checks matters:
      * Game-over states ("Full Time" / "Final Siren") MUST match first so a
        scoreboard left frozen at the siren does not get re-classified as Q4.
      * "3 Qtr Time" must match BEFORE the generic "qtr time" rule so it
        routes forward to 3QT, not backward to QT.
      * In-quarter strings ("Q3 19:11") must match BEFORE break strings, since
        the live string still contains the quarter token.

    Returns None as a sentinel when the status is genuinely unrecognisable -
    the caller must decide whether to skip the cycle or fall back to the last
    known good state. Returning "Q1" as a silent fallback (the previous
    behaviour) was the root cause of the R11 "Final Siren -> Q1 doc" routing
    bug, where 8 polls wrote end-of-game data into the Q1 live document.
    """
    s = (status or "").strip().lower()
    if not s:
        return None  # type: ignore[return-value]
    # ---- Game-over states (must come first) --------------------------------
    if "full time" in s or s == "ft" or "final siren" in s or s == "fs":
        return "FT"
    # ---- Three-quarter-time variants (must come before generic "qtr time") -
    if "three quarter time" in s or "3qt" in s or "3 qtr" in s or "3qtr" in s:
        return "3QT"
    # ---- In-quarter strings (Qn HH:MM) - match before break tokens ----------
    if "q4" in s:
        return "Q4"
    if "q3" in s:
        return "Q3"
    if "q2" in s:
        return "Q2"
    if "q1" in s:
        return "Q1"
    # ---- Break states (only after in-quarter checks have failed) -----------
    if "half time" in s or s == "ht":
        return "HT"
    if "quarter time" in s or "qtr time" in s or s == "qt":
        return "QT"
    # Unrecognised - return None and let the caller decide.
    return None  # type: ignore[return-value]


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


def top_quarter_af(players: list, code: str, q_key: str, n: int = 3) -> list:
    """Return the top-n players on team `code` by current-quarter AF only.

    This shows who is hot RIGHT NOW (e.g. q4_af), not who has built the biggest
    cumulative total over the whole match. A midfielder with 8 AF this quarter
    matters more for the current-block read than a player sitting on 110 total.
    """
    side = [p for p in players if p.get("team") == code]
    return sorted(side, key=lambda p: (p.get(q_key) or 0), reverse=True)[:n]


def find_player(players: list, surname: str, team_code: str | None = None) -> dict | None:
    for p in players:
        if p.get("surname", "").lower() == surname.lower():
            if team_code is None or p.get("team") == team_code:
                return p
    return None


def player_key(p: dict) -> str:
    """Stable key for cross-poll player lookup."""
    pid = p.get("player_id")
    if pid:
        return str(pid)
    return f"{p.get('team', '?')}|{p.get('surname', '?')}|{p.get('first_name', '?')}"


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

def _q_key_for(status_code: str) -> str:
    """Per-PLAYER quarter-AF field name (snapshot schema uses af_qN)."""
    return {"Q1": "af_q1", "QT": "af_q1", "Q2": "af_q2", "HT": "af_q2",
            "Q3": "af_q3", "3QT": "af_q3", "Q4": "af_q4", "FT": "af_q4"}.get(status_code, "af_q4")


def _q_label_short(status_code: str) -> str:
    return {"Q1": "Q1", "QT": "Q1", "Q2": "Q2", "HT": "Q2",
            "Q3": "Q3", "3QT": "Q3", "Q4": "Q4", "FT": "Q4"}.get(status_code, "Qx")


def _build_dynamic_read(
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
    prev_state: dict | None,
) -> str:
    """Compose a tactical read that EVOLVES across polls.

    Replaces the prior Mad-Libs template by:
      * Comparing the current snapshot to `prev_state` (deltas in disposals,
        tackles, score, and per-player stats) and leading with what CHANGED.
      * Surfacing the leader of the *current quarter's* AF, not just cumulative.
      * Choosing a narrative frame from game-state bands (close / chasing /
        blow-out) and from the status_code (quarter break vs in-play).
      * Flagging rising players (>=3 disp or >=2 tackle jump since last poll).

    `prev_state` is None on the first poll - the function falls back to a pure
    snapshot read in that case.
    """
    # Richmond-centric framing.
    if home_code == "RI":
        ric, stk = ht, at
        ric_pts, stk_pts = home_pts, away_pts
    else:
        ric, stk = at, ht
        ric_pts, stk_pts = away_pts, home_pts

    margin = ric_pts - stk_pts
    abs_margin = abs(margin)

    # Game-state band - drives the framing of every other sentence.
    if status_code == "FT":
        band = "final"
    elif abs_margin < 15:
        band = "close"
    elif abs_margin <= 30:
        band = "chasing" if margin < 0 else "leading_mid"
    else:
        band = "blowout_down" if margin < 0 else "blowout_up"

    # Quarter-break vs in-play.
    is_break = status_code in {"QT", "HT", "3QT", "FT"}

    sentences: list[str] = []

    # -----------------------------------------------------------------------
    # 1. SCORE DELTA - the most newsworthy single fact since last poll.
    # -----------------------------------------------------------------------
    if prev_state is not None:
        ric_score_delta = ric_pts - prev_state.get("ric_pts", ric_pts)
        stk_score_delta = stk_pts - prev_state.get("stk_pts", stk_pts)
        # New goals (>= 6 pts) get a special call-out; behinds get a lighter one.
        if stk_score_delta >= 12 and ric_score_delta == 0:
            sentences.append(
                f"Saints have piled on {stk_score_delta} unanswered since last poll - "
                f"margin out to {abs_margin}."
            )
        elif ric_score_delta >= 12 and stk_score_delta == 0:
            sentences.append(
                f"Richmond {ric_score_delta} unanswered - margin "
                f"{('cut to ' + str(abs_margin)) if margin <= 0 else ('out to +' + str(margin))}."
            )
        elif stk_score_delta >= 6 and ric_score_delta == 0:
            sentences.append(f"Saints kicked a goal since last poll, margin now {margin:+d}.")
        elif ric_score_delta >= 6 and stk_score_delta == 0:
            sentences.append(f"Richmond responded with a goal, margin now {margin:+d}.")
        elif stk_score_delta == 0 and ric_score_delta == 0 and not is_break:
            sentences.append(
                f"Scoreboard stalemate over the last block - margin holds at {margin:+d}."
            )
        elif (ric_score_delta + stk_score_delta) > 0:
            sentences.append(
                f"Trade since last poll: RIC +{ric_score_delta} / STK +{stk_score_delta}, "
                f"margin {margin:+d}."
            )
    # First poll fallback - just orient the reader on the scoreboard.
    elif not is_break:
        sentences.append(f"Margin {margin:+d} - first read this block.")

    # -----------------------------------------------------------------------
    # 2. POSSESSION / PRESSURE DELTA - what CHANGED since last poll.
    # -----------------------------------------------------------------------
    if prev_state is not None:
        ric_disp_delta = ric["disposals"] - prev_state.get("ric_disposals", ric["disposals"])
        stk_disp_delta = stk["disposals"] - prev_state.get("stk_disposals", stk["disposals"])
        ric_tk_delta = ric["tackles"] - prev_state.get("ric_tackles", ric["tackles"])
        stk_tk_delta = stk["tackles"] - prev_state.get("stk_tackles", stk["tackles"])
        net_disp = ric_disp_delta - stk_disp_delta
        net_tk = ric_tk_delta - stk_tk_delta
        if abs(net_disp) >= 8:
            who = "Richmond" if net_disp > 0 else "St Kilda"
            sentences.append(
                f"{who} winning the last block's possession by {abs(net_disp)} "
                f"(RIC +{ric_disp_delta} / STK +{stk_disp_delta})."
            )
        elif abs(net_tk) >= 4:
            who = "Richmond" if net_tk > 0 else "St Kilda"
            sentences.append(
                f"{who} have lifted pressure this block - tackle split "
                f"+{ric_tk_delta}/+{stk_tk_delta} (RIC/STK)."
            )
    else:
        # No prev_state - give the cumulative snapshot framing.
        disp_gap = ric["disposals"] - stk["disposals"]
        if abs(disp_gap) > 10:
            who = "Richmond" if disp_gap > 0 else "St Kilda"
            sentences.append(
                f"{who} on top of the possession count {ric['disposals']}-{stk['disposals']}."
            )

    # -----------------------------------------------------------------------
    # 3. CURRENT-QUARTER AF LEADER - who is hot RIGHT NOW.
    # -----------------------------------------------------------------------
    q_key = _q_key_for(status_code)
    q_label = _q_label_short(status_code)
    ric_q_top = top_quarter_af(players, "RI", q_key, 1)
    stk_q_top = top_quarter_af(players, "SK", q_key, 1)
    q_bits: list[str] = []
    if ric_q_top and (ric_q_top[0].get(q_key) or 0) >= 15:
        p = ric_q_top[0]
        q_bits.append(
            f"{p.get('surname')} leading RIC in {q_label} ({p.get(q_key)} AF)"
        )
    if stk_q_top and (stk_q_top[0].get(q_key) or 0) >= 15:
        p = stk_q_top[0]
        q_bits.append(
            f"{p.get('surname')} top of STK in {q_label} ({p.get(q_key)} AF)"
        )
    if q_bits:
        sentences.append("; ".join(q_bits) + ".")

    # -----------------------------------------------------------------------
    # 4. RISING / FALLING PLAYERS - per-player deltas worth flagging.
    # -----------------------------------------------------------------------
    if prev_state is not None:
        prev_disp = prev_state.get("player_disp", {})
        prev_tk = prev_state.get("player_tk", {})
        rising: list[tuple[str, str, int, str]] = []  # (surname, team, delta, kind)
        for p in players:
            key = player_key(p)
            d_now = disp(p)
            t_now = p.get("tackles") or 0
            d_delta = d_now - prev_disp.get(key, d_now)
            t_delta = t_now - prev_tk.get(key, t_now)
            if d_delta >= 4:
                rising.append((p.get("surname", "?"), p.get("team", "?"), d_delta, f"+{d_delta} disp"))
            elif t_delta >= 3:
                rising.append((p.get("surname", "?"), p.get("team", "?"), t_delta, f"+{t_delta} tackles"))
        # Cap at the two biggest movers to keep the read concise.
        rising.sort(key=lambda r: -r[2])
        if rising:
            risers_str = ", ".join(
                f"{surname} ({team}, {kind})" for surname, team, _, kind in rising[:2]
            )
            sentences.append(f"Movers since last poll: {risers_str}.")

    # -----------------------------------------------------------------------
    # 5. GAME-STATE CLOSER - varies by band and status_code.
    # -----------------------------------------------------------------------
    if band == "final":
        if margin > 0:
            sentences.append(f"Full time: Richmond win by {margin}.")
        elif margin < 0:
            sentences.append(f"Full time: Richmond lose by {-margin}.")
        else:
            sentences.append("Full time: scores level.")
    elif is_break:
        sentences.append(
            f"End of {_q_label_short(status_code)} - margin {margin:+d}, reset and reassess."
        )
    elif band == "close":
        sentences.append("Contest live, margin inside a goal-and-a-bit - every chain matters.")
    elif band == "chasing":
        q_remaining = {"Q3": "still 1.5 quarters to make ground",
                       "Q4": "hunting goals in the final term",
                       "Q2": "long way to run but the gap is closing distance",
                       "Q1": "early - shape will tell more than scoreboard"}.get(status_code, "chasing")
        sentences.append(f"Richmond {abs_margin} down - {q_remaining}.")
    elif band == "leading_mid":
        sentences.append(f"Richmond +{margin} - protect the lead, don't chase risk.")
    elif band == "blowout_down":
        if status_code in {"Q4"}:
            sentences.append(
                f"Margin settled at {abs_margin} - Richmond unable to manufacture goals, "
                f"focus shifts to individual minutes."
            )
        else:
            sentences.append(
                f"Richmond {abs_margin} down - tripwire territory, structural change needed."
            )
    elif band == "blowout_up":
        sentences.append(f"Richmond +{margin} - margin established, manage the pressure profile.")

    return " ".join(sentences)


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
    prev_state: dict | None = None,
) -> str:
    """Thin wrapper kept for API compatibility - delegates to the dynamic read."""
    return _build_dynamic_read(
        home_code, home_full, away_code, away_full,
        ht, at, players, home_pts, away_pts, status_code, prev_state,
    )


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


def build_prev_state(
    players: list,
    ric_t: dict,
    stk_t: dict,
    ric_pts: int,
    stk_pts: int,
    status_code: str,
) -> dict:
    """Snapshot the state we need to compute deltas on the NEXT poll."""
    return {
        "ric_disposals": ric_t["disposals"],
        "stk_disposals": stk_t["disposals"],
        "ric_tackles":   ric_t["tackles"],
        "stk_tackles":   stk_t["tackles"],
        "ric_hitouts":   ric_t["hitouts"],
        "stk_hitouts":   stk_t["hitouts"],
        "ric_pts":       ric_pts,
        "stk_pts":       stk_pts,
        "status_code":   status_code,
        "player_disp":   {player_key(p): disp(p) for p in players},
        "player_tk":     {player_key(p): (p.get("tackles") or 0) for p in players},
    }


def format_analysis_block(
    snap: dict,
    status_code: str,
    trend_cache: TrendCache,
    prev_state: dict | None = None,
) -> tuple[str | None, dict]:
    """Build the markdown block for the current poll.

    Returns `(block, new_prev_state)`. If the feed has STALLED - identical
    score, disposals, tackles, and hitouts to the previous poll - returns
    `(None, prev_state)` so the caller can skip writing a duplicate block.
    See Scientist memory `live_pipeline_glitch.md` for the FanFooty-end-of-game
    polling bug that motivated this guard.
    """
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

    # -----------------------------------------------------------------------
    # Skip-if-unchanged guard. If the FanFooty feed has stalled (score,
    # disposals, tackles, AND hitouts all identical to the prior poll), the
    # block would be byte-for-byte the same useless paragraph as last cycle.
    # Refuse to write it. We RETURN the existing prev_state unchanged so the
    # next non-stalled poll still has the right baseline to compute deltas
    # against. We do NOT skip on a quarter-break status though - the break
    # block is allowed even if numbers are frozen, since the routing/heading
    # changes are themselves the new information.
    # -----------------------------------------------------------------------
    if prev_state is not None and status_code not in {"QT", "HT", "3QT", "FT"}:
        ric_pts_now = home_pts if home_pc == "RI" else away_pts
        stk_pts_now = away_pts if home_pc == "RI" else home_pts
        if (
            ric_pts_now == prev_state.get("ric_pts")
            and stk_pts_now == prev_state.get("stk_pts")
            and ric_t["disposals"] == prev_state.get("ric_disposals")
            and stk_t["disposals"] == prev_state.get("stk_disposals")
            and ric_t["tackles"] == prev_state.get("ric_tackles")
            and stk_t["tackles"] == prev_state.get("stk_tackles")
            and ric_t["hitouts"] == prev_state.get("ric_hitouts")
            and stk_t["hitouts"] == prev_state.get("stk_hitouts")
        ):
            return None, prev_state

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
    # Two parallel maps because the team-aggregated dict (team_totals) uses
    # `qN_af` keys while the per-player snapshot fields are `af_qN`.
    q_key_team = {"Q1": "q1_af", "QT": "q1_af", "Q2": "q2_af", "HT": "q2_af",
                  "Q3": "q3_af", "3QT": "q3_af", "Q4": "q4_af", "FT": "q4_af"}[status_code]
    q_key_player = _q_key_for(status_code)
    q_label = {"Q1": "Q1", "QT": "Q1 (end)", "Q2": "Q2", "HT": "Q2 (end)",
               "Q3": "Q3", "3QT": "Q3 (end)", "Q4": "Q4", "FT": "Q4 (final)"}[status_code]

    # Current-quarter AF leaders (who is hot RIGHT NOW, regardless of cumulative).
    ric_q_leaders = top_quarter_af(players, ric_pc, q_key_player, 3)
    stk_q_leaders = top_quarter_af(players, stk_pc, q_key_player, 3)

    def _fmt_q_leader(p: dict) -> str:
        return f"{p.get('first_name','')[:1]}. {p.get('surname','?')} {p.get(q_key_player) or 0}"

    read = generate_read(
        home_pc, home_full, away_pc, away_full, ht, at, players,
        home_pts, away_pts, status_code, prev_state,
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
        f"**{q_label} AF leaders - Richmond:** "
        + (" | ".join(_fmt_q_leader(p) for p in ric_q_leaders) if ric_q_leaders else "(no data)"),
        f"**{q_label} AF leaders - St Kilda:** "
        + (" | ".join(_fmt_q_leader(p) for p in stk_q_leaders) if stk_q_leaders else "(no data)"),
        "",
        "| Metric | RIC | STK |",
        "|--------|-----|-----|",
        f"| Disposals (K+HB) | {ric_t['disposals']} ({ric_t['kicks']}/{ric_t['handballs']}) | {stk_t['disposals']} ({stk_t['kicks']}/{stk_t['handballs']}) |",
        f"| Marks | {ric_t['marks']} | {stk_t['marks']} |",
        f"| Tackles | {ric_t['tackles']} | {stk_t['tackles']} |",
        f"| Hit-outs | {ric_t['hitouts']} | {stk_t['hitouts']} |",
        f"| Frees for | {ric_t['frees_for']} | {stk_t['frees_for']} |",
        f"| Total AF | {ric_t['total_af']} | {stk_t['total_af']} |",
        f"| {q_label} AF | {ric_t[q_key_team]} | {stk_t[q_key_team]} |",
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

    # Compute the prev_state to hand to the NEXT poll. Use Richmond points and
    # St Kilda points (not header home/away) so the read narrative stays
    # Richmond-centric regardless of which side is the nominal home team.
    if home_pc == "RI":
        ric_pts_for_state, stk_pts_for_state = home_pts, away_pts
    else:
        ric_pts_for_state, stk_pts_for_state = away_pts, home_pts
    new_prev = build_prev_state(
        players, ric_t, stk_t,
        ric_pts_for_state, stk_pts_for_state, status_code,
    )

    return "\n".join(lines), new_prev


def format_quarter_break(prev_code: str, snap: dict) -> str:
    """Compact summary written when status transitions across a break.

    Format target: ~10-15 lines. Beyond cumulative totals this block now
    includes top-3 quarter-AF leaders per side, the kick-share tripwire
    state, and a one-line score verdict that compares the just-completed
    quarter's margin to the pre-match target (Richmond must keep each
    quarter's deficit <=15 to stay in the contest).
    """
    h = snap["header"]
    players = snap["players"]
    ric_t = team_totals(players, "RI")
    stk_t = team_totals(players, "SK")
    q_team_key = {"Q1": "q1_af", "Q2": "q2_af",
                  "Q3": "q3_af", "Q4": "q4_af"}.get(prev_code, "q1_af")
    q_player_key = {"Q1": "af_q1", "Q2": "af_q2",
                    "Q3": "af_q3", "Q4": "af_q4"}.get(prev_code, "af_q1")

    home_full = h.get("home_team_full", "Home")
    away_full = h.get("away_team_full", "Away")
    home_pts = parse_score_pts(h.get("home_score", ""))
    away_pts = parse_score_pts(h.get("away_score", ""))
    if home_full == "Richmond":
        ric_pts, stk_pts = home_pts, away_pts
    else:
        ric_pts, stk_pts = away_pts, home_pts

    # Top-3 movers for the quarter just completed, per side.
    ric_q_top = top_quarter_af(players, "RI", q_player_key, 3)
    stk_q_top = top_quarter_af(players, "SK", q_player_key, 3)

    def _fmt(p: dict) -> str:
        return f"{p.get('first_name','')[:1]}. {p.get('surname','?')} {p.get(q_player_key) or 0}"

    ric_top_str = ", ".join(_fmt(p) for p in ric_q_top) if ric_q_top else "(no data)"
    stk_top_str = ", ".join(_fmt(p) for p in stk_q_top) if stk_q_top else "(no data)"

    # Kick-share tripwire (same proxy used in the live block).
    tripwire = (
        "HOLDS - Richmond level/ahead on kick-share proxy"
        if ric_t["kicks"] >= stk_t["kicks"]
        else "TRIGGERED - St Kilda winning the kick-share territory proxy"
    )

    # Score verdict: did Richmond keep this quarter within the 15-pt target?
    # Compute the per-quarter margin from cumulative scores if possible.
    # (We only have cumulative scores at the break, so this measures the
    # GAME margin at the break, not just the quarter. We name it accordingly.)
    margin = ric_pts - stk_pts
    if prev_code == "Q1":
        target_clause = "Richmond's pre-match plan needed <=15 down per quarter to stay live."
        if margin >= 0:
            verdict = f"Richmond +{margin} at QT - ahead of the brief, contest live."
        elif margin >= -15:
            verdict = f"Richmond {margin} at QT - inside the 15-pt target, plan holding."
        else:
            verdict = f"Richmond {margin} at QT - already outside the 15-pt target, structural change needed."
    elif prev_code == "Q2":
        target_clause = "Pre-match brief: stay within 25 by half time to keep a comeback live."
        if margin >= 0:
            verdict = f"Richmond +{margin} at HT - ahead of brief, second half is a contest."
        elif margin >= -25:
            verdict = f"Richmond {margin} at HT - inside the 25-pt half-time threshold."
        else:
            verdict = f"Richmond {margin} at HT - outside the 25-pt threshold, comeback unlikely."
    elif prev_code == "Q3":
        target_clause = "Pre-match brief: within 30 at 3QT to give Q4 a real chance."
        if margin >= 0:
            verdict = f"Richmond +{margin} at 3QT - leading into the final term."
        elif margin >= -30:
            verdict = f"Richmond {margin} at 3QT - inside the 30-pt window, Q4 alive."
        else:
            verdict = f"Richmond {margin} at 3QT - outside the 30-pt window, game effectively gone."
    else:  # Q4
        target_clause = "Final quarter complete."
        if margin > 0:
            verdict = f"Richmond win by {margin}."
        elif margin < 0:
            verdict = f"Richmond lose by {-margin}."
        else:
            verdict = "Scores level."

    return "\n".join([
        "",
        "---",
        f"### QUARTER BREAK: end of {prev_code} - {home_full} {h.get('home_score')} vs {away_full} {h.get('away_score')}",
        "",
        f"**Verdict:** {verdict}",
        f"*{target_clause}*",
        "",
        f"**{prev_code} AF leaders - Richmond:** {ric_top_str}",
        f"**{prev_code} AF leaders - St Kilda:** {stk_top_str}",
        "",
        f"**{prev_code} team AF:** RIC {ric_t[q_team_key]} - STK {stk_t[q_team_key]}",
        f"**Cumulative disposals:** RIC {ric_t['disposals']} - STK {stk_t['disposals']}",
        f"**Cumulative tackles:** RIC {ric_t['tackles']} - STK {stk_t['tackles']}",
        f"**Cumulative hit-outs:** RIC {ric_t['hitouts']} - STK {stk_t['hitouts']}",
        f"**Tripwire (kick-share proxy):** {tripwire} (RIC {ric_t['kicks']} - STK {stk_t['kicks']} kicks)",
        "",
        f"*Routing forward to the next quarter doc.*",
        "",
    ])


def format_quarter_break_analyst(prev_code: str, snap: dict) -> str:
    """ANALYST BLOCK appended to the INCOMING quarter doc at each transition.

    Distinct from `format_quarter_break`, which goes on the OUTGOING doc
    (closing out the quarter that just ended). This block opens the new
    quarter's document with a structured forward-looking read so coaching
    notes for Q3 / Q4 don't start on a blank page.

    Contents:
      * Score verdict vs the pre-match plan for the quarter just completed
      * Top-5 movers in that quarter (quarter-AF leaders across both teams)
      * Key-player tracking vs pre-match predictions (Short / Sinclair / Hill)
      * One forward-looking line for the upcoming quarter

    Triggered automatically at every quarter transition by the main loop.
    Equivalent to having a Scientist / FootyStrategy commentary block fire
    on the schedule without manual prompting.
    """
    h = snap["header"]
    players = snap["players"]
    ric_t = team_totals(players, "RI")
    stk_t = team_totals(players, "SK")
    q_player_key = {"Q1": "af_q1", "Q2": "af_q2",
                    "Q3": "af_q3", "Q4": "af_q4"}.get(prev_code, "af_q1")
    home_full = h.get("home_team_full", "Home")
    away_full = h.get("away_team_full", "Away")
    home_pts = parse_score_pts(h.get("home_score", ""))
    away_pts = parse_score_pts(h.get("away_score", ""))
    if home_full == "Richmond":
        ric_pts, stk_pts = home_pts, away_pts
    else:
        ric_pts, stk_pts = away_pts, home_pts
    margin = ric_pts - stk_pts

    # Score verdict vs pre-match targets (same target ladder as the closing
    # block, repeated here so each doc reads standalone).
    targets = {"Q1": 15, "Q2": 25, "Q3": 30, "Q4": 0}
    next_quarter = {"Q1": "Q2", "Q2": "Q3 (after main break)",
                    "Q3": "Q4", "Q4": "post-match"}.get(prev_code, "next")
    target = targets.get(prev_code, 15)
    if margin >= 0:
        verdict_line = (
            f"Richmond +{margin} after {prev_code} - above the brief; the "
            f"target was to be within {target} down."
        )
    elif abs(margin) <= target:
        verdict_line = (
            f"Richmond {margin} after {prev_code} - inside the {target}-pt "
            f"target, plan still alive."
        )
    else:
        verdict_line = (
            f"Richmond {margin} after {prev_code} - outside the {target}-pt "
            f"target; the {next_quarter} plan needs structural change, not just effort."
        )

    # Top-5 movers across BOTH teams in the quarter just completed.
    both = [(p, p.get(q_player_key) or 0) for p in players]
    both.sort(key=lambda r: -r[1])
    movers = both[:5]
    if movers:
        mover_lines = [
            f"- {p.get('first_name','')[:1]}. {p.get('surname','?')} "
            f"({p.get('team','?')}) {af} AF"
            for p, af in movers if af > 0
        ]
    else:
        mover_lines = ["- (no quarter-AF data available)"]

    # Pre-match matchup tracking (Short, Sinclair, Hill).
    matchup_lines: list[str] = []
    for surname, (team_code, pred, role) in KEY_PLAYERS.items():
        p = find_player(players, surname, team_code)
        if p is None:
            matchup_lines.append(f"- {surname} ({role}): not on field / no data")
            continue
        d = disp(p)
        if pred <= 0:
            ratio_note = ""
        elif d >= pred:
            ratio_note = f" - AT/ABOVE pred {pred}"
        elif d >= 0.7 * pred:
            ratio_note = f" - tracking (pred {pred})"
        elif d >= 0.4 * pred:
            ratio_note = f" - behind rate (pred {pred})"
        else:
            ratio_note = f" - WELL below pred {pred}"
        matchup_lines.append(f"- {surname} ({role}): {d} disp{ratio_note}")

    # Forward-looking line for the next quarter.
    forward = {
        "Q1": "Q2 focus: can Richmond's midfield steady the clearance count and avoid the 15-pt blowout swing?",
        "Q2": "Q3 focus: who lifts after the long break? Watch Sinclair's intercept count and Richmond's I50 entries.",
        "Q3": "Q4 focus: Richmond needs scoring chains, not just disposals. Watch goal-source diversity.",
        "Q4": "Post-match: write the verdict doc; review which pre-match calls held and which missed.",
    }.get(prev_code, "Next: continue the live read on the new quarter's doc.")

    return "\n".join([
        "",
        "---",
        f"### ANALYST BLOCK: opening read for the doc that follows {prev_code}",
        "",
        f"**Score verdict:** {verdict_line}",
        "",
        f"**Top 5 movers in {prev_code} (both sides, by quarter AF):**",
        *mover_lines,
        "",
        "**Pre-match matchup tracking:**",
        *matchup_lines,
        "",
        f"**Forward-looking:** {forward}",
        "",
        f"*Auto-generated at the {prev_code}->next transition. "
        f"Snapshot: game {snap.get('gameid','?')}, {snap.get('fetched_at_utc','?')}.*",
        "",
    ])


def _write_quarter_break_analysis(snap: dict, prev_code: str, doc_path: Path) -> None:
    """Public hook: insert the ANALYST BLOCK into `doc_path`.

    Wraps `format_quarter_break_analyst` and `insert_block` so the main loop
    can trigger structured commentary at every quarter transition without
    re-implementing the doc-write contract. Idempotent w.r.t. the file's
    header: `ensure_header` is called first if the doc is empty.
    """
    ensure_header(doc_path, snap, classify_status(snap.get("header", {}).get("status", "")) or "Q1")
    block = format_quarter_break_analyst(prev_code, snap)
    insert_block(doc_path, block)


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


def _find_header_end(text: str) -> int:
    """Locate the character offset where the document header/intro ends.

    Strategy (in order):
    1. If an explicit `<!-- LIVE UPDATES BELOW -->` marker exists, use the
       position immediately after it.
    2. Else, find the end of the first contiguous header block: the H1 title
       followed by any non-section lines (back-link, intro paragraph) and stop
       at the first blank line that precedes a content section. Concretely:
       find the H1 title, then advance past any lines that are not new section
       markers (`## `, `### `, or `---`) until the first such marker, and
       insert immediately before it.
    3. Fall back to end-of-file (prepend nothing useful possible).
    """
    pre_marker = "<!-- LIVE UPDATES BELOW -->"
    idx = text.find(pre_marker)
    if idx != -1:
        end = idx + len(pre_marker)
        # Skip a single trailing newline if present so blocks start cleanly.
        if end < len(text) and text[end] == "\n":
            end += 1
        return end

    lines = text.splitlines(keepends=True)
    # Find the first H1 (title). Default insertion point is after it.
    title_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("# ")), None
    )
    if title_idx is None:
        # No title at all - insert at very top.
        return 0

    # From after the title, walk forward; the header ends at the first line
    # that opens a content section: `## `, `### `, or a horizontal rule `---`.
    i = title_idx + 1
    while i < len(lines):
        stripped = lines[i].lstrip()
        if (
            stripped.startswith("## ")
            or stripped.startswith("### ")
            or stripped.rstrip() == "---"
        ):
            break
        i += 1
    return sum(len(ln) for ln in lines[:i])


def insert_block(path: Path, block: str) -> None:
    """Insert the newest analysis block at the TOP of the document.

    Layout produced:

        # Title
        > back-links / intro paragraph(s)

        ## Auto-updated live analysis (newest first)

        <!-- LIVE_ANALYSIS_AUTO_BLOCKS_BELOW -->

        <newest block>
        <older block>
        ...

        <pre-existing hand-authored sections>

    If the auto-blocks section doesn't yet exist (e.g. the doc was hand-
    authored before the pipeline started), we splice it in right after the
    header/intro block, BEFORE all existing content sections - so the newest
    auto-block always sits near the top of the file.
    """
    if not path.exists():
        path.write_text(block)
        return
    existing = path.read_text()
    if AUTO_MARKER not in existing:
        header_end = _find_header_end(existing)
        head = existing[:header_end].rstrip() + "\n\n"
        tail = existing[header_end:].lstrip("\n")
        auto_section = (
            "## Auto-updated live analysis (newest first)\n\n"
            + AUTO_MARKER + "\n\n"
        )
        # If there is hand-authored content following, put a horizontal rule
        # between the auto-blocks region and that content, so the boundary
        # reads cleanly.
        if tail.strip():
            existing = head + auto_section + "\n---\n\n" + tail
        else:
            existing = head + auto_section
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
    # prev_state holds the previous poll's team + per-player stats so the
    # Read paragraph can describe what CHANGED, not just the current snapshot.
    # None on the first poll - generate_read handles that gracefully.
    prev_state: dict | None = None
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
        if status_code is None:
            # Unrecognised status - do NOT silently route to Q1 (that was the
            # R11 Final Siren routing bug). Skip this cycle entirely; if the
            # next poll still can't be classified, the user will see it in
            # the log and can intervene rather than getting wrong-quarter docs.
            print(
                f"  [warn] unrecognised status string '{status_raw}' - "
                f"skipping write this cycle (last good status_code={last_status_code})",
                flush=True,
            )
            time.sleep(POLL_SECONDS)
            continue

        print(
            f"  status='{status_raw}' code={status_code} "
            f"score: {h.get('home_team_full')} {h.get('home_score')} - "
            f"{h.get('away_team_full')} {h.get('away_score')}",
            flush=True,
        )

        # On quarter transition: write a QUARTER BREAK summary (closing read)
        # to the OUTGOING doc, then an ANALYST BLOCK (opening read) to the
        # INCOMING doc. Triggered for EVERY transition out of a live quarter,
        # so Q3 and Q4 docs always start with structured forward-looking
        # commentary rather than a blank page.
        paths_touched: list[Path] = []
        is_transition_out_of_live = (
            last_status_code is not None
            and last_status_code != status_code
            and last_status_code in {"Q1", "Q2", "Q3", "Q4"}
            and status_code in {"QT", "HT", "3QT", "FT", "Q2", "Q3", "Q4"}
        )
        if is_transition_out_of_live:
            old_path = doc_path_for(last_status_code)
            ensure_header(old_path, snap, last_status_code)
            insert_block(old_path, format_quarter_break(last_status_code, snap))
            paths_touched.append(old_path)
            print(f"  [break written] end of {last_status_code} -> {old_path.name}", flush=True)

            # ANALYST BLOCK on the incoming doc. We use the destination doc
            # for status_code (the new live quarter or the break doc), so
            # Q2/Q3/Q4/FT docs each open with the structured forward read.
            incoming_path = doc_path_for(status_code)
            try:
                _write_quarter_break_analysis(snap, last_status_code, incoming_path)
                if incoming_path not in paths_touched:
                    paths_touched.append(incoming_path)
                print(
                    f"  [analyst block] {last_status_code}->{status_code} "
                    f"appended to {incoming_path.name}",
                    flush=True,
                )
            except Exception as e:  # pragma: no cover - defensive: never let
                # the analyst block crash the main poll loop.
                print(f"  [warn] analyst block failed: {e}", flush=True)

        # Write the live analysis block to the current quarter's doc.
        # On a quarter transition we reset prev_state so deltas don't
        # straddle break boundaries (a 14-minute gap would otherwise be
        # described as a "block" with massive deltas).
        effective_prev = prev_state
        if (
            last_status_code is not None
            and last_status_code != status_code
            and last_status_code in {"Q1", "Q2", "Q3"}
            and status_code in {"QT", "HT", "3QT", "Q2", "Q3", "Q4"}
        ):
            effective_prev = None

        current_path = doc_path_for(status_code)
        ensure_header(current_path, snap, status_code)
        block, prev_state = format_analysis_block(
            snap, status_code, trend_cache, effective_prev,
        )
        if block is None:
            # FanFooty feed stalled - identical numbers to the prior poll.
            # Don't write a duplicate block, don't commit a no-op. Sleep and
            # try again. (Don't reset prev_state - we still want to compare
            # against the same baseline next cycle.)
            print(
                f"  [skip] feed stalled - score/disposals/tackles/hitouts "
                f"identical to prior poll; no block written",
                flush=True,
            )
        else:
            insert_block(current_path, block)
            paths_touched.append(current_path)
            print(f"  [block written] {status_code} -> {current_path.name}", flush=True)

        if paths_touched:
            git_commit_push(paths_touched, f"{status_raw} -> {current_path.name}")
        last_status_code = status_code

        if status_code == "FT":
            print("\nFull Time reached. Final block written. Exiting.", flush=True)
            break

        time.sleep(POLL_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
