"""FanFooty live text feed: schema/version, reliable fields and sentries (PLAN 10).

Pure parser; no I/O. Facts exposed per player are limited to the reviewed
``reliable_fields`` in ``config/fanfooty_schema.yaml``. Known-misindexed per-player
goals/behinds/clangers, fields the feed does not carry, and every unnamed ``colNN``
column are *never* facts: they are listed in ``unavailable_fields`` and dropped.

Sentries: exact column count (65), per-player AF quarter sum == AF total, scoreboard
``g.b.total`` arithmetic, a recognised match status. Any failure is ``FAIL`` and the
payload must be preserved as an anomaly, not promoted.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from supercoach_via.domain.schemas import CheckOutcome

ADAPTER_VERSION = "1"
_GAME_ID_RE = re.compile(r"^[0-9]{1,8}$")

#: Positional column order decoded by the legacy fetcher (scripts/fetch_live_match.py).
COLUMN_ORDER: tuple[str, ...] = (
    "player_id", "first_name", "surname", "team", "col4_unknown", "af", "sc", "proj_af",
    "proj_low", "proj_sc", "kicks", "handballs", "marks", "tackles", "hitouts", "goals",
    "behinds", "frees_for", "frees_against", "status", "tag_primary", "blurb_template",
    "tag_secondary", "matchup_note", "col24_unknown", "col25_unknown", "col26_unknown",
    "col27_unknown", "position", "jumper", "col30_unknown", "col31_unknown", "col32_unknown",
    "col33_pct", "col34_pct", "col35_pct", "col36_unknown", "col37_unknown", "col38_unknown",
    "clangers", "col40_unknown", "col41_unknown", "de_pct", "tog_pct", "col44_unknown",
    "col45_unknown", "af_q1", "sc_q1", "af_q2", "sc_q2", "af_q3", "sc_q3", "af_q4", "sc_q4",
    "col54_unknown", "col55_unknown", "col56_unknown", "col57_unknown", "col58_unknown",
    "col59_unknown", "col60_unknown", "col61_unknown", "col62_unknown", "col63_unknown",
    "col64_unknown",
)  # fmt: skip
TEXT_FIELDS = frozenset({"status", "position", "team", "first_name", "surname", "player_id"})


def feed_url(game_id: str) -> str:
    if not _GAME_ID_RE.fullmatch(game_id):
        raise ValueError(f"invalid FanFooty game id {game_id!r}")
    return f"https://www.fanfooty.com.au/live/{game_id}.txt"


@dataclass(frozen=True)
class FanFootySchema:
    expected_columns: int
    column_order: tuple[str, ...]
    reliable: frozenset[str]
    unreliable: dict[str, int]
    unavailable: frozenset[str]
    version: str


def load_schema(path: Path) -> FanFootySchema:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    reliable = frozenset(str(x) for x in raw.get("reliable_fields", []))
    unreliable = {str(k): int(v["snapshot_index"]) for k, v in (raw.get("unreliable_fields") or {}).items()}
    unavailable = frozenset(str(x) for x in raw.get("unavailable_fields", []))
    if not reliable or reliable & set(unreliable) or not reliable <= set(COLUMN_ORDER):
        raise ValueError("fanfooty schema: reliable fields must be known, non-empty and disjoint")
    for name, idx in unreliable.items():
        if COLUMN_ORDER[idx] != name:
            raise ValueError(f"fanfooty schema: {name} is not at index {idx}")
    digest = hashlib.sha256(
        repr((COLUMN_ORDER, sorted(reliable), sorted(unreliable.items()), sorted(unavailable))).encode()
    ).hexdigest()[:12]
    return FanFootySchema(
        expected_columns=len(COLUMN_ORDER),
        column_order=COLUMN_ORDER,
        reliable=reliable,
        unreliable=unreliable,
        unavailable=unavailable,
        version=f"fanfooty-{len(COLUMN_ORDER)}col-{digest}",
    )


# ---------------------------------------------------------------------------
# Match phase
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    index: int  # monotonic order: Q1=1, QT=2, Q2=3, HT=4, Q3=5, 3QT=6, Q4=7, final=8
    label: str
    quarter: int | None
    is_break: bool
    final: bool
    clock: str | None = None

    @property
    def completed_quarters(self) -> int:
        if self.final:
            return 4
        if self.is_break:
            return {2: 1, 4: 2, 6: 3}[self.index]
        return (self.quarter or 1) - 1


_BREAKS = {"qtr time": (2, "QT"), "half time": (4, "HT"), "3 qtr time": (6, "3QT"), "three qtr time": (6, "3QT")}


def phase_of(status: str) -> Phase | None:
    s = " ".join(status.split()).strip()
    low = s.lower()
    m = re.fullmatch(r"q([1-4])(?:\s+(\d{1,2}:\d{2}))?", low)
    if m:
        q = int(m.group(1))
        return Phase(2 * q - 1, f"Q{q}", q, False, False, m.group(2))
    if low in _BREAKS:
        idx, label = _BREAKS[low]
        return Phase(idx, label, None, True, False)
    if low in ("full time", "final siren"):
        return Phase(8, "final", None, False, True)
    return None


# ---------------------------------------------------------------------------
# Feed parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedHeader:
    home_name: str
    away_name: str
    round_label: str
    status: str
    home_goals: int | None
    home_behinds: int | None
    home_score: int | None
    away_goals: int | None
    away_behinds: int | None
    away_score: int | None


@dataclass(frozen=True)
class FeedPlayer:
    source_player_id: str
    name: str
    team_code: str
    reliable: dict[str, int | float | str | None]


@dataclass
class FeedParse:
    outcome: CheckOutcome
    schema_version: str
    payload_sha256: str
    header: FeedHeader | None = None
    phase: Phase = field(default_factory=lambda: Phase(0, "unknown", None, False, False))
    players: list[FeedPlayer] = field(default_factory=list)
    commentary: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    unavailable_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def completed_quarters(self) -> int:
        return self.phase.completed_quarters if self.phase.index else 0


_SAFE_TEXT_RE = re.compile(r"[^A-Za-z0-9 .,'\-:;!?()%/&]")
_INJECTION_RE = re.compile(r"\[(SYSTEM|INST)[^\]]*\]?", re.IGNORECASE)


def sanitize_text(text: str, limit: int = 200) -> str:
    """Untrusted feed text -> inert plain text (no markup, no directive markers)."""
    t = re.sub(r"<[^>]*>", "", html.unescape(text))
    t = _INJECTION_RE.sub("", t)
    t = _SAFE_TEXT_RE.sub("", t)
    return " ".join(t.split())[:limit]


def _scoreboard(text: str) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text.strip())
    if not m:
        return None
    g, b, t = (int(x) for x in m.groups())
    return (g, b, t) if g * 6 + b == t else None


def _num(v: str) -> int | float | None:
    v = v.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return None


def parse_feed(content: bytes, schema: FanFootySchema) -> FeedParse:
    sha = hashlib.sha256(content).hexdigest()
    out = FeedParse(outcome=CheckOutcome.FAIL, schema_version=schema.version, payload_sha256=sha)
    out.unavailable_fields = sorted(
        set(schema.unreliable) | set(schema.unavailable) | {c for c in schema.column_order if c.startswith("col")}
    )
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 5:
        out.anomalies.append("feed shorter than header+meta+commentary+chat+players")
        return out
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 8:
        out.anomalies.append(f"header has {len(parts)} fields, expected >= 8")
        return out
    home_sb, away_sb = _scoreboard(parts[5]), _scoreboard(parts[6])
    if home_sb is None or away_sb is None:
        out.anomalies.append("scoreboard is not a consistent goals.behinds.total")
    phase = phase_of(parts[7])
    if phase is None:
        out.anomalies.append(f"unrecognised match status {parts[7][:30]!r}")
    else:
        out.phase = phase
    out.header = FeedHeader(
        home_name=sanitize_text(parts[0], 60),
        away_name=sanitize_text(parts[2], 60),
        round_label=sanitize_text(parts[4], 20),
        status=sanitize_text(parts[7], 30),
        home_goals=home_sb[0] if home_sb else None,
        home_behinds=home_sb[1] if home_sb else None,
        home_score=home_sb[2] if home_sb else None,
        away_goals=away_sb[0] if away_sb else None,
        away_behinds=away_sb[1] if away_sb else None,
        away_score=away_sb[2] if away_sb else None,
    )
    for chunk in re.split(r"<br\s*/?>", lines[2].lstrip("#")):
        cleaned = sanitize_text(re.sub(r"^\s*m0nty:\s*", "", html.unescape(chunk)))
        if cleaned:
            out.commentary.append(cleaned)

    goals_by_team: dict[str, int] = {}
    for n, line in enumerate(lines[4:], start=5):
        if not line.strip():
            continue
        cols = line.split(",")
        if len(cols) != schema.expected_columns:
            out.anomalies.append(f"line {n}: {len(cols)} columns, expected {schema.expected_columns}")
            continue
        values = dict(zip(schema.column_order, cols, strict=True))
        af = _num(values["af"])
        quarters = [_num(values[f"af_q{i}"]) or 0 for i in range(1, 5)]
        if af is not None and af != sum(quarters):
            out.anomalies.append(f"line {n}: AF quarter sum {sum(quarters)} != total {af}")
        done = out.completed_quarters
        if phase is not None and not phase.final:
            live_q = done + (0 if phase.is_break else 1)
            if any(quarters[i] for i in range(live_q, 4)):
                out.anomalies.append(f"line {n}: values present for a quarter not yet played")
        reliable: dict[str, int | float | str | None] = {}
        for name in sorted(schema.reliable):
            raw = values[name]
            reliable[name] = sanitize_text(raw, 40) or None if name in TEXT_FIELDS else _num(raw)
        team = sanitize_text(values["team"], 4)
        g = _num(values["goals"])
        goals_by_team[team] = goals_by_team.get(team, 0) + (int(g) if isinstance(g, int) else 0)
        out.players.append(
            FeedPlayer(
                source_player_id=re.sub(r"[^0-9]", "", values["player_id"]),
                name=sanitize_text(f"{values['first_name']} {values['surname']}", 60),
                team_code=team,
                reliable=reliable,
            )
        )
    if not out.players:
        out.anomalies.append("no player rows")
    if home_sb and away_sb and sorted(goals_by_team.values()) != sorted((home_sb[0], away_sb[0])):
        out.notes.append("per-player goals do not reconcile to the scoreboard (known unreliable field)")
    out.outcome = CheckOutcome.FAIL if out.anomalies else CheckOutcome.PASS
    return out
