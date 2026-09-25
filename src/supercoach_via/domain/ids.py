"""Identity registry: players, explicit aliases, clubs (entity vs lineage) and venues.

Rules (PLAN section 4.1):
- A legacy player ID is ``legacy:<exact slug incl. DOB token>``; display names are never keys.
- Duplicate/alias decisions are explicit registry entries with evidence, never fuzzy merges.
- A club alias is valid only inside its season interval; historical entities stay distinct
  and share a ``lineage_id`` only for documented renames/relocations.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

from supercoach_via.domain.schemas import BirthDateQuality, is_safe_id

LEGACY_PREFIX = "legacy:"
PERFORMANCE_SUFFIX = "_performance_details.csv"
PERSONAL_SUFFIX = "_personal_details.csv"
_DOB_TOKEN = re.compile(r"^\d{8}$")
#: The legacy parser substituted this DOB when parsing failed (AUDIT C08). Not a birth date.
LEGACY_DEFAULT_DOB = date(1900, 1, 1)


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateIdentity:
    duplicate_slug: str
    canonical_slug: str
    reason: str
    source_url: str


#: Verified by the 2026-09-24 source reconciliation (docs/rewrite/DATA_REFRESH.md,
#: evidence/refresh-sources.json ``legacy_duplicates_preserved``).
KNOWN_DUPLICATES: tuple[DuplicateIdentity, ...] = (
    DuplicateIdentity(
        "green_william_08092005",
        "green_will_08092005",
        "William/Will spelling variant with identical source DOB and overlapping game",
        "https://afltables.com/afl/stats/players/W/Will_Green.html",
    ),
    DuplicateIdentity(
        "steele_roan_19092002",
        "steele_roan_22102001",
        "Older filename/DOB differs from current source; both game rows overlap the source-confirmed career",
        "https://afltables.com/afl/stats/players/R/Roan_Steele.html",
    ),
)


@dataclass(frozen=True)
class SourceAlias:
    slug: str
    source_name: str
    source_url: str


#: Multi-part source surnames verified against afltables (refresh-sources.json
#: ``verified_source_name``). The legacy slug/personal-details split the surname.
VERIFIED_SOURCE_ALIASES: tuple[SourceAlias, ...] = (
    SourceAlias(
        "wyk_alex_01072004",
        "Alex Van Wyk",
        "https://afltables.com/afl/stats/players/A/Alex_Van_Wyk.html",
    ),
    SourceAlias(
        "chee_callum_09101997",
        "Callum Ah Chee",
        "https://afltables.com/afl/stats/players/C/Callum_Ah_Chee.html",
    ),
    SourceAlias(
        "achkar_hussien_02042007",
        "Hussien El Achkar",
        "https://afltables.com/afl/stats/players/H/Hussien_El_Achkar.html",
    ),
    SourceAlias(
        "rooyen_jacob_16042003",
        "Jacob van Rooyen",
        "https://afltables.com/afl/stats/players/J/Jacob_van_Rooyen.html",
    ),
    SourceAlias(
        "goey_jordan_15031996",
        "Jordan de Goey",
        "https://afltables.com/afl/stats/players/J/Jordan_de_Goey.html",
    ),
    SourceAlias(
        "koning_sam_26022001",
        "Sam De Koning",
        "https://afltables.com/afl/stats/players/S/Sam_De_Koning.html",
    ),
    SourceAlias(
        "koning_tom_16071999",
        "Tom De Koning",
        "https://afltables.com/afl/stats/players/T/Tom_De_Koning.html",
    ),
)


def slug_from_filename(name: str) -> str:
    if not name.endswith(PERFORMANCE_SUFFIX):
        raise ValueError(f"not a performance-details filename: {name!r}")
    return name[: -len(PERFORMANCE_SUFFIX)]


def player_id_for_slug(slug: str) -> str:
    pid = LEGACY_PREFIX + slug
    if not slug or not is_safe_id(pid):
        raise ValueError(f"unsafe legacy slug {slug!r}")
    return pid


def dob_token(slug: str) -> str | None:
    token = slug.rsplit("_", 1)[-1]
    return token if _DOB_TOKEN.match(token) else None


def _parse_ddmmyyyy(value: str) -> date | None:
    m = re.fullmatch(r"(\d{2})-?(\d{2})-?(\d{4})", value.strip())
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def resolve_birth_date(born_date: str | None, slug: str) -> tuple[date | None, BirthDateQuality, str | None]:
    """Birth date, its quality and an optional quality-issue rule id.

    Personal-details ``born_date`` (DD-MM-YYYY) is the source; the slug DOB token is the
    fallback (``legacy_filename``). The legacy default 01-01-1900 is never a birth date.
    """
    token = dob_token(slug)
    token_date = _parse_ddmmyyyy(token) if token else None
    source_date = _parse_ddmmyyyy(born_date) if born_date else None
    if source_date == LEGACY_DEFAULT_DOB:
        return None, BirthDateQuality.UNKNOWN, "legacy_default_birth_date"
    if source_date is not None:
        if token_date is not None and token_date != source_date:
            return source_date, BirthDateQuality.CONFLICTING, "birth_date_conflicts_filename"
        return source_date, BirthDateQuality.SOURCE, None
    if token_date is not None and token_date != LEGACY_DEFAULT_DOB:
        return token_date, BirthDateQuality.LEGACY_FILENAME, "birth_date_from_filename"
    return None, BirthDateQuality.UNKNOWN, "birth_date_unknown"


def parse_measure(raw: str | None) -> float | None:
    """Height/weight: the legacy files use 0 and -1 as 'unknown' sentinels -> null."""
    try:
        value = float(raw) if raw not in (None, "") else None
    except ValueError:
        return None
    return value if value is not None and value > 0 else None


def normalize_name(name: str) -> str:
    """Case/accent/punctuation-insensitive comparison key (never an identity by itself)."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_ = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_ = ascii_.casefold().replace("'", "").replace("’", "").replace(".", "")
    ascii_ = re.sub(r"[^a-z0-9]+", " ", ascii_)
    return " ".join(ascii_.split())


def slugify(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", normalize_name(name))).strip("_")


# ---------------------------------------------------------------------------
# Clubs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClubAlias:
    alias: str
    club_id: str
    club_name: str
    lineage_id: str
    is_primary: bool
    valid_from: int
    valid_to: int | None
    note: str

    def valid(self, season: int) -> bool:
        return self.valid_from <= season and (self.valid_to is None or season <= self.valid_to)


def _looks_like_club_name(name: str) -> bool:
    stripped = name.strip()
    return bool(stripped) and bool(re.search(r"[A-Za-z]", stripped)) and not re.search(r"\d", stripped)


@dataclass
class ClubRegistry:
    aliases: list[ClubAlias]
    created: dict[str, str] = field(default_factory=dict)  # auto-created club_id -> source name

    @classmethod
    def from_csv(cls, path: Path) -> ClubRegistry:
        rows: list[ClubAlias] = []
        with path.open(newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                alias = ClubAlias(
                    alias=rec["alias"],
                    club_id=rec["club_id"],
                    club_name=rec["club_name"],
                    lineage_id=rec["lineage_id"],
                    is_primary=rec["is_primary"].strip().lower() == "true",
                    valid_from=int(rec["valid_from_season"]),
                    valid_to=int(rec["valid_to_season"]) if rec["valid_to_season"].strip() else None,
                    note=rec.get("note") or "",
                )
                if not is_safe_id(alias.club_id) or not is_safe_id(alias.lineage_id):
                    raise ValueError(f"unsafe club/lineage id in {path}: {alias}")
                rows.append(alias)
        seen: set[tuple[str, int]] = set()
        for a in rows:
            if (a.alias, a.valid_from) in seen:
                raise ValueError(f"duplicate alias interval {a.alias!r} from {a.valid_from}")
            seen.add((a.alias, a.valid_from))
        return cls(rows)

    def __post_init__(self) -> None:
        self._by_alias: dict[str, list[ClubAlias]] = {}
        for a in self.aliases:
            self._by_alias.setdefault(a.alias, []).append(a)
        self._cache: dict[tuple[str, int], str | None] = {}

    def resolve(self, name: str, season: int) -> str | None:
        key = (name, season)
        if key in self._cache:
            return self._cache[key]
        hits = {a.club_id for a in self._by_alias.get(name, ()) if a.valid(season)}
        result = next(iter(hits)) if len(hits) == 1 else None
        if result is None and name in self.created.values():
            result = next(k for k, v in self.created.items() if v == name)
        self._cache[key] = result
        return result

    def is_known_name(self, name: str) -> bool:
        return name in self._by_alias

    def resolve_or_create(self, name: str, season: int) -> str | None:
        """Resolve; an entirely unknown (well-formed) name becomes its own entity + lineage."""
        found = self.resolve(name, season)
        if found is not None or self.is_known_name(name) or not _looks_like_club_name(name):
            return found
        base = slugify(name)
        existing = {a.club_id for a in self.aliases} | set(self.created)
        cid = base if base not in existing else f"auto_{base}"
        if not is_safe_id(cid):
            return None
        self.created[cid] = name
        self._cache = {k: v for k, v in self._cache.items() if k[0] != name}
        return cid

    def lineage(self, club_id: str) -> str:
        for a in self.aliases:
            if a.club_id == club_id:
                return a.lineage_id
        if club_id in self.created:
            return club_id
        raise KeyError(club_id)

    def club_rows(self) -> list[dict[str, object]]:
        names: dict[str, tuple[str, str]] = {}
        spans: dict[str, list[tuple[int, int | None]]] = {}
        for a in self.aliases:
            names.setdefault(a.club_id, (a.club_name, a.lineage_id))
            spans.setdefault(a.club_id, [])
            if a.is_primary:
                spans[a.club_id].append((a.valid_from, a.valid_to))
        out: list[dict[str, object]] = []
        for cid, (name, lineage) in names.items():
            primary = spans[cid]
            tos = [t for _, t in primary]
            last = None if (not tos or None in tos) else max(t for t in tos if t is not None)
            first = min(f for f, _ in primary) if primary else None
            out.append(
                {
                    "club_id": cid,
                    "name": name,
                    "lineage_id": lineage,
                    "first_season": first,
                    "last_season": last,
                    "active": last is None,
                }
            )
        for cid, name in self.created.items():
            out.append(
                {
                    "club_id": cid,
                    "name": name,
                    "lineage_id": cid,
                    "first_season": None,
                    "last_season": None,
                    "active": True,
                }
            )
        return sorted(out, key=lambda r: str(r["club_id"]))

    def alias_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = [
            {
                "alias": a.alias,
                "club_id": a.club_id,
                "valid_from_season": a.valid_from,
                "valid_to_season": a.valid_to,
                "note": a.note or None,
            }
            for a in self.aliases
        ]
        rows += [
            {
                "alias": name,
                "club_id": cid,
                "valid_from_season": 0,
                "valid_to_season": None,
                "note": "auto-created: name absent from config/team_aliases.csv",
            }
            for cid, name in self.created.items()
        ]
        return sorted(rows, key=lambda r: (str(r["alias"]), str(r["valid_from_season"]).zfill(4)))


# ---------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------


@dataclass
class VenueRegistry:
    by_source: dict[str, tuple[str, str, str | None]]  # source name -> (venue_id, name, tz)

    @classmethod
    def from_csv(cls, path: Path) -> VenueRegistry:
        out: dict[str, tuple[str, str, str | None]] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                tz = rec["timezone"].strip() or None
                if tz is not None:
                    try:
                        ZoneInfo(tz)
                    except Exception as exc:  # ZoneInfoNotFoundError / ValueError
                        raise ValueError(f"unknown timezone {tz!r} in {path}") from exc
                if not is_safe_id(rec["venue_id"]):
                    raise ValueError(f"unsafe venue id {rec['venue_id']!r}")
                out[rec["source_name"]] = (rec["venue_id"], rec["venue_name"], tz)
        return cls(out)

    def resolve(self, source_name: str) -> str | None:
        hit = self.by_source.get(source_name)
        return hit[0] if hit else None

    def venue_rows(self, observed: Mapping[str, Iterable[str]]) -> list[dict[str, object]]:
        """Rows for venues actually observed: venue_id -> source names seen."""
        meta = {vid: (name, tz) for vid, name, tz in self.by_source.values()}
        rows: list[dict[str, object]] = []
        for vid in sorted(observed):
            name, tz = meta[vid]
            rows.append(
                {
                    "venue_id": vid,
                    "name": name,
                    "source_names": json.dumps(sorted(set(observed[vid]))),
                    "timezone": tz,
                }
            )
        return rows
