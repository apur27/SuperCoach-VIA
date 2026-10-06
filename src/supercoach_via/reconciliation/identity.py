"""Evidence-backed player identity and match alignment (DESIGN section 7).

The source profile URL is the player key. A local player maps to it by, in order: a verified
local source URL (checked against profile evidence), a versioned override with a captured
locator, a unique normalised name plus full date of birth corroborated by club/season
membership, a unique full DOB with an identical set of (season, club) memberships (names
differ), or, where a DOB is missing or low precision, a globally unique exact equality of
appearance sets. There is no similarity score and no URL is ever built from a name: anything
else stays unresolved and visible. Mapping uniqueness is enforced globally.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from supercoach_via.reconciliation.rules import IdentityOverride

_WS = re.compile(r"\s+")
ALIAS_STATUSES = frozenset({"alias", "duplicate", "quarantined_duplicate"})


def norm_name(name: str) -> str:
    """NFC, one apostrophe form, collapsed whitespace, case-folded. Hyphens and spaces are kept, so
    distinct names stay distinct."""
    text = unicodedata.normalize("NFC", name).replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    return _WS.sub(" ", text).strip().casefold()


def full_dob(born: str | None) -> str | None:
    """A date of birth usable as a key: present and not a 1 January placeholder."""
    if born is None or born.endswith("-01-01"):
        return None
    return born


@dataclass(frozen=True)
class SourceProfile:
    url: str
    name: str | None
    born: str | None
    appearances: frozenset[tuple[str, str]]  # (source match url, club)
    memberships: frozenset[tuple[int, str]]  # (season, club)
    in_census: bool


@dataclass(frozen=True)
class LocalIdentityRec:
    layer: str
    key: str
    name: str
    born: str | None
    source_urls: tuple[str, ...]
    appearances: frozenset[tuple[str, str]]
    memberships: frozenset[tuple[int, str]]
    identity_status: str
    canonical_id: str | None


@dataclass(frozen=True)
class Resolution:
    key: str
    status: str  # resolved | alias | ambiguous | unresolved | conflict
    url: str | None
    rule: str | None
    candidates: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class IdFinding:
    category: str
    rule_id: str
    severity: str  # unknown | info
    keys: tuple[str, ...]
    detail: str


@dataclass
class IdentityResult:
    resolutions: dict[str, Resolution] = field(default_factory=dict)
    profile_to_locals: dict[str, list[str]] = field(default_factory=dict)
    owner: dict[str, str] = field(default_factory=dict)
    findings: list[IdFinding] = field(default_factory=list)
    source_unmatched: list[str] = field(default_factory=list)


def _canonical_chain(rec: LocalIdentityRec, by_key: dict[str, LocalIdentityRec]) -> tuple[str | None, str]:
    """The canonical owner of an alias, or ``(None, reason)`` for a loop or a dangling reference."""
    seen = [rec.key]
    cur = rec
    for _ in range(20):
        nxt = cur.canonical_id
        if nxt is None:
            return None, f"{cur.key} is an alias with no canonical id"
        if nxt in seen:
            return None, f"alias loop: {' -> '.join([*seen, nxt])}"
        target = by_key.get(nxt)
        if target is None:
            return None, f"canonical id {nxt} does not exist"
        if target.identity_status not in ALIAS_STATUSES:
            return nxt, ""
        seen.append(nxt)
        cur = target
    return None, "alias chain too long"


def resolve_identities(
    profiles: dict[str, SourceProfile], locals_: list[LocalIdentityRec], overrides: tuple[IdentityOverride, ...]
) -> IdentityResult:
    out = IdentityResult()
    by_key = {r.key: r for r in locals_}
    ov = {o.local_player_id: o for o in overrides}
    canon: list[LocalIdentityRec] = []
    alias_of: dict[str, str] = {}
    for rec in sorted(locals_, key=lambda r: r.key):
        if rec.identity_status in ALIAS_STATUSES:
            owner, why = _canonical_chain(rec, by_key)
            if owner is None:
                out.resolutions[rec.key] = Resolution(rec.key, "conflict", None, None, (), why)
                out.findings.append(IdFinding("IDENTITY_CONFLICT", "ID-ALIAS", "unknown", (rec.key,), why))
            else:
                alias_of[rec.key] = owner
                out.owner[rec.key] = owner
        else:
            canon.append(rec)
    # an alias contributes its appearances and memberships to its canonical player
    merged_apps: dict[str, set[tuple[str, str]]] = {r.key: set(r.appearances) for r in canon}
    merged_ms: dict[str, set[tuple[int, str]]] = {r.key: set(r.memberships) for r in canon}
    for a, target in alias_of.items():
        if target in merged_apps:
            merged_apps[target] |= by_key[a].appearances
            merged_ms[target] |= by_key[a].memberships

    prof_by_name_dob: dict[tuple[str, str], list[str]] = defaultdict(list)
    prof_by_dob: dict[str, list[str]] = defaultdict(list)
    prof_by_apps: dict[frozenset[tuple[str, str]], list[str]] = defaultdict(list)
    for url, p in sorted(profiles.items()):
        d = full_dob(p.born)
        if d is not None:
            prof_by_dob[d].append(url)
            if p.name:
                prof_by_name_dob[(norm_name(p.name), d)].append(url)
        if p.appearances:
            prof_by_apps[p.appearances].append(url)
    loc_by_name_dob: dict[tuple[str, str], list[str]] = defaultdict(list)
    loc_by_dob: dict[str, list[str]] = defaultdict(list)
    loc_by_apps: dict[frozenset[tuple[str, str]], list[str]] = defaultdict(list)
    for r in canon:
        d = full_dob(r.born)
        if d is not None:
            loc_by_dob[d].append(r.key)
            loc_by_name_dob[(norm_name(r.name), d)].append(r.key)
        apps = frozenset(merged_apps[r.key])
        if apps:
            loc_by_apps[apps].append(r.key)

    pending: dict[str, Resolution] = {}
    for r in canon:
        ms = frozenset(merged_ms[r.key])
        apps = frozenset(merged_apps[r.key])
        res = _resolve_one(
            r,
            ms,
            apps,
            profiles,
            ov.get(r.key),
            prof_by_name_dob,
            prof_by_dob,
            prof_by_apps,
            loc_by_name_dob,
            loc_by_dob,
            loc_by_apps,
            out,
        )
        pending[r.key] = res

    claimed: dict[str, list[str]] = defaultdict(list)
    for key, res in pending.items():
        if res.status == "resolved" and res.url:
            claimed[res.url].append(key)
    for url, keys in sorted(claimed.items()):
        if len(keys) > 1:
            msg = f"profile {url} is claimed by {len(keys)} non-alias local players: {sorted(keys)}"
            out.findings.append(
                IdFinding("IDENTITY_CONFLICT", "ID-DUPLICATE-TARGET", "unknown", tuple(sorted(keys)), msg)
            )
            for k in keys:
                pending[k] = Resolution(k, "conflict", None, None, (url,), msg)
    out.resolutions.update(pending)
    for key, owner in alias_of.items():
        o = out.resolutions.get(owner)
        out.resolutions[key] = Resolution(
            key,
            "alias" if o and o.status == "resolved" else (o.status if o else "unresolved"),
            o.url if o else None,
            f"alias-of:{owner}",
            (),
            "",
        )
    for key, res in out.resolutions.items():
        if res.status == "resolved" and res.url:
            out.profile_to_locals.setdefault(res.url, []).append(key)
    for lst in out.profile_to_locals.values():
        lst.sort()
    out.source_unmatched = sorted(u for u in profiles if u not in out.profile_to_locals)
    return out


def _resolve_one(
    r: LocalIdentityRec,
    ms: frozenset[tuple[int, str]],
    apps: frozenset[tuple[str, str]],
    profiles: dict[str, SourceProfile],
    override: IdentityOverride | None,
    prof_by_name_dob: dict[tuple[str, str], list[str]],
    prof_by_dob: dict[str, list[str]],
    prof_by_apps: dict[frozenset[tuple[str, str]], list[str]],
    loc_by_name_dob: dict[tuple[str, str], list[str]],
    loc_by_dob: dict[str, list[str]],
    loc_by_apps: dict[frozenset[tuple[str, str]], list[str]],
    out: IdentityResult,
) -> Resolution:
    # rule 1: a verified local source URL, checked against the profile's own evidence
    if r.source_urls:
        found = [u for u in r.source_urls if u in profiles]
        if len(found) == 1:
            p = profiles[found[0]]
            ld, pd = full_dob(r.born), full_dob(p.born)
            if ld and pd and ld != pd:
                msg = f"verified URL {p.url}: local born {ld} but profile born {pd}"
                out.findings.append(IdFinding("IDENTITY_CONFLICT", "ID-R1-DOB", "unknown", (r.key,), msg))
                return Resolution(r.key, "conflict", None, "R1-verified-url", (p.url,), msg)
            if ms and p.memberships and not (ms & p.memberships):
                msg = f"verified URL {p.url}: no club/season in common with the local career"
                out.findings.append(IdFinding("IDENTITY_CONFLICT", "ID-R1-CAREER", "unknown", (r.key,), msg))
                return Resolution(r.key, "conflict", None, "R1-verified-url", (p.url,), msg)
            if r.name and p.name and norm_name(r.name) != norm_name(p.name):
                out.findings.append(
                    IdFinding(
                        "IDENTITY_VARIANCE", "ID-R1-NAME", "info", (r.key,), f"local {r.name!r} source {p.name!r}"
                    )
                )
            return Resolution(r.key, "resolved", p.url, "R1-verified-url")
        msg = f"verified URL(s) {list(r.source_urls)} are not among the captured profiles"
        out.findings.append(IdFinding("IDENTITY_UNRESOLVED", "ID-R1-MISSING", "unknown", (r.key,), msg))
        return Resolution(r.key, "unresolved", None, None, tuple(r.source_urls), msg)
    # rule 2: a versioned override with a captured locator (resolves identity only)
    if override is not None:
        if override.source_url in profiles:
            return Resolution(r.key, "resolved", override.source_url, "R2-override", (), override.reason)
        msg = f"override names {override.source_url}, which is not a captured profile"
        out.findings.append(IdFinding("IDENTITY_UNRESOLVED", "ID-R2-MISSING", "unknown", (r.key,), msg))
        return Resolution(r.key, "unresolved", None, None, (override.source_url,), msg)
    dob = full_dob(r.born)
    # rule 3: unique normalised name + full DOB, corroborated by club/season membership
    if dob is not None:
        k = (norm_name(r.name), dob)
        ps, ls = prof_by_name_dob.get(k, []), loc_by_name_dob.get(k, [])
        if len(ps) == 1 and ls:
            p = profiles[ps[0]]
            if not ms or (ms & p.memberships):
                return Resolution(r.key, "resolved", p.url, "R3-name-dob-membership")
        # rule 3b: names diverge: unique DOB + identical membership sets
        ps, ls = prof_by_dob.get(dob, []), loc_by_dob.get(dob, [])
        if len(ps) == 1 and ls and ms and profiles[ps[0]].memberships == ms:
            p = profiles[ps[0]]
            if p.name and norm_name(p.name) != norm_name(r.name):
                out.findings.append(
                    IdFinding(
                        "IDENTITY_VARIANCE", "ID-R3B-NAME", "info", (r.key,), f"local {r.name!r} source {p.name!r}"
                    )
                )
            return Resolution(r.key, "resolved", p.url, "R3b-dob-memberships")
    # rule 4: DOB missing or low precision on either side: globally unique exact appearance-set equality
    if apps:
        cands = prof_by_apps.get(apps, [])
        lcands = loc_by_apps.get(apps, [])
        weak = dob is None or any(profiles[u].born is None or full_dob(profiles[u].born) is None for u in cands)
        if len(cands) == 1 and lcands and weak:
            return Resolution(r.key, "resolved", cands[0], "R4-appearance-set")
    cands2 = sorted({*prof_by_name_dob.get((norm_name(r.name), dob or ""), []), *(prof_by_dob.get(dob or "", []))})
    msg = "no identity rule resolved this player uniquely"
    out.findings.append(IdFinding("IDENTITY_UNRESOLVED", "ID-NO-RULE", "unknown", (r.key,), msg))
    return Resolution(r.key, "unresolved", None, None, tuple(cands2), msg)


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceMatchHeader:
    url: str
    season: int
    stage_text: str
    teams: frozenset[str]
    date: str | None
    #: the final scores were level (so a later same-pair game of the stage can be its replay)
    drawn: bool


@dataclass(frozen=True)
class LocalMatchRec:
    match_id: str
    season: int
    stage_text: str
    teams: frozenset[str]
    date: str | None
    replay: int


@dataclass
class MatchMap:
    local_to_source: dict[str, str] = field(default_factory=dict)
    source_to_local: dict[str, str] = field(default_factory=dict)
    ambiguous: list[str] = field(default_factory=list)
    source_only: list[str] = field(default_factory=list)
    local_only: list[str] = field(default_factory=list)


def map_matches(local: list[LocalMatchRec], source: list[SourceMatchHeader]) -> MatchMap:
    """Pair local and source matches by (season, stage, unordered club pair); same-pair finals are told apart by
    date order and the replay ordinal, never by round text alone."""
    out = MatchMap()
    sgroups: dict[tuple[int, str, frozenset[str]], list[SourceMatchHeader]] = defaultdict(list)
    lgroups: dict[tuple[int, str, frozenset[str]], list[LocalMatchRec]] = defaultdict(list)
    for s in source:
        sgroups[(s.season, s.stage_text, s.teams)].append(s)
    for m in local:
        lgroups[(m.season, m.stage_text, m.teams)].append(m)
    for key in sorted(set(sgroups) | set(lgroups), key=lambda k: (k[0], k[1], sorted(k[2]))):
        ss = sorted(sgroups.get(key, []), key=lambda x: (x.date or "", x.url))
        ls = sorted(lgroups.get(key, []), key=lambda x: (x.replay, x.date or "", x.match_id))
        if not ss:
            out.local_only += [m.match_id for m in ls]
            continue
        if not ls:
            out.source_only += [s.url for s in ss]
            continue
        if len(ss) > 1 and not ss[0].drawn:
            out.ambiguous += [m.match_id for m in ls]  # a repeat without a preceding draw: cannot derive the ordinal
            out.source_only += [s.url for s in ss]
            continue
        paired: set[str] = set()
        for m in ls:
            if m.replay < len(ss) and (len(ss) > 1 or m.replay == 0):
                s = ss[m.replay]
                # a repeated pairing is only trusted when the local date agrees with the ordinal's date
                if (len(ss) > 1 and m.date and s.date and m.date != s.date) or s.url in paired:
                    out.ambiguous.append(m.match_id)
                else:
                    paired.add(s.url)
                    out.local_to_source[m.match_id] = s.url
                    out.source_to_local[s.url] = m.match_id
            else:
                out.ambiguous.append(m.match_id)
        out.source_only += [s.url for s in ss if s.url not in paired]
    out.source_only.sort()
    out.local_only.sort()
    out.ambiguous.sort()
    return out
