"""Identity corrections proposed from the audit's identity results (DESIGN section 14, "Corrections").

Three kinds, each backed by the captured profile page and never by a similarity score:

* ``bind``: a resolved local player gets the verified profile URL as its source URL, so future audits and source
  refreshes recognise the player by the source's own key instead of by name and birth date;
* ``repair``: an unresolved local player whose every appearance is contained in exactly one profile, which no other
  local player resolves to, takes the name and/or birth date that profile prints (a dropped surname prefix such as
  "Paul Haar" for "Paul Vander Haar", or a wrong birth date), and is bound to it;
* ``duplicate``: two local players with identical appearances who both claim one profile; the one whose name the
  profile prints stays canonical, the other is marked a duplicate of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from supercoach_via.reconciliation.identity import (
    IdentityResult,
    LocalIdentityRec,
    SourceProfile,
    full_dob,
    norm_name,
)


@dataclass(frozen=True)
class Proposal:
    kind: str  # bind | repair | duplicate
    key: str  # the local player key
    url: str  # the source profile URL
    fields: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def _split(profile_name: str, local_name: str) -> dict[str, str]:
    """The name fields to change: the profile's full name, and its surname when the first name agrees."""
    out = {"display_name": profile_name}
    first = local_name.split(" ", 1)[0]
    if profile_name.startswith(first + " "):
        out["last_name"] = profile_name[len(first) + 1 :]
    else:
        out["first_name"], _, out["last_name"] = profile_name.partition(" ")
    return out


def identity_proposals(
    profiles: dict[str, SourceProfile], locals_: list[LocalIdentityRec], res: IdentityResult
) -> list[Proposal]:
    by_key = {r.key: r for r in locals_}
    owned = {
        r.url for r in res.resolutions.values() if r.status in ("resolved", "alias") and r.url is not None
    }  # fmt: skip
    out: list[Proposal] = []
    # duplicates first: two local players with identical appearances and birth date pointing at one profile
    dup_keys: set[str] = set()
    groups: dict[tuple[frozenset[tuple[str, str]], str | None], list[LocalIdentityRec]] = {}
    for rec in sorted(locals_, key=lambda x: x.key):
        if rec.appearances:
            groups.setdefault((rec.appearances, rec.born), []).append(rec)
    for recs in groups.values():
        if len(recs) != 2:
            continue
        urls = set()
        for rec in recs:
            r = res.resolutions.get(rec.key)
            if r is not None:
                urls |= {r.url} if r.url else set(r.candidates)
        if len(urls) != 1:
            continue
        url = next(iter(urls))
        prof = profiles.get(url)
        named = [r for r in recs if prof is not None and prof.name and norm_name(r.name) == norm_name(prof.name)]
        if prof is None or len(named) != 1:
            continue
        canon, dup = named[0], next(r for r in recs if r is not named[0])
        out.append(Proposal("duplicate", dup.key, url, {"canonical": canon.key}, "identical appearances, same DOB"))
        if url not in canon.source_urls:
            out.append(Proposal("bind", canon.key, url, {}, "canonical of a duplicate pair"))
        dup_keys |= {canon.key, dup.key}
        owned.add(url)
    # partial duplicates: a stub local record whose games are a strict subset of another local record of the same
    # profile (the harness created a second file for a debutant); the complete record is canonical
    pointing: dict[str, list[LocalIdentityRec]] = {}
    for key, r in sorted(res.resolutions.items()):
        rec0 = by_key.get(key)
        if rec0 is None or key in dup_keys or not rec0.appearances:
            continue
        if r.url:
            target: str | None = r.url
        elif len(r.candidates) == 1:
            target = r.candidates[0]
        else:
            sup = [u for u, p in profiles.items() if rec0.appearances <= p.appearances]
            # several profiles hold these games (teammates): the one a same-named local record containing them owns
            owners: set[str] = {
                o.url
                for k2, o in res.resolutions.items()
                if k2 != key
                and o.url is not None
                and o.url in sup
                and k2 in by_key
                and norm_name(by_key[k2].name) == norm_name(rec0.name)
                and rec0.appearances <= by_key[k2].appearances
            }
            target = sup[0] if len(sup) == 1 else (next(iter(owners)) if len(owners) == 1 else None)
        if target is not None:
            pointing.setdefault(target, []).append(rec0)
    for url, recs in sorted(pointing.items()):
        prof = profiles.get(url)
        if prof is None or len(recs) != 2:
            continue
        a, b = sorted(recs, key=lambda x: len(x.appearances))
        if not (a.appearances < b.appearances <= prof.appearances):
            continue
        out.append(Proposal("duplicate", a.key, url, {"canonical": b.key}, "games are a subset of the canonical's"))
        if url not in b.source_urls:
            out.append(Proposal("bind", b.key, url, {}, "canonical of a duplicate pair"))
        dup_keys |= {a.key, b.key}
        owned.add(url)
    for key, r in sorted(res.resolutions.items()):
        found = by_key.get(key)
        if found is None or key in dup_keys:
            continue
        rec = found
        if r.status == "resolved" and r.url is not None:
            if r.url not in rec.source_urls:
                out.append(Proposal("bind", key, r.url, {}, f"resolved by {r.rule}"))
            continue
        if r.status != "unresolved" or not rec.appearances:
            continue
        cands = [u for u, p in sorted(profiles.items()) if rec.appearances <= p.appearances and u not in owned]
        if len(cands) != 1:
            continue
        prof = profiles[cands[0]]
        fields: dict[str, Any] = {}
        if prof.name and norm_name(prof.name) != norm_name(rec.name):
            fields.update(_split(prof.name, rec.name))
        dob = full_dob(prof.born)
        if dob is not None and dob != rec.born:
            fields["birth_date"] = dob
        if not fields:
            # nothing the page contradicts (e.g. a missing local game blocks the exact-appearance rule): binding the
            # verified URL resolves the identity, and the missing game then surfaces as an appearance finding
            out.append(Proposal("bind", key, cands[0], {}, "every local appearance is in this unclaimed profile"))
        else:
            out.append(Proposal("repair", key, cands[0], fields, "every local appearance is in this unclaimed profile"))
        owned.add(cands[0])
    return sorted(out, key=lambda p: (p.kind, p.key))
