"""Versioned comparison rules (``config/reconciliation_rules.toml``) and identity overrides.

A rule here may resolve an ambiguity from captured evidence (an exact source locator and a
reason are mandatory); it can never waive a statistic mismatch. The whole file is hashed into
the plan and into every cache key, so editing a rule invalidates exactly what depends on it.

Brownlow award facts (the seasons with no medal, the votes awarded per game) are never typed in:
the file pins the URL and SHA-256 of the captured Brownlow index (DESIGN section 15, A9) and the
facts are parsed from those stored bytes by ``evidence.parse_brownlow`` (``Rules.with_evidence``).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from supercoach_via.reconciliation.cells import OneSidedRule
from supercoach_via.reconciliation.evidence import BROWNLOW_URL, BrownlowFacts, EvidenceError, parse_brownlow
from supercoach_via.reconciliation.schema import STAT_FIELDS, sha256_hex
from supercoach_via.settings import default_config_dir

RULES_FILE = "reconciliation_rules.toml"
OVERRIDES_FILE = "reconciliation_identity_overrides.csv"
OVERRIDE_COLUMNS = ("local_player_id", "source_url", "evidence_sha256", "evidence_locator", "reason", "reviewer")


class RulesError(ValueError):
    """The rules or override file is malformed (exit code 2)."""


@dataclass(frozen=True)
class NotesClubAlias:
    season: int
    name: str  # as printed in the notes exception table
    club: str  # season-valid source club name
    evidence_locator: str
    reason: str


@dataclass(frozen=True)
class BrownlowRef:
    """The pinned evidence page: where it came from and the digest its stored bytes must have."""

    url: str
    sha256: str
    evidence_locator: str
    reason: str


@dataclass(frozen=True)
class IdentityOverride:
    local_player_id: str
    source_url: str
    evidence_sha256: str
    evidence_locator: str
    reason: str
    reviewer: str


@dataclass(frozen=True)
class Rules:
    version: int
    notes_labels: dict[str, str] = field(default_factory=dict)
    notes_club_aliases: tuple[NotesClubAlias, ...] = ()
    one_sided: tuple[OneSidedRule, ...] = ()
    brownlow_ref: BrownlowRef | None = None
    #: parsed from the stored evidence page by ``with_evidence``; ``None`` until attached
    brownlow: BrownlowFacts | None = None
    overrides: tuple[IdentityOverride, ...] = ()
    rules_sha256: str = ""
    overrides_sha256: str = ""

    @property
    def no_award_seasons(self) -> frozenset[int]:
        """Seasons in which the Brownlow Medal was not conducted, as the captured evidence states them."""
        return self.brownlow.no_award_seasons if self.brownlow is not None else frozenset()

    def br_award_total(self, season: int) -> int | None:
        """Votes awarded per game in ``season`` per the captured evidence (``None``: none awarded or no evidence)."""
        return self.brownlow.award_total(season) if self.brownlow is not None else None

    def with_evidence(self, body: bytes) -> Rules:
        """Attach the award facts parsed from ``body``; the bytes must hash to the digest the rules pin."""
        if self.brownlow_ref is None:
            raise RulesError(f"{RULES_FILE} pins no evidence: it has no [brownlow] section")
        got = hashlib.sha256(body).hexdigest()
        if got != self.brownlow_ref.sha256:
            raise RulesError(
                "the Brownlow evidence does not hash to the pinned digest "
                f"({got[:12]} != {self.brownlow_ref.sha256[:12]})"
            )
        try:
            facts = parse_brownlow(body)
        except EvidenceError as exc:
            raise RulesError(f"the Brownlow evidence could not be read: {exc}") from exc
        return dataclasses.replace(self, brownlow=facts)

    def club_alias(self, season: int, name: str) -> str:
        for a in self.notes_club_aliases:
            if a.season == season and a.name == name:
                return a.club
        return name


def _need(d: dict[str, object], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise RulesError(f"{where}: missing {missing}")


def parse_rules(rules_text: str, overrides_text: str = "") -> Rules:
    try:
        raw = tomllib.loads(rules_text)
    except tomllib.TOMLDecodeError as exc:
        raise RulesError(f"{RULES_FILE}: {exc}") from exc
    allowed = {"version", "notes_labels", "notes_club_alias", "one_sided_rule", "brownlow"}
    unknown = set(raw) - allowed
    if unknown:
        raise RulesError(f"{RULES_FILE}: unknown sections {sorted(unknown)}")
    if raw.get("version") != 1:
        raise RulesError(f"{RULES_FILE}: unsupported version {raw.get('version')!r}")
    aliases = []
    for a in raw.get("notes_club_alias", []):
        _need(a, ("season", "name", "club", "evidence_locator", "reason"), "notes_club_alias")
        if not str(a["evidence_locator"]).strip() or not str(a["reason"]).strip():
            raise RulesError(f"notes_club_alias {a['name']!r}: an evidence locator and a reason are mandatory")
        aliases.append(
            NotesClubAlias(
                int(a["season"]),
                str(a["name"]),
                str(a["club"]),
                str(a["evidence_locator"]).strip(),
                str(a["reason"]).strip(),
            )
        )
    one_sided = []
    for r in raw.get("one_sided_rule", []):
        _need(
            r,
            ("rule_id", "field", "first_season", "last_season", "printed_on", "evidence_locator", "reason"),
            "one_sided_rule",
        )
        if r["field"] not in STAT_FIELDS or r["printed_on"] not in ("profile", "match"):
            raise RulesError(f"one_sided_rule {r['rule_id']}: bad field or printed_on")
        if not str(r["evidence_locator"]).strip() or not str(r["reason"]).strip():
            raise RulesError(f"one_sided_rule {r['rule_id']}: an evidence locator and a reason are mandatory")
        one_sided.append(
            OneSidedRule(
                str(r["rule_id"]), str(r["field"]), int(r["first_season"]), int(r["last_season"]), str(r["printed_on"])
            )
        )
    brown = raw.get("brownlow")
    brownlow_ref: BrownlowRef | None = None
    if brown is not None:
        extra = set(brown) - {"evidence_url", "evidence_sha256", "evidence_locator", "reason"}
        if extra:
            raise RulesError(
                f"brownlow: {sorted(extra)} must be parsed from the evidence, never typed in "
                "(only evidence_url, evidence_sha256, evidence_locator and reason are allowed)"
            )
        _need(brown, ("evidence_url", "evidence_sha256", "evidence_locator", "reason"), "brownlow")
        if brown["evidence_url"] != BROWNLOW_URL:
            raise RulesError(f"brownlow.evidence_url must be {BROWNLOW_URL}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(brown["evidence_sha256"])):
            raise RulesError("brownlow.evidence_sha256 must be a lowercase hex sha256")
        if not str(brown["evidence_locator"]).strip() or not str(brown["reason"]).strip():
            raise RulesError("brownlow: an evidence locator and a reason are mandatory")
        brownlow_ref = BrownlowRef(
            str(brown["evidence_url"]),
            str(brown["evidence_sha256"]),
            str(brown["evidence_locator"]).strip(),
            str(brown["reason"]).strip(),
        )
    overrides = []
    if overrides_text.strip():
        reader = csv.DictReader(io.StringIO(overrides_text))
        if tuple(reader.fieldnames or ()) != OVERRIDE_COLUMNS:
            raise RulesError(f"{OVERRIDES_FILE}: header must be {','.join(OVERRIDE_COLUMNS)}")
        for row in reader:
            if any(not (row[c] or "").strip() for c in OVERRIDE_COLUMNS):
                raise RulesError(f"{OVERRIDES_FILE}: every override needs all of {OVERRIDE_COLUMNS}: {row}")
            overrides.append(IdentityOverride(*(row[c].strip() for c in OVERRIDE_COLUMNS)))
    return Rules(
        version=1,
        notes_labels={str(k): str(v) for k, v in raw.get("notes_labels", {}).items()},
        notes_club_aliases=tuple(aliases),
        one_sided=tuple(one_sided),
        brownlow_ref=brownlow_ref,
        overrides=tuple(overrides),
        rules_sha256=sha256_hex(rules_text.encode()),
        overrides_sha256=sha256_hex(overrides_text.encode()),
    )


def load_rules(config_dir: Path | None = None) -> Rules:
    cfg = config_dir or default_config_dir()
    return parse_rules(
        (cfg / RULES_FILE).read_text(encoding="utf-8"), (cfg / OVERRIDES_FILE).read_text(encoding="utf-8")
    )
