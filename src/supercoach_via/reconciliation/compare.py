"""The reconciliation audit engine: ``scvia reconcile-afltables compare`` (DESIGN sections 7-10).

Offline and deterministic. Phases: (1) verify the plan, manifest, rules and local inputs; (2) parse
the frozen capture into cached facts; (3) inventory the source scope and align matches; (4) resolve
player identity per layer; (5) compare every season for every requested layer (parallel by season);
(6) reduce per-player aggregates; (7) account for coverage, derive per-layer verdicts and write the
reports. Worker count, cache state and traversal order never change the canonical outputs.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import platform
import resource
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from supercoach_via.reconciliation import facts as FX
from supercoach_via.reconciliation import identity as ID
from supercoach_via.reconciliation import matches as MT
from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation.cache import FactsCache, parser_identity
from supercoach_via.reconciliation.findings import make_finding
from supercoach_via.reconciliation.inventory import PlanError, load_plan, pin_legacy
from supercoach_via.reconciliation.local import (
    LocalDataError,
    LocalGame,
    LocalLegacy,
    LocalSnapshot,
    game_from_json,
    game_to_json,
)
from supercoach_via.reconciliation.reduce import ReduceResult, reduce_profile
from supercoach_via.reconciliation.rules import Rules, RulesError, load_rules
from supercoach_via.reconciliation.season import (
    PROFILE_KEYS,
    ClubSeasonAgg,
    ProfileView,
    QRow,
    SeasonInput,
    SeasonResult,
    compare_season,
    season_rows_of,
)
from supercoach_via.reconciliation.sink import ChunkInfo, merge_to_file, write_chunk
from supercoach_via.reconciliation.source import NotesFacts

LAYERS = ("snapshot", "legacy_csv")
FINAL_STAGES = frozenset(
    {"Wildcard Final", "Qualifying Final", "Elimination Final", "Semi Final", "Preliminary Final", "Grand Final"}
)
_GAME_YEAR = "/games/"


class CompareError(RuntimeError):
    """Invalid invocation (exit code 2)."""


@dataclass
class CompareOptions:
    plan: Path
    capture_manifest: Path
    out: Path
    workers: int = 1
    cache: Path | None = None
    previous: Path | None = None
    config_dir: Path | None = None


def _year_of(url: str) -> int:
    return int(url.split(_GAME_YEAR, 1)[1][:4])


# ---------------------------------------------------------------------------
# Worker side (module level so a process pool can import it)
# ---------------------------------------------------------------------------

_SHARED: dict[str, Any] = {}


def _init_worker(shared: dict[str, Any]) -> None:
    _SHARED.clear()
    _SHARED.update(shared)


@dataclass
class SeasonTask:
    layer: str
    season: int
    match_items: list[tuple[str, str]]
    absent: dict[str, str]
    #: (profile url, body sha256, name)
    profile_items: list[tuple[str, str, str | None]]
    local_by_pair: dict[tuple[str, str], list[LocalGame]] = field(default_factory=dict)
    local_by_profile: dict[str, list[LocalGame]] = field(default_factory=dict)
    unplaced: list[tuple[LocalGame, str]] = field(default_factory=list)
    local_ids: dict[str, str] = field(default_factory=dict)
    quarantine: dict[str, list[QRow]] = field(default_factory=dict)
    profile_status: dict[str, str] = field(default_factory=dict)
    emit_source: bool = False
    track_matches: frozenset[str] = frozenset()
    #: source match URLs of this season inside the audited scope (the event boundary)
    scope_matches: frozenset[str] = frozenset()
    local_excluded: int = 0
    #: (profile url, club) -> statistic -> season value the local layer stores (season award rows)
    local_summary: dict[tuple[str, str], dict[str, Decimal]] = field(default_factory=dict)
    #: award rows of a local player no identity rule maps (reported, never attributed)
    unplaced_awards: int = 0


def run_season_task(task: SeasonTask) -> SeasonResult:
    cache = FactsCache(Path(_SHARED["cache_root"]), _SHARED["phash"])
    notes: NotesFacts = _SHARED["notes"]
    matches = {}
    shas = {}
    absent = dict(task.absent)
    for url, sha in task.match_items:
        m = FX.load_match(cache, sha)
        if m is None:
            absent[url] = "parsed facts unavailable"
        else:
            matches[url] = m
            shas[url] = sha
    profiles: dict[str, ProfileView] = {}
    for url, sha, name in task.profile_items:
        all_games = FX.load_profile_games(cache, sha, task.season) or []
        games = [g for g in all_games if g.match_url in task.scope_matches]
        excluded = [g for g in all_games if g.match_url not in task.scope_matches]
        core = FX.load_core(cache, sha)
        bad: dict[str, frozenset[str]] = {}
        rows: dict[str, Any] = {}
        if core is not None:
            bad = {cs.club: frozenset(cs.bad_fields) for cs in core.club_seasons if cs.season == task.season}
            rows = season_rows_of(core, task.season)
        profiles[url] = ProfileView(url, name, games, bad, sha, excluded_games=excluded, season_rows=rows)
    inp = SeasonInput(
        layer=task.layer,
        season=task.season,
        matches=matches,
        match_sha=shas,
        absent_matches=absent,
        profiles=profiles,
        notes=notes,
        rules=_SHARED["rules"],
        local_by_pair=task.local_by_pair,
        local_by_profile=task.local_by_profile,
        unplaced=task.unplaced,
        local_ids=task.local_ids,
        quarantine=task.quarantine,
        profile_status=task.profile_status,
        captured=_SHARED["captured"],
        emit_source=task.emit_source,
        track_matches=task.track_matches,
        local_excluded=task.local_excluded,
        local_summary=task.local_summary,
    )
    res = compare_season(inp)
    if task.unplaced_awards:
        res.counters["local_awards_unattributed"] += task.unplaced_awards
    res.chunk = write_chunk(Path(_SHARED["chunk_dir"]), f"{task.layer}-{task.season:05d}", res.findings)
    res.findings = []
    return res


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


@dataclass
class LayerState:
    name: str
    available: bool = True
    problems: list[str] = field(default_factory=list)
    identity: ID.IdentityResult | None = None
    match_result: MT.MatchResult | None = None
    season_results: dict[int, SeasonResult] = field(default_factory=dict)
    reduce_counters: Counter[str] = field(default_factory=Counter)
    reduce_by_stat: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    findings: list[dict[str, Any]] = field(default_factory=list)
    local_players: int = 0
    local_games: int = 0
    players_rows: list[dict[str, Any]] = field(default_factory=list)
    #: kept only for correction proposals (``Audit.keep_identity``): the identity inputs
    identity_inputs: tuple[dict[str, Any], list[Any]] | None = None


def _canon(obj: Any) -> bytes:
    from supercoach_via.integrity.report import canonical_bytes

    return canonical_bytes(obj)


class UnitCache:
    """Stored season results keyed by the unit's full input digest plus a code/parser salt (permitted S-14 cache).

    A hit is reused evidence derived from frozen inputs, never a fresh fetch; every entry carries the digest of
    its payload, so a corrupt, truncated or mismatching file is a miss and is recomputed.
    """

    def __init__(self, root: Path, salt: str) -> None:
        self.root = root / "units"
        self.salt = salt

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.{self.salt}.bin"

    def get(self, digest: str, chunk_dir: Path, chunk_name: str) -> SeasonResult | None:
        import pickle

        try:
            raw = self._path(digest).read_bytes()
        except FileNotFoundError:
            return None
        head, sep, payload = raw.partition(b"\n")
        if not sep or head.decode("ascii", "replace") != hashlib_sha(payload):
            return None
        try:
            res, chunk_bytes, info = pickle.loads(payload)  # noqa: S301 - our own file, digest-verified above
        except (pickle.UnpicklingError, EOFError, AttributeError, ValueError):
            return None
        chunk_dir.mkdir(parents=True, exist_ok=True)
        path = chunk_dir / f"{chunk_name}.jsonl"
        path.write_bytes(chunk_bytes)
        info.path = str(path)
        res.chunk = info
        return res  # type: ignore[no-any-return]

    def put(self, digest: str, res: SeasonResult) -> None:
        import pickle

        assert res.chunk is not None
        chunk_bytes = Path(res.chunk.path).read_bytes()
        keep = res.chunk
        res.chunk = None
        try:
            payload = pickle.dumps((res, chunk_bytes, keep), protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            res.chunk = keep
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(hashlib_sha(payload).encode("ascii") + b"\n" + payload)
        os.replace(tmp, path)


class Audit:
    def __init__(self, opts: CompareOptions) -> None:
        self.opts = opts
        self.timings: dict[str, float] = {}
        self.rss_after: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self.layers: dict[str, LayerState] = {}
        self.source_findings: list[dict[str, Any]] = []
        self.source_counters: Counter[str] = Counter()

    def _tick(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = round(self.timings.get(name, 0.0) + now - self._t0, 3)  # a phase may run per layer
        self.rss_after[name] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        self._t0 = now

    # -- phase 1: verify -----------------------------------------------------------------------------

    def load(self) -> None:
        o = self.opts
        self.plan = load_plan(o.plan)
        try:
            self.manifest = S.Manifest.model_validate_json(o.capture_manifest.read_bytes())
        except (OSError, ValueError) as exc:
            raise CompareError(f"cannot read capture manifest {o.capture_manifest}: {exc}") from exc
        if self.manifest.capture_identity != self.plan.capture_identity:
            raise CompareError("the capture manifest was produced under a different capture identity than the plan")
        try:
            self.rules: Rules = load_rules(o.config_dir)
        except (RulesError, OSError) as exc:
            raise CompareError(f"cannot load comparison rules: {exc}") from exc
        pins = self.plan.policies
        if (
            self.rules.rules_sha256 != pins.rules_sha256
            or self.rules.overrides_sha256 != pins.identity_overrides_sha256
        ):
            raise CompareError("the rules or identity overrides differ from the plan; re-run `plan` to pin them")
        if self.rules.brownlow_ref is not None:
            # the award facts come from the run's own evidence archive, bytes re-verified against the pinned digest
            from supercoach_via.reconciliation import evidence as EV

            self.evidence_dir = Path(self.plan.operational.run_dir) / EV.EVIDENCE_DIR
            try:
                body = EV.load_evidence(self.evidence_dir, self.rules.brownlow_ref.sha256)
                self.rules = self.rules.with_evidence(body)
            except (EV.EvidenceError, RulesError) as exc:
                raise CompareError(f"cannot load the pinned Brownlow evidence: {exc}") from exc
        self.capture_dir = o.capture_manifest.parent
        # The manifest builder counts only first-pass tasks as pending, so a capture interrupted during the
        # end-of-acquisition revalidation can still print capture_complete=true. The receipt written when the
        # coordinator exits records how it ended; completeness requires both.
        self.receipt_problem: str | None = None
        try:
            receipt = json.loads((self.capture_dir / "receipt.json").read_bytes())
            if receipt.get("state") != "complete" or receipt.get("capture_complete") is not True:
                self.receipt_problem = f"capture receipt says state={receipt.get('state')!r}"
        except (OSError, ValueError):
            self.receipt_problem = "capture receipt.json is missing or unreadable"
        self.capture_complete = self.manifest.capture_complete and self.receipt_problem is None
        self.through = date.fromisoformat(self.plan.scope.through_date)
        # every directory the audit reads; nothing it writes (report, work dir, cache) may live inside one (T23, B4)
        roots = input_roots_of(self.plan, self.capture_dir)
        if getattr(self, "evidence_dir", None) is not None:
            roots.append(self.evidence_dir.resolve())
        self.cache_root = (o.cache or (o.out.parent / "cache")).resolve()
        for label, target in (("--out", o.out.resolve()), ("--cache", self.cache_root)):
            for root in roots:
                if target == root or root in target.parents or target in root.parents:
                    raise CompareError(f"{label} {target} overlaps the input root {root}; nothing was written")
        if (o.out / "output-manifest.json").exists():
            raise CompareError(f"{o.out} already holds a completed report; refusing to overwrite it")
        import tempfile

        o.out.parent.mkdir(parents=True, exist_ok=True)
        self.work_dir = Path(tempfile.mkdtemp(prefix=f".{o.out.name}.work-", dir=o.out.parent))
        self.chunk_dir = self.work_dir / "chunks"
        self.legacy_root = Path(self.plan.operational.legacy_root) if self.plan.operational.legacy_root else None
        self.snap: LocalSnapshot | None = None
        self.layers["snapshot"] = LayerState("snapshot")
        try:
            self.snap = LocalSnapshot.open(Path(self.plan.operational.data_root), self.plan.inputs.snapshot)
        except LocalDataError as exc:
            self.layers["snapshot"].available = False
            self.layers["snapshot"].problems.append(str(exc))
        self.legacy: LocalLegacy | None = None
        if self.plan.inputs.legacy is not None:
            self.layers["legacy_csv"] = LayerState("legacy_csv")
            try:
                if self.legacy_root is None:
                    raise PlanError("the plan requests the legacy layer but records no legacy root")
                now = pin_legacy(self.legacy_root)
                if now != self.plan.inputs.legacy:
                    raise PlanError("legacy CSV membership or content differs from the plan's pinned hashes")
                self.legacy = LocalLegacy(self.legacy_root)
            except PlanError as exc:
                self.layers["legacy_csv"].available = False
                self.layers["legacy_csv"].problems.append(str(exc))
        # season-level values each layer stores (``player_season_awards``; ``data/awards/*.csv``)
        self.local_awards = {
            "snapshot": self.snap.season_awards() if self.snap is not None else [],
            "legacy_csv": self.legacy.season_awards() if self.legacy is not None else [],
        }
        self._tick("verify")

    # -- phase 2: parse --------------------------------------------------------------------------------

    def parse(self) -> None:
        notes_labels = self.rules.notes_labels
        self.phash = parser_identity(json.dumps(sorted(notes_labels.items())))
        self.res_by_url = {r.url: r for r in self.manifest.resources}
        self.headers = FX.parse_all(
            self.manifest.resources,
            objects_root=self.capture_dir,
            cache_root=self.cache_root,
            phash=self.phash,
            notes_labels=notes_labels,
            workers=self.opts.workers,
        )
        # every appearance repeats a handful of strings (club, opponent, round, the shared match URL): intern them
        import sys

        for h in self.headers.values():
            if "appearances" in h:
                h["appearances"] = [
                    tuple(sys.intern(x) if isinstance(x, str) else x for x in a) for a in h["appearances"]
                ]
        self.cache_stats = Counter(h.get("cache", "?") for h in self.headers.values())
        cache = FactsCache(self.cache_root, self.phash)
        notes_res = next((r for r in self.manifest.resources if r.kind == "notes" and r.status == "usable"), None)
        self.notes = FX.load_notes(cache, notes_res.sha256 or "") if notes_res else None
        if self.notes is None:
            self.notes = NotesFacts(
                availability={}, exceptions=(), problems=("NOTES_UNAVAILABLE: no usable notes page",)
            )
            self.source_findings.append(
                make_finding(
                    "CAPTURE_GAP",
                    layer="source",
                    rule_id="R-NOTES-MISSING",
                    detail="the source notes page is missing: statistic availability after 1964 cannot be established",
                )
            )
        for url, h in sorted(self.headers.items()):
            for p in h.get("problems", []):
                self.source_counters["source_page_problems"] += 1
                code = p.split(":", 1)[0]
                self.source_findings.append(
                    make_finding(
                        "SCHEMA_GAP" if code != "OBJECT_MISSING" else "CAPTURE_GAP",
                        layer="source",
                        rule_id=f"R-PAGE-{code}",
                        evidence={"source_url": url, "body_sha256": self.res_by_url[url].sha256, "locator": "page"},
                        detail=p,
                    )
                )
        for r in sorted(self.manifest.resources, key=lambda x: (x.kind, x.url)):
            if r.status in ("usable", "absent") or r.kind == "match":  # match gaps are reported by the inventory
                continue
            self.source_findings.append(
                make_finding(
                    "CAPTURE_GAP",
                    layer="source",
                    rule_id=f"R-{r.kind.upper()}-{r.status.upper()}",
                    evidence={"source_url": r.url, "locator": f"{r.kind} resource"},
                    detail=f"{r.status}: {r.reason or 'no reason recorded'}",
                )
            )
        for u in self.manifest.census.lineup_profiles_not_in_directory:
            self.source_findings.append(
                make_finding(
                    "CAPTURE_GAP",
                    layer="source",
                    rule_id="R-CENSUS-CLOSURE",
                    player={"source_url": u},
                    detail="a profile linked from a match lineup is absent from the directory census",
                )
            )
        self._tick("parse")

    # -- phase 3: scope and match alignment ------------------------------------------------------------

    def inventory(self) -> None:
        self.fixture_by_url: dict[str, tuple[str, str, str, str | None]] = {}
        self.excluded_urls: set[str] = set()
        seasons_in_scope: set[int] = set()
        undated: list[str] = []
        excluded = 0
        for _url, h in sorted(self.headers.items()):
            if h.get("kind") != "season" or h.get("year") is None:
                continue
            seasons_in_scope.add(int(h["year"]))
            for stage, home, away, d, game_url, _scored in h["fixtures"]:
                if game_url is None:
                    continue
                if d is not None and d > self.through.isoformat():
                    excluded += 1
                    self.excluded_urls.add(game_url)
                    continue
                if d is None:
                    undated.append(game_url)
                self.fixture_by_url[game_url] = (stage, home, away, d)
        self.source_counters["matches_excluded_after_boundary"] = excluded
        self.source_counters["matches_undated_in_scope"] = len(undated)
        for u in sorted(undated):
            self.source_findings.append(
                make_finding(
                    "CAPTURE_GAP",
                    layer="source",
                    rule_id="R-UNDATED",
                    match={"source_url": u},
                    detail="the season page gives this match no date: inclusion before the boundary is unclear",
                )
            )
        # profile-linked games absent from every season page are fetched as in-scope when their season is
        profile_linked = {
            a[0]
            for h in self.headers.values()
            if h.get("kind") == "profile"
            for a in h.get("appearances", [])
            if a[0]
            and a[0] not in self.fixture_by_url
            and a[0] not in self.excluded_urls
            and _year_of(a[0]) <= self.through.year
            and (self.plan.scope.population != "seasons" or _year_of(a[0]) in self.plan.scope.seasons)
        }
        self.unlisted_matches = sorted(profile_linked)
        self.scope_urls: dict[str, int] = {u: _year_of(u) for u in self.fixture_by_url}
        for u in self.unlisted_matches:
            self.scope_urls[u] = _year_of(u)
            self.source_findings.append(
                make_finding(
                    "SOURCE_CONFLICT",
                    layer="source",
                    rule_id="R-MATCH-NOT-ON-SEASON-PAGE",
                    match={"source_url": u},
                    field="inventory",
                    detail="a profile links this game but no season page lists it",
                )
            )
        self.usable_matches: dict[str, str] = {}
        self.absent_matches: dict[str, str] = {}
        for url in sorted(self.scope_urls):
            r = self.res_by_url.get(url)
            hdr = self.headers.get(url)
            if r is None:
                self.absent_matches[url] = "not part of the capture"
            elif r.status != "usable":
                self.absent_matches[url] = f"{r.status}: {r.reason or ''}".strip()
            elif hdr is None or hdr.get("parser_error") or r.sha256 is None:
                self.absent_matches[url] = "parsed facts unavailable"
            else:
                self.usable_matches[url] = r.sha256
        for url, why in sorted(self.absent_matches.items()):
            self.source_findings.append(
                make_finding(
                    "CAPTURE_GAP",
                    layer="source",
                    rule_id="R-MATCH-NOT-CAPTURED",
                    match={"source_url": url},
                    detail=why,
                )
            )
        finals = [
            (d, url, stage)
            for url, (stage, _h, _a, d) in self.fixture_by_url.items()
            if d is not None and stage in FINAL_STAGES
        ]
        self.latest_final = max(finals) if finals else None
        self.seasons = sorted(seasons_in_scope | set(self.scope_urls.values()))
        if self.plan.scope.population == "seasons":  # a seasons audit: other seasons of a career are out of scope
            self.seasons = [y for y in self.seasons if y in self.plan.scope.seasons]
        self.source_infos: list[MT.SourceMatchInfo] = []
        self.match_facts_cache = FactsCache(self.cache_root, self.phash)
        for url in sorted(self.scope_urls):
            sha = self.usable_matches.get(url)
            fx = self.fixture_by_url.get(url)
            if sha is None:
                stage, home, away, d = fx if fx else ("", "", "", None)
                self.source_infos.append(
                    MT.SourceMatchInfo(
                        url,
                        _year_of(url),
                        stage.replace("Round ", ""),
                        (home, away),
                        d,
                        {},
                        None,
                        False,
                        self.absent_matches.get(url),
                    )
                )
                continue
            m = FX.load_match(self.match_facts_cache, sha)
            if m is None:
                self.source_infos.append(
                    MT.SourceMatchInfo(url, _year_of(url), "", ("", ""), None, {}, sha, False, "facts unreadable")
                )
                continue
            names = sorted(t.name for t in m.teams)
            teams = (names[0], names[1]) if len(names) == 2 else ("", "")
            scores = {t.name: (t.quarters[-1][0], t.quarters[-1][1]) for t in m.teams}
            self.source_infos.append(
                MT.SourceMatchInfo(
                    url, _year_of(url), m.stage_text or "", teams, m.match_date, scores, sha, len(names) == 2
                )
            )
        teams_by_season: dict[int, set[str]] = defaultdict(set)
        for info in self.source_infos:
            teams_by_season[info.season].update(t for t in info.teams if t)
        if self.notes is not None:
            seen_names: set[tuple[int, str]] = set()
            for e in self.notes.exceptions:
                if e.season not in teams_by_season:
                    continue  # no source match of that season is in this audit's inventory
                names = list(e.teams) + [n for pair in e.matchups for n in pair]
                for n in names:
                    club = self.rules.club_alias(e.season, n)
                    if club not in teams_by_season.get(e.season, set()) and (e.season, n) not in seen_names:
                        seen_names.add((e.season, n))
                        self.source_findings.append(
                            make_finding(
                                "SCHEMA_GAP",
                                layer="source",
                                rule_id="R-NOTES-CLUB",
                                season=e.season,
                                field=e.category,
                                detail=(
                                    f"notes exception names {n!r}, which matches no team of {e.season}; add a "
                                    "notes_club_alias rule with an evidence locator (DESIGN section 8, S-08)"
                                ),
                                extra_id=n,
                            )
                        )
        self.profiles_with_apps = {
            u
            for u, h in self.headers.items()
            if h.get("kind") == "profile"
            and not h.get("parser_error")
            and any(a[0] in self.scope_urls for a in h.get("appearances", []))
        }
        self._tick("inventory")

    # -- phase 4: identity ------------------------------------------------------------------------------

    def _source_profiles(self, legacy: bool) -> dict[str, ID.SourceProfile]:
        census = {p[1] for h in self.headers.values() if h.get("kind") == "letter" for p in h.get("profiles", [])}
        out: dict[str, ID.SourceProfile] = {}
        for url, h in sorted(self.headers.items()):
            if h.get("kind") != "profile" or h.get("parser_error"):
                continue
            apps: set[tuple[str, str]] = set()
            ms: set[tuple[int, str]] = set()
            for murl, club, season, rd, opp, _counter in h["appearances"]:
                if murl is None:
                    continue
                # a seasons audit still identifies legacy players by CAREER: within one season teammates share
                # identical appearance sets, and the legacy key (season|opponent|round, club) needs no match URL
                career = legacy and self._scoped and season not in self.seasons
                if murl not in self.scope_urls and not career:
                    continue
                ms.add((season, club))
                apps.add((f"{season}|{opp}|{rd}", club) if legacy else (murl, club))
            out[url] = ID.SourceProfile(url, h.get("h1"), h.get("born"), frozenset(apps), frozenset(ms), url in census)
        return out

    def identity_snapshot(self) -> None:
        st = self.layers["snapshot"]
        if self.snap is None:
            return
        local_matches = self._in_scope(MT.snapshot_matches(self.snap.matches()))
        st.match_result = MT.compare_matches(
            "snapshot", local_matches, self.source_infos, through=self.through.isoformat()
        )
        self.match_map = st.match_result.mapping
        self.local_match_date = {m.rec.match_id: m.rec.date for m in local_matches}
        urls_by_player: dict[str, set[str]] = defaultdict(set)
        for a in self.snap.aliases():
            if a.get("source_url"):
                urls_by_player[a["player_id"]].add(a["source_url"])
        apps: dict[str, set[tuple[str, str]]] = defaultdict(set)
        ms: dict[str, set[tuple[int, str]]] = defaultdict(set)
        total = 0
        for season in self.snap.seasons:
            if self._scoped and season not in self.seasons:
                continue  # a seasons audit: the other seasons' rows are neither judged nor counted
            for g in self.snap.games(season):
                total += 1
                murl = self.match_map.local_to_source.get(g.match_key or "")
                ms[g.player_key].add((g.season, g.club))
                if murl is not None:
                    apps[g.player_key].add((murl, g.club))
        st.local_games = total
        players = self.snap.players()
        if self._scoped:  # only players with rows in the audited seasons take part in identity
            players = [p for p in players if p.key in ms or (p.canonical_player_id or "") in ms]
        st.local_players = len(players)
        recs = [
            ID.LocalIdentityRec(
                "snapshot",
                p.key,
                p.display_name,
                p.birth_date,
                tuple(sorted(set(p.source_urls) | urls_by_player[p.key])),
                frozenset(apps[p.key]),
                frozenset(ms[p.key]),
                p.identity_status,
                p.canonical_player_id,
            )
            for p in players
        ]
        profiles = self._source_profiles(False)
        st.identity = ID.resolve_identities(profiles, recs, self.rules.overrides)
        if getattr(self, "keep_identity", False):
            st.identity_inputs = (profiles, recs)
        self.snapshot_players = {p.key: p for p in players}
        self._tick("identity_snapshot")

    def identity_legacy(self) -> None:
        st = self.layers.get("legacy_csv")
        if st is None or self.legacy is None:
            return
        local_matches, problems = MT.legacy_matches(self.legacy.root)
        local_matches = self._in_scope(local_matches)
        st.problems.extend(problems)
        st.match_result = MT.compare_matches(
            "legacy_csv", local_matches, self.source_infos, through=self.through.isoformat()
        )
        recs = []
        total = 0
        for p in self.legacy.players():
            if self._scoped and not any(g.season in self.seasons for g in p.games):
                continue  # a seasons audit: a player with no rows in the audited seasons is not part of it
            total += 1
            st.local_games += sum(1 for g in p.games if not self._scoped or g.season in self.seasons)
            for prob in p.problems:
                st.findings.append(
                    make_finding(
                        "LOCAL_INPUT_GAP",
                        layer="legacy_csv",
                        rule_id="R-LEGACY-READ",
                        player={"local_id": p.slug},
                        detail=prob,
                    )
                )
            name = f"{p.first_name or ''} {p.last_name or ''}".strip()
            apps = frozenset(
                (f"{g.season}|{g.opponent or ''}|{g.stage}", g.club)
                for g in p.games
                if g.season in self.seasons or self._scoped  # scoped: career appearances, as the profiles carry
            )
            ms = frozenset((g.season, g.club) for g in p.games)
            recs.append(ID.LocalIdentityRec("legacy_csv", p.slug, name, p.birth_date, (), apps, ms, "canonical", None))
        st.local_players = total
        profiles = self._source_profiles(True)
        st.identity = ID.resolve_identities(profiles, recs, ())
        if getattr(self, "keep_identity", False):
            st.identity_inputs = (profiles, recs)
        self._tick("identity_legacy")

    @property
    def _scoped(self) -> bool:
        return self.plan.scope.population == "seasons"

    def _in_scope(self, local_matches: list[Any]) -> list[Any]:
        if not self._scoped:
            return local_matches
        return [m for m in local_matches if m.rec.season in self.seasons]

    def release_parse_state(self) -> None:
        """Drop what only identity needed (every profile's appearance list); later phases read cached facts."""
        for h in self.headers.values():
            h.pop("appearances", None)
        gc.collect()

    # -- phase 5: season units ---------------------------------------------------------------------------

    def _claimed(self, layer: str) -> set[str]:
        ident = self.layers[layer].identity
        out: set[str] = set()
        if ident is not None:
            for r in ident.resolutions.values():
                if r.status in ("unresolved", "ambiguous", "conflict"):
                    out.update(r.candidates)
        return out

    def _profile_index(self) -> None:
        self.profile_seasons: dict[int, list[tuple[str, str, str | None]]] = defaultdict(list)
        self.profile_info: dict[str, tuple[str, str | None]] = {}
        captured: set[str] = set()
        for url, h in sorted(self.headers.items()):
            if h.get("kind") != "profile" or h.get("parser_error"):
                continue
            r = self.res_by_url[url]
            if r.sha256 is None:
                continue
            captured.add(url)
            self.profile_info[url] = (r.sha256, h.get("h1"))
            for season in h.get("seasons", []):
                self.profile_seasons[int(season)].append((url, r.sha256, h.get("h1")))
        self.captured = frozenset(captured)

    def _quarantine_rows(self) -> dict[int, dict[str, list[QRow]]]:
        out: dict[int, dict[str, list[QRow]]] = defaultdict(lambda: defaultdict(list))
        st = self.layers["snapshot"]
        if self.snap is None or st.identity is None:
            return out
        for q in self.snap.quarantine():
            if q.get("table_name") != "player_games":
                continue
            raw = json.loads(q["raw"]) if isinstance(q.get("raw"), str) else (q.get("raw") or {})
            pid = raw.get("player_id")
            if pid is None:
                continue
            owner = st.identity.owner.get(pid, pid)
            res = st.identity.resolutions.get(owner)
            if res is None or not res.url:
                continue
            cands = (
                tuple(json.loads(q["candidates"])) if isinstance(q.get("candidates"), str) and q["candidates"] else ()
            )
            season = int(raw.get("season") or q.get("season") or 0)
            counter = raw.get("career_game_counter")
            out[season][res.url].append(
                QRow(
                    q["quarantine_id"],
                    pid,
                    season,
                    raw.get("club_source_name", ""),
                    raw.get("opponent_source_name"),
                    int(counter) if counter is not None else None,
                    cands,
                )
            )
        return out

    def _legacy_shards(self) -> Path:
        shard_dir = self.cache_root / "work" / "legacy"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for old in shard_dir.glob("*.jsonl"):
            old.unlink()
        st = self.layers["legacy_csv"]
        assert self.legacy is not None and st.identity is not None
        files: dict[int, Any] = {}
        try:
            for p in self.legacy.players():
                res = st.identity.resolutions.get(p.slug)
                purl = res.url if res is not None and res.status == "resolved" else None
                reason = "no_profile" if res is None or res.status == "unresolved" else f"identity_{res.status}"
                for g in p.games:
                    fh = files.get(g.season)
                    if fh is None:
                        fh = files[g.season] = (shard_dir / f"{g.season}.jsonl").open("a", encoding="utf-8")
                    rec = {"profile": purl, "reason": reason, "game": game_to_json(g)}
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
        finally:
            for fh in files.values():
                fh.close()
        return shard_dir

    def _build_task(
        self,
        layer: str,
        season: int,
        emit_source: bool,
        quarantine: dict[int, dict[str, list[QRow]]],
        shard_dir: Path | None,
    ) -> SeasonTask:
        st = self.layers[layer]
        ident = st.identity
        assert ident is not None
        claimed = self._claimed(layer)
        items = {url: (sha, name) for url, sha, name in self.profile_seasons.get(season, [])}
        task = SeasonTask(
            layer=layer,
            season=season,
            match_items=sorted((u, sha) for u, sha in self.usable_matches.items() if self.scope_urls[u] == season),
            absent={u: why for u, why in self.absent_matches.items() if self.scope_urls[u] == season},
            profile_items=[],
            emit_source=emit_source,
            track_matches=frozenset({self.latest_final[1]} if self.latest_final else ()),
            scope_matches=frozenset(u for u, y in self.scope_urls.items() if y == season),
        )
        if layer == "snapshot":
            assert self.snap is not None
            for g in self.snap.games(season):
                owner = ident.owner.get(g.player_key, g.player_key)
                res = ident.resolutions.get(owner)
                murl = self.match_map.local_to_source.get(g.match_key or "")
                if res is None or res.status not in ("resolved", "alias") or not res.url:
                    task.unplaced.append((g, f"identity_{res.status if res else 'unknown'}"))
                elif murl is None:
                    ld = self.local_match_date.get(g.match_key or "")
                    if ld is not None and ld[:10] > self.through.isoformat():
                        task.local_excluded += 1
                    else:
                        task.unplaced.append((g, "match_unmapped"))
                elif res.url not in self.captured:
                    task.unplaced.append((g, "profile_not_captured"))
                else:
                    task.local_by_pair.setdefault((res.url, murl), []).append(g)
                    items.setdefault(res.url, self.profile_info[res.url])
            task.quarantine = {u: rows for u, rows in quarantine.get(season, {}).items()}
        else:
            assert shard_dir is not None
            path = shard_dir / f"{season}.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        g = game_from_json(rec["game"])
                        purl = rec["profile"]
                        if purl is None:
                            task.unplaced.append((g, rec["reason"]))
                        elif purl not in self.captured:
                            task.unplaced.append((g, "profile_not_captured"))
                        else:
                            task.local_by_profile.setdefault(purl, []).append(g)
                            items.setdefault(purl, self.profile_info[purl])
        for key, season_, club, award, value in getattr(self, "local_awards", {}).get(layer, []):
            if season_ != season:
                continue
            res = ident.resolutions.get(ident.owner.get(key, key))
            if res is None or res.status not in ("resolved", "alias") or not res.url:
                task.unplaced_awards += 1
                continue
            slot = task.local_summary.setdefault((res.url, club), {})
            slot[award] = slot.get(award, Decimal(0)) + Decimal(value)
        task.profile_items = sorted((u, sha, name) for u, (sha, name) in items.items())
        for u, _sha, _n in task.profile_items:
            locs = ident.profile_to_locals.get(u)
            task.local_ids[u] = ",".join(locs) if locs else ""
            task.profile_status[u] = "mapped" if locs else ("unresolved" if u in claimed else "missing")
        return task

    def _unit_salt(self) -> str:
        """What a stored season result depends on besides its inputs: the comparison code and the parser."""
        from supercoach_via.reconciliation.cache import code_digest

        return hashlib_sha((code_digest(("reconciliation/season.py",)) + self.phash).encode())[:16]

    def _task_digest(self, t: SeasonTask) -> str:
        """Identity of everything a season unit depends on, sliced to that season where the dependency is."""
        assert self.notes is not None
        n = self.notes
        notes_slice = {
            "availability": list(n.availability.get(t.season, ())) if t.season in n.availability else None,
            "table_range": [min(n.availability, default=None), max(n.availability, default=None)],
            "exceptions": [e.model_dump(mode="json") for e in n.exceptions if e.season == t.season],
            "usable": bool(n.availability),
        }
        rules_slice = {
            "aliases": [[a.name, a.club] for a in self.rules.notes_club_aliases if a.season == t.season],
            "one_sided": [
                [r.rule_id, r.field, r.printed_on]
                for r in self.rules.one_sided
                if r.first_season <= t.season <= r.last_season
            ],
            "brownlow": [
                self.rules.brownlow_ref.sha256 if self.rules.brownlow_ref else None,
                self.rules.br_award_total(t.season),
                t.season in self.rules.no_award_seasons,
            ],
        }
        payload = {
            "layer": t.layer, "season": t.season, "matches": t.match_items, "absent": sorted(t.absent.items()),
            "profiles": t.profile_items, "status": sorted(t.profile_status.items()), "ids": sorted(t.local_ids.items()),
            "pairs": [[k[0], k[1], [game_to_json(g) for g in v]] for k, v in sorted(t.local_by_pair.items())],
            "profile_rows": [[k, [game_to_json(g) for g in v]] for k, v in sorted(t.local_by_profile.items())],
            "unplaced": [[game_to_json(g), why] for g, why in t.unplaced],
            "quarantine": [[k, [list(vars(q).values()) for q in v]] for k, v in sorted(t.quarantine.items())],
            "notes": notes_slice, "rules": rules_slice, "phash": self.phash, "emit": t.emit_source,
            "captured": len(self.captured),
            "scope": sorted(t.scope_matches), "local_excluded": t.local_excluded,
            "summary": [
                [u, c, sorted((k, str(v)) for k, v in vals.items())] for (u, c), vals in sorted(t.local_summary.items())
            ],
        }  # fmt: skip
        return hashlib_sha(_canon(payload))

    def season_units(self, only: str | None = None) -> None:
        """Season units of every requested layer, or of ``only`` (the audit runs one layer at a time so only one
        layer's club-season aggregates are ever held in memory before its reduce)."""
        if not hasattr(self, "profile_info"):
            self._profile_index()
        active = [n for n in LAYERS if self.layers.get(n) is not None and self.layers[n].identity is not None]
        emit = active[0] if active else None  # the first layer emits the source's own findings
        todo = [n for n in active if only is None or n == only]
        quarantine = self._quarantine_rows() if "snapshot" in todo else {}
        shard_dir = self._legacy_shards() if "legacy_csv" in todo else None
        specs: list[tuple[str, int, bool]] = [
            (layer, season, layer == emit) for layer in todo for season in self.seasons
        ]
        if not hasattr(self, "unit_digests"):
            self.unit_digests: dict[str, str] = {}
        shared = {
            "cache_root": str(self.cache_root),
            "phash": self.phash,
            "notes": self.notes,
            "rules": self.rules,
            "captured": self.captured,
            "chunk_dir": str(self.chunk_dir),
        }

        def make(spec: tuple[str, int, bool]) -> SeasonTask:
            task = self._build_task(spec[0], spec[1], spec[2], quarantine, shard_dir)
            self.unit_digests[f"{task.layer}:{task.season}"] = self._task_digest(task)
            return task

        ucache = UnitCache(self.cache_root, self._unit_salt())
        self.units_reused = getattr(self, "units_reused", 0)  # accumulates over the per-layer calls
        results: list[SeasonResult] = []

        def settle(spec: tuple[str, int, bool], digest: str, res: SeasonResult) -> SeasonResult:
            ucache.put(digest, res)
            return res

        def lookup(spec: tuple[str, int, bool], digest: str) -> SeasonResult | None:
            hit = ucache.get(digest, self.chunk_dir, f"{spec[0]}-{spec[1]:05d}")
            if hit is not None:
                self.units_reused += 1
            return hit

        if self.opts.workers <= 1 or len(specs) < 4:
            _init_worker(shared)
            for sp in specs:
                task = make(sp)
                digest = self.unit_digests[f"{task.layer}:{task.season}"]
                cached = lookup(sp, digest)
                results.append(cached if cached is not None else settle(sp, digest, run_season_task(task)))
        else:
            from collections import deque

            with ProcessPoolExecutor(
                max_workers=self.opts.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(shared,),
            ) as pool:
                # (spec, digest, ready result or future); tasks are built lazily so few hold local rows at once
                inflight: deque[tuple[tuple[str, int, bool], str, Any]] = deque()

                def drain_one() -> None:
                    sp0, dg0, item = inflight.popleft()
                    results.append(item if isinstance(item, SeasonResult) else settle(sp0, dg0, item.result()))

                for sp in specs:
                    task = make(sp)
                    digest = self.unit_digests[f"{task.layer}:{task.season}"]
                    cached = lookup(sp, digest)
                    inflight.append((sp, digest, cached if cached is not None else pool.submit(run_season_task, task)))
                    while len(inflight) > self.opts.workers + 1:
                        drain_one()
                while inflight:
                    drain_one()
        for (layer, season, _first), res in zip(specs, results, strict=True):
            self.layers[layer].season_results[season] = res
        self._tick("season_units")

    # -- phase 6: per-player aggregates ---------------------------------------------------------------

    def reduce_players(self, only: str | None = None) -> None:
        cache = FactsCache(self.cache_root, self.phash)
        emit_layer = next((n for n in LAYERS if n in self.layers and self.layers[n].identity is not None), None)
        for layer in LAYERS:
            st = self.layers.get(layer)
            if st is None or st.identity is None or (only is not None and layer != only):
                continue
            by_profile: dict[str, dict[tuple[str, int], ClubSeasonAgg]] = defaultdict(dict)
            for season in sorted(st.season_results):
                for (purl, club), agg in sorted(st.season_results[season].club_seasons.items()):
                    by_profile[purl][(club, agg.season)] = agg
            for purl in sorted(by_profile):
                info = self.profile_info.get(purl)
                core = FX.load_core(cache, info[0]) if info else None
                if core is None:
                    st.reduce_counters["profiles_core_unavailable"] += 1
                    continue
                locs = st.identity.profile_to_locals.get(purl)
                rr: ReduceResult = reduce_profile(
                    layer=layer,
                    purl=purl,
                    local_id=",".join(locs) if locs else None,
                    core=core,
                    aggs=by_profile[purl],
                    rules=self.rules,
                    emit_source=layer == emit_layer,
                    judge_local=locs is not None,
                    profile_sha256=info[0] if info else None,
                    career=not self._scoped,
                )
                st.reduce_counters.update(rr.counters)
                st.reduce_by_stat.update(rr.by_stat)
                for f in rr.findings:
                    (self.source_findings if f["layer"] == "source" else st.findings).append(f)
            by_profile.clear()
            for res in st.season_results.values():
                res.club_seasons.clear()  # the aggregates are spent: free them before the next layer
            gc.collect()
        self._tick("reduce")

    # -- phase 7: coverage, verdicts, report -------------------------------------------------------------

    def _parent_findings(self, st: LayerState) -> list[dict[str, Any]]:
        """Findings produced outside the season units: identity, matches, local inputs, aggregates."""
        out = list(st.findings)
        if st.match_result is not None:
            out.extend(st.match_result.findings)
        if st.identity is not None:
            for f in st.identity.findings:
                out.append(
                    make_finding(
                        f.category,
                        layer=st.name,
                        rule_id=f.rule_id,
                        player={"local_id": f.keys[0] if f.keys else None},
                        detail=f.detail,
                        extra_id=list(f.keys),
                    )
                )
            # one finding per (layer, profile), never one per season unit (F-H1)
            claimed = self._claimed(st.name)
            for purl in sorted(self.profiles_with_apps):
                if purl in st.identity.profile_to_locals or purl in claimed:
                    continue
                sha, name = self.profile_info.get(purl, (None, None))
                out.append(
                    make_finding(
                        "PLAYER_MISSING_LOCAL",
                        layer=st.name,
                        rule_id="R-PLAYER-MISSING",
                        player={"source_url": purl, "local_id": "", "name": name},
                        evidence={"source_url": purl, "body_sha256": sha, "locator": "profile"},
                        detail="no local player maps to this profile",
                    )
                )
        if not st.available:
            for p in st.problems:
                out.append(make_finding("LOCAL_INPUT_GAP", layer=st.name, rule_id="R-LOCAL-INPUT", detail=p))
        return out

    def _population(self, st: LayerState) -> dict[str, Any]:
        census = self.manifest.census
        res = self.manifest.resource_counts
        profiles_with_apps = self.profiles_with_apps
        pop: dict[str, Any] = {
            "source_profiles_in_directory": census.profiles_in_directory,
            "source_profiles_required": sum(res.get("profile", {}).values()),
            "source_profiles_usable": res.get("profile", {}).get("usable", 0),
            "source_profiles_with_in_scope_appearances": len(profiles_with_apps),
            "source_profiles_excluded_no_in_scope_appearance": len(self.captured - profiles_with_apps),
            "season_pages_required": sum(res.get("season", {}).values()),
            "season_pages_usable": res.get("season", {}).get("usable", 0),
            "match_pages_required": sum(res.get("match", {}).values()),
            "match_pages_usable": res.get("match", {}).get("usable", 0),
            "local_players": st.local_players,
            "local_games_read": st.local_games,
        }
        ident = st.identity
        if ident is not None:
            claimed = self._claimed(st.name)
            counts = Counter(r.status for r in ident.resolutions.values())
            pop.update(
                {
                    "local_players_resolved": counts.get("resolved", 0),
                    "local_players_alias": counts.get("alias", 0),
                    "local_players_unresolved": counts.get("unresolved", 0) + counts.get("ambiguous", 0),
                    "local_players_conflict": counts.get("conflict", 0),
                    "local_only_players": sum(
                        1 for r in ident.resolutions.values() if r.status == "unresolved" and not r.candidates
                    ),
                    "source_players_mapped": sum(1 for u in profiles_with_apps if u in ident.profile_to_locals),
                    "source_players_missing_locally": sum(
                        1 for u in profiles_with_apps if u not in ident.profile_to_locals and u not in claimed
                    ),
                    "source_players_unresolved": sum(
                        1 for u in profiles_with_apps if u not in ident.profile_to_locals and u in claimed
                    ),
                    "source_players_compared": sum(1 for u in profiles_with_apps if u in ident.profile_to_locals),
                }
            )
        return pop

    def _counters(self, st: LayerState) -> Counter[str]:
        c: Counter[str] = Counter()
        for season in sorted(st.season_results):
            c.update(st.season_results[season].counters)
        if st.match_result is not None:
            c.update(st.match_result.counters)
        c.update(st.reduce_counters)
        return c

    def _identities(self, c: Counter[str]) -> dict[str, bool]:
        buckets = (
            "cell_equal",
            "cell_mismatch",
            "cell_source_unavailable",
            "cell_not_applicable",
            "cell_not_applicable_dntf",
            "cell_unresolved",
            "cell_in_missing_appearance",
            "cell_in_unresolved_appearance",
        )
        states = (
            "RECORDED_VALUE",
            "RECORDED_ZERO",
            "NOT_RECORDED",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE_DNTF",
            "UNRESOLVED_BLANK",
            "MALFORMED",
            "SOURCE_CONFLICT",
        )
        ok = {
            "appearances_partition": c["app_expected"]
            == c["app_matched"] + c["app_missing_local"] + c["app_unresolved"],
            "cells_requested": c["cells_expected"] == len(S.STAT_FIELDS) * c["app_expected"],
            "cells_partition": c["cells_expected"] == sum(c[b] for b in buckets),
            "source_states_partition": c["cells_expected"] == sum(c[f"src_{s}"] for s in states),
            "stint_aggregates": c["agg_stint_judged"] == len(S.STAT_FIELDS) * c["agg_stint_units"],
            "career_aggregates": c["agg_career_judged"] == len(S.STAT_FIELDS) * c["agg_career_units"],
            "matches_source": c["source_matches_usable"] == c["matches_paired"] + c["matches_missing_local"],
            "matches_local": c["local_matches_in_scope"]
            == c["matches_paired"] + c["matches_extra_local"] + c["matches_unresolved"],
        }
        for level in ("stint", "season", "career"):
            outcomes = (
                "equal",
                "mismatch",
                "local_missing_summary",
                "source_unavailable",
                "not_applicable",
                "unresolved",
            )
            ok[f"{level}_outcomes_partition"] = c[f"agg_{level}_judged"] == sum(c[f"agg_{level}_{o}"] for o in outcomes)
        return ok

    def _drift(self) -> list[str]:
        out: list[str] = []
        if self.snap is not None:
            out.extend(self.snap.drift())
        if self.legacy_root is not None and self.plan.inputs.legacy is not None and self.legacy is not None:
            try:
                if pin_legacy(self.legacy_root) != self.plan.inputs.legacy:
                    out.append("legacy CSV files changed during the audit")
            except PlanError as exc:
                out.append(f"legacy CSV inventory failed after the audit: {exc}")
        for r in self.manifest.resources:
            if r.status == "usable" and r.sha256 and FX.object_bytes(self.capture_dir, r.sha256) is None:
                out.append(f"captured object {r.sha256[:12]} for {r.url} changed or vanished during the audit")
        return out

    def finalize(self) -> AuditResult:
        cache_stats = dict(self.cache_stats)
        cache_stats["units_reused_from_cache"] = getattr(self, "units_reused", 0)
        drift = self._drift()
        full = self.plan.scope.full_population
        layer_docs: dict[str, Any] = {}
        chunks: list[ChunkInfo] = [write_chunk(self.chunk_dir, "parent-source", self.source_findings)]
        for st0 in self.layers.values():
            chunks.append(write_chunk(self.chunk_dir, f"parent-{st0.name}", self._parent_findings(st0)))
            chunks.extend(r.chunk for _s, r in sorted(st0.season_results.items()) if r.chunk is not None)
        by_cat: Counter[tuple[str, str, str]] = Counter()
        by_player: Counter[tuple[str, str, str]] = Counter()
        for ch in chunks:
            by_cat.update(ch.by_cat)
            by_player.update(ch.by_player)
        players_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        any_fail = False
        any_unknown = False
        accounting_ok = True
        source_cats = Counter({cat: n for (lyr, cat, _sev), n in by_cat.items() if lyr == "source"})
        for name in LAYERS:
            st = self.layers.get(name)
            if st is None:
                continue
            c = self._counters(st)
            cats = Counter({(cat, sev): n for (lyr, cat, sev), n in by_cat.items() if lyr == name})
            fails = sum(n for (cat, sev), n in cats.items() if sev == "fail")
            unknown_reasons: list[str] = []
            if not st.available:
                unknown_reasons.append("the requested local layer could not be read as pinned")
            if not self.capture_complete:
                why = self.manifest.incomplete_reasons[:3] or ([self.receipt_problem] if self.receipt_problem else [])
                unknown_reasons.append("capture incomplete: " + "; ".join(why))
            if not full and self.plan.scope.population == "sample":
                # a sample can never pass; a seasons audit is complete for its declared seasons (the report's scope
                # says which, and full_population stays false)
                unknown_reasons.append("sample scope: not a full population")
            for label, n in (
                ("unresolved appearances", c["app_unresolved"]),
                ("unresolved cells", c["cell_unresolved"]),
                ("local numbers where the source records none", c["cell_unsupported_local_numeric"]),
                ("unresolved aggregates", c["agg_career_unresolved"]),
                ("unpaired matches", c["matches_unresolved"]),
                ("source conflicts", source_cats["SOURCE_CONFLICT"]),
                ("source schema gaps", source_cats["SCHEMA_GAP"]),
                ("source capture gaps", source_cats["CAPTURE_GAP"]),
            ):
                if n:
                    unknown_reasons.append(f"{n} {label}")
            id_unknown = sum(n for (cat, sev), n in cats.items() if cat.startswith("IDENTITY_") and sev == "unknown")
            if id_unknown:
                unknown_reasons.append(f"{id_unknown} unresolved or conflicting identities")
            if drift:
                unknown_reasons.append("input drift: " + "; ".join(drift[:2]))
            if c["local_unplaced"]:
                unknown_reasons.append(f"{c['local_unplaced']} local rows could not be attributed")
            identities = self._identities(c) if st.available and st.identity is not None else {}
            if identities and not all(identities.values()):
                accounting_ok = False
            verdict = S.Verdict.FAIL if fails else (S.Verdict.UNKNOWN if unknown_reasons else S.Verdict.PASS)
            any_fail = any_fail or verdict is S.Verdict.FAIL
            any_unknown = any_unknown or verdict is S.Verdict.UNKNOWN
            exp = c["cells_expected"]
            denom_avail = exp - c["cell_source_unavailable"] - c["cell_not_applicable"] - c["cell_not_applicable_dntf"]
            census_ok = not self.manifest.census.letters_failed and self.manifest.census.letters_usable == 26
            frac: dict[str, Any] = {
                "verified_numeric_fraction": (c["cell_equal"] / exp) if exp and census_ok else None,
                "available_statistic_fraction": (c["cell_equal"] / denom_avail)
                if denom_avail > 0 and census_ok
                else None,
                "null_reason": None if census_ok else "the source census is incomplete, so the denominator is unknown",
            }
            layer_docs[name] = {
                "verdict": verdict.value,
                "unknown_reasons": unknown_reasons,
                "finding_counts": {f"{cat}|{sev}": n for (cat, sev), n in sorted(cats.items())},
                "completeness": {
                    "execution_complete": st.available and st.identity is not None,
                    "capture_complete": self.capture_complete,
                    "identity_complete": id_unknown == 0 and c["app_unresolved"] == 0,
                    "comparison_complete": st.available
                    and self.capture_complete
                    and id_unknown == 0
                    and not unknown_reasons,
                    "source_consistent": source_cats["SOURCE_CONFLICT"] == 0
                    and source_cats["SOURCE_DERIVED_INCONSISTENCY"] == 0,
                },
                "population": self._population(st),
                "coverage": {k: v for k, v in sorted(c.items())},
                "accounting_identities": identities,
                "fractions": frac,
                "problems": st.problems,
            }
            self._rows(st, by_player, players_rows, coverage_rows)
        overall = (
            S.Verdict.FAIL if any_fail else (S.Verdict.UNKNOWN if any_unknown or not layer_docs else S.Verdict.PASS)
        )
        merged = self.work_dir / "findings.jsonl"
        n_findings, digest = merge_to_file(chunks, merged)
        report = self._report(layer_docs, overall, n_findings, digest, by_cat, drift)
        exit_code = 9 if not accounting_ok else S.EXIT_CODES[overall.value]
        exe = self._execution(cache_stats, drift, accounting_ok)
        return AuditResult(report, merged, n_findings, players_rows, coverage_rows, exe, exit_code)

    def _rows(
        self,
        st: LayerState,
        findings_by_player: Counter[tuple[str, str, str]],
        players: list[dict[str, Any]],
        coverage: list[dict[str, Any]],
    ) -> None:
        by_player: dict[str, Counter[str]] = defaultdict(Counter)
        for (lyr, url, sev), n in findings_by_player.items():
            if lyr == st.name:
                by_player[url][sev] += n
        ident = st.identity
        merged: dict[str, list[int]] = {}
        for season in sorted(st.season_results):
            for purl, season_cnt in st.season_results[season].by_profile.items():
                row = merged.setdefault(purl, [0] * len(PROFILE_KEYS))
                for i, v in enumerate(season_cnt):
                    row[i] += v
        claimed = self._claimed(st.name)
        for purl in sorted(self.captured):
            found = merged.get(purl)
            if not found and purl not in by_player:
                continue
            cnt: Counter[str] = Counter(dict(zip(PROFILE_KEYS, found, strict=True))) if found else Counter()
            locs = ident.profile_to_locals.get(purl) if ident else None
            players.append(
                {
                    "layer": st.name,
                    "source_url": purl,
                    "name": self.profile_info[purl][1] or "",
                    "local_ids": ",".join(locs) if locs else "",
                    "status": "mapped" if locs else ("unresolved" if purl in claimed else "missing"),
                    "appearances_expected": cnt["app_expected"],
                    "appearances_matched": cnt["app_matched"],
                    "appearances_missing": cnt["app_missing_local"],
                    "appearances_unresolved": cnt["app_unresolved"],
                    "appearances_local_only": cnt["app_local_only"],
                    "cells_expected": cnt["cells_expected"],
                    "cells_equal": cnt["cell_equal"],
                    "cells_mismatch": cnt["cell_mismatch"],
                    "cells_source_unavailable": cnt["cell_source_unavailable"],
                    "cells_not_applicable": cnt["cell_not_applicable"] + cnt["cell_not_applicable_dntf"],
                    "cells_unresolved": cnt["cell_unresolved"],
                    "findings_fail": by_player[purl]["fail"],
                    "findings_unknown": by_player[purl]["unknown"],
                }
            )
        if ident is not None:
            for key in sorted(ident.resolutions):
                res = ident.resolutions[key]
                if res.status in ("unresolved", "ambiguous", "conflict"):
                    players.append(
                        {
                            "layer": st.name, "source_url": "", "name": "", "local_ids": key, "status": res.status,
                            "findings_unknown": 1,
                        }
                    )  # fmt: skip
        for season in sorted(st.season_results):
            r = st.season_results[season]
            for k, v in sorted(r.counters.items()):
                coverage.append({"layer": st.name, "season": season, "statistic": "", "metric": k, "count": v})
            for (fld, bucket), v in sorted(r.by_stat.items()):
                coverage.append({"layer": st.name, "season": season, "statistic": fld, "metric": bucket, "count": v})
        for (level, fld, outcome), v in sorted(st.reduce_by_stat.items()):
            coverage.append(
                {"layer": st.name, "season": "", "statistic": fld, "metric": f"agg_{level}_{outcome}", "count": v}
            )

    def _latest_final(self) -> dict[str, Any] | None:
        if self.latest_final is None:
            return None
        d, url, stage = self.latest_final
        info = self.fixture_by_url[url]
        doc: dict[str, Any] = {
            "source_url": url,
            "season": _year_of(url),
            "date": d,
            "stage": stage,
            "teams": [info[1], info[2]],
            "present_locally": {},
            "participants": {},
        }
        cache = FactsCache(self.cache_root, self.phash)
        sha = self.usable_matches.get(url)
        facts = FX.load_match(cache, sha) if sha else None
        links = sorted({p.link for p in facts.players if p.link}) if facts else []
        doc["participant_profiles"] = len(links)
        for name in LAYERS:
            st = self.layers.get(name)
            if st is None or st.match_result is None:
                continue
            doc["present_locally"][name] = url in st.match_result.mapping.source_to_local
            tracked: dict[tuple[str, str], str] = {}
            for res in st.season_results.values():
                tracked.update(res.tracked)
            outcome = Counter(tracked.get((u, url), "profile_not_compared") for u in links)
            doc["participants"][name] = dict(sorted(outcome.items()))
        return doc

    def _report(
        self,
        layers: dict[str, Any],
        overall: S.Verdict,
        n_findings: int,
        findings_digest: str,
        by_cat: Counter[tuple[str, str, str]],
        drift: list[str],
    ) -> dict[str, Any]:
        o = self.opts
        scope = self.plan.scope.model_dump(mode="json")
        scope["seasons_with_source_pages"] = len(self.seasons)
        if not self._scoped:
            scope.pop("seasons", None)  # appended field: absent from a full audit's report, as before
        scope["matches_in_scope"] = len(self.scope_urls)
        scope["matches_excluded_after_boundary"] = self.source_counters["matches_excluded_after_boundary"]
        scope["reference_mode"] = "observed_current"
        scope["acquisition_window_utc"] = [
            self.manifest.acquisition_started_utc,
            self.manifest.acquisition_finished_utc,
        ]
        return {
            "kind": "afltables-reconciliation-report",
            "schema_version": S.SCHEMA_VERSION if hasattr(S, "SCHEMA_VERSION") else 1,
            "plan_id": self.plan.plan_id,
            "capture_identity": self.plan.capture_identity,
            "capture_manifest_sha256": hashlib_sha(o.capture_manifest.read_bytes()),
            "snapshot_id": self.plan.inputs.snapshot.snapshot_id,
            "legacy_inputs": self.plan.inputs.legacy.model_dump(mode="json") if self.plan.inputs.legacy else None,
            "scope": scope,
            "policies": self.plan.policies.model_dump(mode="json"),
            "code": {"capture_files": self.plan.code.capture_files, "parser_identity": self.phash},
            "result": {"overall": overall.value, "layers": {k: v["verdict"] for k, v in layers.items()}},
            "layers": layers,
            "source": {
                "finding_counts": {
                    f"{cat}": n
                    for cat, n in sorted(
                        Counter({c: n for (lyr, c, _sev), n in by_cat.items() if lyr == "source"}).items()
                    )
                },
                "counters": dict(sorted(self.source_counters.items())),
                "census": self.manifest.census.model_dump(mode="json"),
                "manifest_resource_counts": self.manifest.resource_counts,
                "capture_complete": self.capture_complete,
                "capture_incomplete_reasons": self.manifest.incomplete_reasons
                + ([self.receipt_problem] if self.receipt_problem else []),
            },
            "latest_completed_final": self._latest_final(),
            "findings": {
                "count": n_findings,
                "sha256": findings_digest,
                "by_layer_category_severity": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(by_cat.items())},
                "stream": "findings.jsonl",
            },
            "input_drift": drift,
            "unit_digests": dict(sorted(self.unit_digests.items())),
            "exclusions": list(S.EXCLUSIONS),
            "limitations": [
                "AFL Tables states its figures are unofficial and may contain errors: agreement is with captured "
                "AFL Tables evidence, not independent proof of historical truth.",
                "The audit is a reference_mode=observed_current capture: pages were retrieved over the acquisition "
                "window, not as a snapshot of the --through-date.",
                "No correction is applied; findings describe differences only.",
            ],
        }

    def _execution(self, cache_stats: dict[str, int], drift: list[str], accounting_ok: bool) -> dict[str, Any]:
        ru_self = resource.getrusage(resource.RUSAGE_SELF)
        ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
        return {
            "timings_s": self.timings,
            "peak_rss_mib_after_phase": self.rss_after,
            "workers": self.opts.workers,
            "parse_cache": cache_stats,
            "facts_cache": self.match_facts_cache.stats() if hasattr(self, "match_facts_cache") else {},
            "peak_rss_mib_self": round(ru_self.ru_maxrss / 1024, 1),
            "peak_rss_mib_largest_child": round(ru_kids.ru_maxrss / 1024, 1),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "accounting_identities_hold": accounting_ok,
            "input_drift": drift,
            "cache_root": str(self.cache_root),
            "previous": getattr(self, "previous_accounting", None),
        }


def hashlib_sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


@dataclass
class AuditResult:
    report: dict[str, Any]
    findings_path: Path
    findings_count: int
    players: list[dict[str, Any]]
    coverage: list[dict[str, Any]]
    execution: dict[str, Any]
    exit_code: int

    @property
    def findings(self) -> list[dict[str, Any]]:
        """The merged findings stream as records (tests and small audits; the writer streams the file)."""
        with self.findings_path.open("rb") as fh:
            return [json.loads(line) for line in fh]


def input_roots_of(plan: S.Plan, capture_dir: Path) -> list[Path]:
    roots = [Path(plan.operational.data_root).resolve(), capture_dir.resolve()]
    if plan.operational.legacy_root:
        base = Path(plan.operational.legacy_root)
        roots += [(base / "data" / "player_data").resolve(), (base / "data" / "matches").resolve()]
        if (base / "data" / "awards").is_dir():
            roots.append((base / "data" / "awards").resolve())
    return roots


def cleanup(audit: Audit) -> None:
    import shutil

    shutil.rmtree(audit.work_dir, ignore_errors=True)


def run_audit(opts: CompareOptions) -> tuple[AuditResult, Audit]:
    audit = Audit(opts)
    audit.load()
    audit.parse()
    audit.inventory()
    audit.identity_snapshot()
    audit.identity_legacy()
    audit.release_parse_state()
    for layer in LAYERS:  # one layer at a time: its aggregates are reduced and freed before the next layer's
        audit.season_units(layer)
        audit.reduce_players(layer)
    result = audit.finalize()
    if opts.previous is not None:
        from supercoach_via.reconciliation.report import read_previous_units

        prev = read_previous_units(opts.previous)
        if prev is None:
            result.execution["previous"] = {"error": "previous report unreadable: every unit recomputed"}
        else:
            same = sorted(k for k, v in audit.unit_digests.items() if prev.get(k) == v)
            result.execution["previous"] = {
                "units_total": len(audit.unit_digests),
                "units_unchanged_reused_evidence": len(same),
                "units_changed_or_new": sorted(k for k in audit.unit_digests if k not in same),
                "units_removed": sorted(k for k in prev if k not in audit.unit_digests),
            }
    result.execution["timings_s"] = audit.timings
    return result, audit


def run_compare_cli(
    *,
    plan: Path,
    capture_manifest: Path,
    out: Path,
    workers: int,
    cache: Path | None,
    previous: Path | None,
    json_out: bool,
) -> int:
    """CLI body: returns the process exit code (0 PASS, 2 invalid, 4 FAIL, 8 UNKNOWN, 9 software failure)."""
    import sys

    from supercoach_via.reconciliation.report import OutputWriteError, write_report_dir

    opts = CompareOptions(
        plan=plan, capture_manifest=capture_manifest, out=out, workers=workers, cache=cache, previous=previous
    )
    try:
        result, audit = run_audit(opts)
        try:
            manifest = write_report_dir(out, result, input_roots_of(audit.plan, audit.capture_dir))
        finally:
            cleanup(audit)  # the work directory never outlives the run, written or refused (B4)
    except (CompareError, PlanError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OutputWriteError as exc:
        print(f"error: {exc}; written: {exc.written or 'none'}; not written: {exc.not_written}", file=sys.stderr)
        return 2 if not exc.written else 9
    doc = {
        "overall": result.report["result"]["overall"],
        "layers": result.report["result"]["layers"],
        "report_sha256": manifest["report_sha256"],
        "findings": result.report["findings"]["count"],
        "out": str(out),
    }
    print(json.dumps(doc, sort_keys=True) if json_out else "\n".join(f"{k}: {v}" for k, v in doc.items()))
    return result.exit_code
