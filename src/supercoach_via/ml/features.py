"""The sole historical and prospective feature builder (PLAN 7.2).

Temporal policy (``features_v1``)
---------------------------------
An observation is a canonical ``player_games`` row. It is *eligible history* only when:

1. its time is verifiable: either its own ``date_quality`` is ``fixture_verified`` /
   ``source``, or it is linked to its match by the deterministic source key
   (``link_method=key``). The row's own ``match_date`` is NEVER used as the event time:
   in the legacy corpus it is a synthesized date (``inferred``, median ~17 days off the
   match), so the linked match record supplies the event date. Rows with an
   ``inferred``/``unknown`` date AND a weaker link (``date_tiebreak``/``row_order``) are
   not time-verified and are excluded from every feature and every training target
   (``excluded_unverified_date`` in ``FeatureFrame.diagnostics``);
2. its match exists in ``matches`` with ``status=complete`` and a non-null ``match_date``.
   The match's local calendar date is the authoritative event date;
3. it is strictly before the target's ``forecast_cutoff``:
   * ``event_date < cutoff_day`` where ``cutoff_day`` is the cutoff's **UTC** calendar
     date. Because every AFL venue has a non-negative UTC offset, a game on local date
     ``d`` has ended before ``(d+1) 00:00Z``, so the rule never admits a game that had not
     finished (it is conservative, never leaky);
   * a same-day observation (``event_date == cutoff_day``) is admitted only when its
     start is known to the minute *and* its venue has a verified IANA timezone *and*
     ``start_utc + game_hours <= cutoff``; otherwise the same-day order is unknown and the
     observation is excluded (``excluded_same_day_ambiguous``);
   * when an archived ``available_at`` exists it must also be ``<= cutoff``; such rows
     enter the ordered history on the later of event date and availability date.
4. Ties on the same date are broken by the stable ``match_id``.

Target rows carry an explicit ``forecast_cutoff``; a cutoff after the target match's
start (or after its local date when the start time is unknown) is rejected. Target stats
are never read: feature values are functions of eligible history only, so a completed
historical row and a scheduled future row with the same identity/cutoff produce
identical features (M05).

Missingness indicators (``<feature>__missing``) are derived here, before any imputation,
by the same function for training and inference. Imputation and categorical encoding
are fitted later, inside each training fold's estimator pipeline (``ml.train``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.storage.queries import SnapshotQuery
from supercoach_via.storage.snapshots import load_snapshot

FEATURE_CODE_VERSION = "features_v1"

#: The legacy predictor's six base rolling stats (supercoach/prediction.py
#: ``base_rolling_features``). Clearances and inside-50s are recorded from 1998 only; for
#: earlier history they are missing (NaN + indicator), never zero.
BASE_STATS: tuple[str, ...] = (
    "disposals",
    "kicks",
    "handballs",
    "tackles",
    "clearances",
    "inside_50s",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "player_id",
    "club_id",
    "match_id",
    "season",
    "stage_id",
    "stage_label",
    "stage_type",
    "stage_order",
    "match_date",
    "opponent_club_id",
    "venue_id",
    "forecast_cutoff",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("club_id", "opponent_club_id", "venue_id")
MISSING_SUFFIX = "__missing"
_DAY_BITS = 20  # date ordinals (~740k) fit in 20 bits


class TargetCutoffError(ValueError):
    """A target's forecast cutoff is not strictly before the target match."""


class FeatureOrderError(RuntimeError):
    """History ordering violated an invariant the feature algorithm relies on."""


@dataclass(frozen=True)
class FeatureSpec:
    version: str = FEATURE_CODE_VERSION
    base_stats: tuple[str, ...] = BASE_STATS
    window_long: int = 5
    window_season: int = 3
    ewm_span: int = 3
    verified_date_qualities: tuple[str, ...] = ("fixture_verified", "source")
    verified_link_methods: tuple[str, ...] = ("key",)
    include_age: bool = True
    age_birth_qualities: tuple[str, ...] = ("source",)
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES
    same_day_game_hours: float = 4.0

    def fingerprint(self) -> str:
        """Hash of the spec plus this module's source (feature code version)."""
        code = Path(__file__).read_bytes()
        payload = json.dumps(asdict(self), sort_keys=True).encode() + hashlib.sha256(code).digest()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class History:
    """In-memory canonical tables needed by the ML layer (one identified snapshot)."""

    snapshot_id: str
    matches: pd.DataFrame
    player_games: pd.DataFrame
    players: pd.DataFrame
    clubs: pd.DataFrame
    venues: pd.DataFrame
    lineups: pd.DataFrame
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: derived-array cache; init=False so dataclasses.replace() never carries a stale cache
    cache: dict[Any, Any] = field(default_factory=dict, init=False, repr=False, compare=False)


_MATCH_COLS = (
    "match_id", "season", "stage_label", "stage_type", "round_number", "stage_order",
    "stage_id", "replay_occurrence", "home_club_id", "away_club_id", "venue_id",
    "local_start", "match_date", "date_precision", "status", "venue_source_name",
    "home_final_goals", "home_final_behinds", "home_score",
    "away_final_goals", "away_final_behinds", "away_score",
)
_PG_COLS = (
    "match_id", "player_id", "club_id", "season", "opponent_club_id", "stage_label",
    "match_date", "date_quality", "link_method", "career_game_counter", *PLAYER_STAT_COLUMNS, "available_at",
)


def _to_date(s: pd.Series) -> pd.Series:
    return s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT
                 else (v.date() if isinstance(v, datetime | pd.Timestamp) else v))


def load_history(data_root: Path, selector: str = "current", *, verify: bool = False,
                 extra_tables: tuple[str, ...] = ()) -> History:
    """Load the tables the ML layer needs from an identified snapshot."""
    manifest = load_snapshot(data_root, selector, verify=verify)
    wanted = {"matches", "player_games", "players", "clubs", "venues", "lineups", *extra_tables}
    frames: dict[str, pd.DataFrame] = {}
    with SnapshotQuery(data_root, manifest, tables=wanted) as q:
        for name in sorted(wanted):
            if name not in manifest.tables:
                frames[name] = pd.DataFrame()
                continue
            if name == "matches":
                sql = f"select {', '.join(_MATCH_COLS)} from matches"  # noqa: S608 (fixed allowlist)
            elif name == "player_games":
                sql = f"select {', '.join(_PG_COLS)} from player_games"  # noqa: S608
            else:
                sql = f'select * from "{name}"'  # noqa: S608 (name from fixed set)
            frames[name] = q.arrow(sql).to_pandas()
    return _normalise(History(
        snapshot_id=manifest.snapshot_id,
        matches=frames["matches"],
        player_games=frames["player_games"],
        players=frames["players"],
        clubs=frames["clubs"],
        venues=frames["venues"],
        lineups=frames["lineups"],
        extra={k: frames[k] for k in extra_tables},
    ))


def _normalise(h: History) -> History:
    m = h.matches.copy()
    if len(m):
        m["match_date"] = _to_date(m["match_date"])
        m = m.astype({"match_id": str})
    pg = h.player_games.copy()
    if len(pg):
        pg["match_date"] = _to_date(pg["match_date"])
        if "available_at" in pg:
            pg["available_at"] = pd.to_datetime(pg["available_at"], utc=True)
    return History(h.snapshot_id, m, pg, h.players, h.clubs, h.venues, h.lineups, h.extra)


def time_verified(pg: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Rows whose event time can be taken from their linked match (module policy 1)."""
    ok = pg["date_quality"].isin(spec.verified_date_qualities)
    if "link_method" in pg:
        ok = ok | pg["link_method"].isin(spec.verified_link_methods)
    return ok


def match_day_cutoff(day: date) -> datetime:
    """Default cutoff for a target on local date ``day``: 00:00 UTC that day.

    00:00Z is 08:00-11:00 local across Australian venues, before any AFL start time, and
    makes all of the previous local day eligible (see module policy).
    """
    return datetime.combine(day, time(0), UTC)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def historical_targets(
    history: History,
    *,
    seasons: tuple[int, ...] | None = None,
    spec: FeatureSpec | None = None,
) -> pd.DataFrame:
    """Target rows for completed, time-verified player-games (training/holdout/replay).

    Each row gets ``forecast_cutoff = match_day_cutoff(match_date)``. Returns only the
    identity/fixture columns in ``TARGET_COLUMNS`` — never the row's own statistics.
    """
    spec = spec or FeatureSpec()
    pg = history.player_games
    m = history.matches
    if seasons is not None:
        pg = pg[pg["season"].isin(seasons)]
    pg = pg[time_verified(pg, spec)]
    cols = ["match_id", "stage_id", "stage_type", "stage_order", "venue_id", "match_date", "status",
            "stage_label"]
    j = pg[["player_id", "club_id", "match_id", "season", "opponent_club_id"]].merge(
        m[cols], on="match_id", how="inner", validate="many_to_one"
    )
    j = j[(j["status"] == "complete") & j["match_date"].notna()]
    j["forecast_cutoff"] = pd.to_datetime(j["match_date"].map(match_day_cutoff), utc=True)
    out = j[list(TARGET_COLUMNS)].sort_values(["match_date", "match_id", "player_id"], kind="stable")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureFrame:
    keys: pd.DataFrame  # TARGET_COLUMNS, aligned to X
    X: pd.DataFrame  # numeric features (+ indicators) then categoricals, fixed order
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    history_games: np.ndarray
    spec: FeatureSpec
    snapshot_id: str
    diagnostics: dict[str, int]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.numeric_features, *self.categorical_features)

    def dtypes(self) -> dict[str, str]:
        return {str(c): str(t) for c, t in self.X.dtypes.items()}


def numeric_feature_names(spec: FeatureSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(value features, features that get a missingness indicator)."""
    stat_feats = [
        f"{s}_{kind}"
        for s in spec.base_stats
        for kind in ("prior5_mean", "season3_mean", "season_mean", "ewm3")
    ]
    nullable = [*stat_feats, "tog_last", "tog_prior5_mean", "days_since_last_game",
                "career_counter_last"]
    if spec.include_age:
        nullable.append("age_years")
    always = ["history_games", "season_games_prior", "is_final", "stage_order", "season"]
    return (*nullable, *always), tuple(nullable)


def _utc_start(local_start: Any, tz: Any) -> float:
    """UTC epoch seconds of a minute-precision local start; NaN when unverifiable."""
    if not isinstance(local_start, str) or not isinstance(tz, str) or not tz:
        return np.nan
    try:
        naive = datetime.strptime(local_start, "%Y-%m-%d %H:%M")
        return naive.replace(tzinfo=ZoneInfo(tz)).timestamp()
    except (ValueError, KeyError):
        return np.nan


def _venue_tz(history: History) -> pd.Series:
    v = history.venues
    if len(v) and "timezone" in v:
        return v.set_index("venue_id")["timezone"]
    return pd.Series(dtype=object)


def _start_epochs(local_start: pd.Series, precision: pd.Series, venue: pd.Series, tz: pd.Series) -> np.ndarray:
    """UTC start epochs where minute precision AND a verified venue tz exist, else NaN."""
    zone = venue.map(tz)
    out = np.full(len(local_start), np.nan)
    ok = (precision == "minute").to_numpy() & zone.notna().to_numpy() & local_start.notna().to_numpy()
    for i in np.flatnonzero(ok):
        out[i] = _utc_start(local_start.iloc[i], zone.iloc[i])
    return out


def _prepare_observations(history: History, spec: FeatureSpec) -> tuple[pd.DataFrame, dict[str, int]]:
    diag: dict[str, int] = {}
    pg = history.player_games
    diag["observations_total"] = len(pg)
    ok_q = time_verified(pg, spec)
    diag["excluded_unverified_date"] = int((~ok_q).sum())
    pg = pg[ok_q]
    m = history.matches
    mm = m[["match_id", "match_date", "status", "local_start", "date_precision", "venue_id"]].rename(
        columns={"match_date": "event_date"}
    )
    obs = pg.merge(mm, on="match_id", how="left", validate="many_to_one")
    ok_m = (obs["status"] == "complete") & obs["event_date"].notna()
    diag["excluded_no_complete_dated_match"] = int((~ok_m).sum())
    obs = obs[ok_m].reset_index(drop=True)
    diag["date_disagrees_with_match"] = int(
        (obs["match_date"].notna() & (obs["match_date"] != obs["event_date"])).sum()
    )
    start = _start_epochs(obs["local_start"], obs["date_precision"], obs["venue_id"], _venue_tz(history))
    obs["end_epoch"] = start + spec.same_day_game_hours * 3600.0
    obs["event_ord"] = np.fromiter((d.toordinal() for d in obs["event_date"]), dtype=np.int64, count=len(obs))
    if "available_at" in obs and obs["available_at"].notna().any():
        av = pd.to_datetime(obs["available_at"], utc=True)
        obs["avail_epoch"] = av.map(lambda x: x.timestamp() if pd.notna(x) else np.nan).astype(float)
        # enters ordered history on the later of event date and (availability date + 1)
        av_ord = av.map(lambda t: (t.date() + timedelta(days=1)).toordinal() if pd.notna(t) else 0)
        obs["eff_ord"] = np.maximum(obs["event_ord"].to_numpy(), av_ord.to_numpy(dtype=np.int64))
    else:
        obs["avail_epoch"] = np.nan
        obs["eff_ord"] = obs["event_ord"]
    obs = obs.sort_values(["player_id", "eff_ord", "event_ord", "match_id"], kind="stable")
    return obs.reset_index(drop=True), diag


@dataclass(frozen=True)
class _Prepared:
    """Target-independent history arrays (computed once per History and spec)."""

    obs: pd.DataFrame
    diag: dict[str, int]
    players: pd.Index
    o_key: np.ndarray
    s_key: np.ndarray
    o_eff: np.ndarray
    o_event: np.ndarray
    match_ids: np.ndarray
    end_epoch: np.ndarray
    avail: np.ndarray
    cums: dict[str, tuple[np.ndarray, np.ndarray]]
    last: dict[str, np.ndarray]  # value at each observation (NaN if missing)
    ewm: dict[str, np.ndarray]  # EWM state after each observation


def _ewm_states(values: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    """State after each observation (adjust=False, missing values skipped, reset per player)."""
    if not len(values):
        return np.array([], dtype=float)
    s = pd.Series(values)
    out = s.groupby(groups, sort=False).ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    arr = out.reset_index(level=0, drop=True).sort_index().to_numpy(dtype=float)
    # groupby-ewm carries the last state forward over missing values once started, matching
    # the definition; positions before the first observed value stay NaN.
    return arr


def _prepare(history: History, spec: FeatureSpec) -> _Prepared:
    key = (spec.version, spec.base_stats, spec.ewm_span, spec.verified_date_qualities,
           spec.verified_link_methods, spec.same_day_game_hours)
    cached = history.cache.get(key)
    if isinstance(cached, _Prepared):
        return cached
    obs, diag = _prepare_observations(history, spec)
    players = pd.Index(sorted(set(obs["player_id"])))
    o_pc = players.get_indexer(pd.Index(obs["player_id"])).astype(np.int64)
    o_eff = obs["eff_ord"].to_numpy(dtype=np.int64)
    o_key = (o_pc << _DAY_BITS) | o_eff
    if len(o_key) > 1 and (np.diff(o_key) < 0).any():
        raise FeatureOrderError("observation keys not sorted")
    s_key = (o_pc << 16) | (obs["season"].to_numpy(dtype=np.int64) - 1800)
    if len(s_key) > 1 and (np.diff(s_key) < 0).any():
        raise FeatureOrderError("season order disagrees with date order for a player")
    alpha = 2.0 / (spec.ewm_span + 1.0)
    cums: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    last: dict[str, np.ndarray] = {}
    ewm: dict[str, np.ndarray] = {}
    for col in (*spec.base_stats, "time_on_ground_pct", "career_game_counter"):
        v = obs[col].to_numpy(dtype=float, na_value=np.nan)
        last[col] = v
        cums[col] = (np.concatenate([[0.0], np.cumsum(np.nan_to_num(v, nan=0.0))]),
                     np.concatenate([[0], np.cumsum(~np.isnan(v))]))
        if col in spec.base_stats:
            ewm[col] = _ewm_states(v, o_pc, alpha)
    prep = _Prepared(
        obs=obs, diag=diag, players=players, o_key=o_key, s_key=s_key, o_eff=o_eff,
        o_event=obs["event_ord"].to_numpy(dtype=np.int64), match_ids=obs["match_id"].to_numpy(dtype=object),
        end_epoch=obs["end_epoch"].to_numpy(dtype=float), avail=obs["avail_epoch"].to_numpy(dtype=float),
        cums=cums, last=last, ewm=ewm,
    )
    history.cache[key] = prep
    return prep


def _window_mean(cs_v: np.ndarray, cs_c: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    s = cs_v[hi] - cs_v[lo]
    c = cs_c[hi] - cs_c[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.where(c > 0, c, 1), np.nan)


def build_features(history: History, targets: pd.DataFrame, spec: FeatureSpec | None = None) -> FeatureFrame:
    """Build the feature matrix for explicit target rows (see module policy)."""
    spec = spec or FeatureSpec()
    missing = [c for c in TARGET_COLUMNS if c not in targets.columns]
    if missing:
        raise ValueError(f"targets missing columns {missing}")
    t = targets[list(TARGET_COLUMNS)].reset_index(drop=True).copy()
    t["match_date"] = _to_date(t["match_date"])
    cutoff = pd.to_datetime(t["forecast_cutoff"], utc=True)
    if cutoff.isna().any():
        raise TargetCutoffError("every target needs an explicit forecast_cutoff")
    t["forecast_cutoff"] = cutoff
    cutoff_epoch = cutoff.map(lambda c: c.timestamp()).to_numpy(dtype=float)
    cutoff_ord = cutoff.map(lambda c: c.date().toordinal()).to_numpy(dtype=np.int64)
    tgt_ord = t["match_date"].map(lambda d: d.toordinal() if d is not None else -1).to_numpy(dtype=np.int64)
    if (tgt_ord < 0).any():
        raise TargetCutoffError("target rows need a known match_date")
    if (cutoff_ord > tgt_ord).any():
        raise TargetCutoffError("forecast_cutoff falls after the target match date")
    _check_target_start(history, t, cutoff_epoch)

    P = _prepare(history, spec)
    diag = dict(P.diag)
    n = len(P.o_key)
    t_pc = P.players.get_indexer(pd.Index(t["player_id"])).astype(np.int64)
    known = t_pc >= 0
    pc = np.where(known, t_pc, 0)
    seg_start = np.where(known, np.searchsorted(P.o_key, pc << _DAY_BITS, "left"), 0)
    seg_end = np.where(known, np.searchsorted(P.o_key, (pc + 1) << _DAY_BITS, "left"), 0)
    k = np.where(known, np.searchsorted(P.o_key, (pc << _DAY_BITS) | cutoff_ord, "left"), 0)

    # same-day admission: minute-precision start with verified venue tz, ended before cutoff
    t_mid = t["match_id"].to_numpy(dtype=object)
    ambiguous = 0
    for _ in range(4):
        cand = k < seg_end
        if not cand.any():
            break
        idx = np.where(cand, k, 0)
        same = cand & (P.o_event[idx] == cutoff_ord) & (P.o_eff[idx] == cutoff_ord)
        ok = same & ~np.isnan(P.end_epoch[idx]) & (P.end_epoch[idx] <= cutoff_epoch) & (
            np.isnan(P.avail[idx]) | (P.avail[idx] <= cutoff_epoch)
        )
        # the target's own match is never "history", so it is not an ambiguity
        ambiguous += int((same & ~ok & (P.match_ids[idx] != t_mid)).sum())
        if not ok.any():
            break
        k = np.where(ok, k + 1, k)
    diag["excluded_same_day_ambiguous"] = ambiguous
    # rows with an availability stamp enter on (availability date + 1) > their stamp, so
    # the day-prefix rule already guarantees available_at < cutoff for in-window rows.
    hist = (k - seg_start).astype(np.int64)
    t_season = t["season"].to_numpy(dtype=np.int64)
    j0 = np.where(known, np.searchsorted(P.s_key, (pc << 16) | (t_season - 1800), "left"), 0)
    j0 = np.clip(j0, seg_start, k)
    season_n = k - j0

    lo5 = np.maximum(seg_start, k - spec.window_long)
    lo3 = np.maximum(j0, k - spec.window_season)
    has_prev = k > seg_start
    prev = np.where(has_prev, k - 1, 0)

    def at_prev(arr: np.ndarray) -> np.ndarray:
        if not n:
            return np.full(len(t), np.nan)
        return np.where(has_prev, arr[prev], np.nan)

    feats: dict[str, np.ndarray] = {}
    for s in spec.base_stats:
        cs_v, cs_c = P.cums[s]
        feats[f"{s}_prior5_mean"] = _window_mean(cs_v, cs_c, lo5, k)
        feats[f"{s}_season3_mean"] = _window_mean(cs_v, cs_c, lo3, k)
        feats[f"{s}_season_mean"] = _window_mean(cs_v, cs_c, j0, k)
        feats[f"{s}_ewm3"] = at_prev(P.ewm[s])
    feats["tog_last"] = at_prev(P.last["time_on_ground_pct"])
    cs_v, cs_c = P.cums["time_on_ground_pct"]
    feats["tog_prior5_mean"] = _window_mean(cs_v, cs_c, lo5, k)
    feats["days_since_last_game"] = (cutoff_ord - at_prev(P.o_event.astype(float))).astype(float)
    feats["career_counter_last"] = at_prev(P.last["career_game_counter"])
    if spec.include_age:
        feats["age_years"] = _age(history, t, cutoff_ord, spec)
    feats["history_games"] = hist.astype(float)
    feats["season_games_prior"] = season_n.astype(float)
    feats["is_final"] = (t["stage_type"] == "final").to_numpy(dtype=float)
    feats["stage_order"] = t["stage_order"].to_numpy(dtype=float)
    feats["season"] = t_season.astype(float)

    names, nullable = numeric_feature_names(spec)
    X = pd.DataFrame({c: feats[c] for c in names})
    for c in nullable:
        X[c + MISSING_SUFFIX] = np.isnan(feats[c]).astype(float)
    numeric = (*names, *(c + MISSING_SUFFIX for c in nullable))
    for c in spec.categorical:
        X[c] = [None if pd.isna(v) else str(v) for v in t[c]]
    diag["targets"] = len(t)
    return FeatureFrame(
        keys=t,
        X=X,
        numeric_features=tuple(numeric),
        categorical_features=tuple(spec.categorical),
        history_games=hist,
        spec=spec,
        snapshot_id=history.snapshot_id,
        diagnostics=diag,
    )


def _check_target_start(history: History, t: pd.DataFrame, cutoff_epoch: np.ndarray) -> None:
    """When a target's start is known to the minute (with venue tz), cutoff must precede it."""
    m = history.matches
    if not len(m):
        return
    info = m.set_index("match_id")[["local_start", "date_precision", "venue_id"]]
    j = t[["match_id"]].join(info, on="match_id")
    st = _start_epochs(j["local_start"], j["date_precision"], j["venue_id"], _venue_tz(history))
    bad = ~np.isnan(st) & (cutoff_epoch > st)
    if bad.any():
        raise TargetCutoffError(f"cutoff after start of target match {t['match_id'].iloc[int(np.argmax(bad))]}")


def _age(history: History, t: pd.DataFrame, cutoff_ord: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    p = history.players
    if not len(p) or "birth_date" not in p:
        return np.full(len(t), np.nan)
    ok = p[p["birth_date_quality"].isin(spec.age_birth_qualities) & p["birth_date"].notna()]
    bd = dict(zip(ok["player_id"], _to_date(ok["birth_date"]), strict=True))
    out = np.full(len(t), np.nan)
    for i, pid in enumerate(t["player_id"]):
        b = bd.get(pid)
        if b is not None:
            out[i] = (cutoff_ord[i] - b.toordinal()) / 365.25
    return out
