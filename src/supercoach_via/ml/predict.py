"""Forecasts for an explicit fixture/roster universe (PLAN 7.1).

``forecast(history, bundle, request)`` targets ``(player_id, match_id)`` pairs of actual
fixtures at an explicit ``forecast_cutoff``:

* prospective: only ``status=scheduled`` matches whose date is on/after the cutoff's UTC
  day. Without an explicit scope the earliest such stage is used. If none exists the
  artifact is ``unavailable`` with reason ``no_valid_future_fixture`` — no round number is
  ever manufactured;
* replay: completed (or scheduled) matches in an explicit scope, rebuilt from pre-cutoff
  history only — identical features to a prospective run with the same cutoff (M05).

Candidate universe per (match, club):
* a verified announced lineup (``role=named`` with ``announced_at < cutoff``) ->
  ``selection_status=confirmed``;
* otherwise players whose most recent eligible pre-cutoff appearance was for that club
  within ``roster_lookback_seasons`` -> ``selection_status=unconfirmed``;
* or an explicit ``candidates`` list. Every intended candidate is forecast or omitted
  with a reason (``unresolved_identity``, ``insufficient_history``, ``not_in_fixture``).
No injury inference, no age/DOB filtering.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from supercoach_via.domain.schemas import ForecastStatus, Origin, SelectionStatus, is_safe_id
from supercoach_via.ml.bundles import ModelBundle, canonical_json, json_safe
from supercoach_via.ml.features import (
    FeatureFrame,
    FeatureSpec,
    History,
    _prepare_observations,
    _utc_start,
    build_features,
)
from supercoach_via.ml.models import COLD_START_NAME
from supercoach_via.storage.snapshots import atomic_write_bytes, sha256_file

CANONICAL_FIELDS: tuple[str, ...] = (
    "prediction_id", "prediction_run_id", "snapshot_id", "model_id", "player_id", "club_id",
    "match_id", "season", "stage_id", "stage_label", "scheduled_at", "forecast_cutoff",
    "origin", "generated_at", "selection_status", "eligibility_basis", "history_games",
    "predicted_disposals", "interval_low", "interval_high", "interval_level",
    "interval_method", "warnings",
)
EXTRA_FIELDS: tuple[str, ...] = (
    "opponent_club_id", "venue_id", "match_date", "recent_mean_5", "baseline_model_id",
    "baseline_prediction",
)
NO_FIXTURE = "no_valid_future_fixture"


def _rows(df: pd.DataFrame) -> Iterator[Any]:
    """Row namedtuples typed as Any (pandas-stubs types attributes as a scalar union)."""
    return iter(df.itertuples())


class ArtifactExistsError(FileExistsError):
    pass


class ArtifactIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ForecastRequest:
    forecast_cutoff: datetime
    generated_at: datetime
    origin: Origin = Origin.PROSPECTIVE
    season: int | None = None
    stage_id: str | None = None
    match_ids: tuple[str, ...] | None = None
    candidates: tuple[tuple[str, str], ...] | None = None  # explicit (player_id, club_id)
    roster_lookback_seasons: int = 1
    allow_cold_start: bool = True
    prediction_run_id: str | None = None

    def __post_init__(self) -> None:
        for f in ("forecast_cutoff", "generated_at"):
            v = getattr(self, f)
            if v.tzinfo is None:
                raise ValueError(f"{f} must be timezone-aware (UTC)")
        if self.origin is Origin.LEGACY_UNKNOWN:
            raise ValueError("legacy_unknown artifacts are imported, never produced by forecast()")


class PredictionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    kind: str = "disposal_forecast"
    prediction_run_id: str
    origin: str
    status: str
    reason: str | None
    snapshot_id: str
    model_id: str
    model_name: str
    model_promoted: bool
    baseline_model_id: str
    feature_version: str
    forecast_cutoff: datetime
    generated_at: datetime
    season: int | None
    stage_id: str | None
    stage_label: str | None
    target_matches: dict[str, dict[str, Any]]
    intended: int
    predicted: int
    omissions: dict[str, int]
    interval: dict[str, Any]
    rows_file: str = "rows.parquet"
    rows_sha256: str = ""
    omissions_file: str = "omissions.json"
    omissions_sha256: str = ""
    warnings: list[str]


@dataclass(frozen=True)
class PredictionArtifact:
    manifest: PredictionManifest
    rows: pd.DataFrame
    omissions: pd.DataFrame  # player_id, club_id, match_id, reason, detail
    features: FeatureFrame | None = None

    def with_rows(self, rows: pd.DataFrame) -> PredictionArtifact:
        return replace(self, rows=rows)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def _select_matches(history: History, req: ForecastRequest) -> pd.DataFrame:
    m = history.matches
    cutoff_day = req.forecast_cutoff.astimezone(UTC).date()
    sel = m[m["match_date"].notna()]
    sel = sel[sel["match_date"].map(lambda d: d >= cutoff_day)]
    if req.season is not None:
        sel = sel[sel["season"] == req.season]
    if req.stage_id is not None:
        sel = sel[sel["stage_id"] == req.stage_id]
    if req.match_ids is not None:
        sel = sel[sel["match_id"].isin(req.match_ids)]
    if req.origin is Origin.PROSPECTIVE:
        sel = sel[sel["status"] == "scheduled"]
    else:
        sel = sel[sel["status"].isin(["complete", "scheduled"])]
    # the cutoff must precede a minute-precision start with a verified venue timezone
    tz = history.venues.set_index("venue_id")["timezone"] if len(history.venues) else pd.Series(dtype=object)
    cut = req.forecast_cutoff.timestamp()
    ok = [
        not (p == "minute" and not np.isnan(st := _utc_start(ls, tz.get(v) if v is not None else None))
             and st < cut)
        for ls, p, v in zip(sel["local_start"], sel["date_precision"], sel["venue_id"], strict=True)
    ]
    sel = sel[np.asarray(ok, dtype=bool)] if len(sel) else sel
    if len(sel) and req.season is None and req.stage_id is None and req.match_ids is None:
        first = sel.sort_values(["season", "stage_order", "match_date"]).iloc[0]
        sel = sel[(sel["season"] == first["season"]) & (sel["stage_id"] == first["stage_id"])]
    return sel.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def _roster(history: History, spec: FeatureSpec, req: ForecastRequest, seasons: set[int]) -> pd.DataFrame:
    """Most recent eligible pre-cutoff appearance per player -> (player_id, club_id, season)."""
    obs, _ = _prepare_observations(history, spec)
    cutoff_ord = req.forecast_cutoff.astimezone(UTC).date().toordinal()
    obs = obs[obs["eff_ord"] < cutoff_ord]
    if obs.empty:
        return pd.DataFrame(columns=["player_id", "club_id", "season"])
    last = obs.groupby("player_id", sort=True).tail(1)
    lo = min(seasons) - req.roster_lookback_seasons
    return last[last["season"] >= lo][["player_id", "club_id", "season"]].reset_index(drop=True)


def _lineups(history: History, req: ForecastRequest, match_ids: set[str]) -> pd.DataFrame:
    lu = history.lineups
    if lu is None or lu.empty or "announced_at" not in lu:
        return pd.DataFrame(columns=["match_id", "club_id", "player_id"])
    lu = lu[lu["match_id"].isin(match_ids) & (lu["role"] == "named")]
    ann = pd.to_datetime(lu["announced_at"], utc=True)
    return lu[ann.notna() & (ann < pd.Timestamp(req.forecast_cutoff))][["match_id", "club_id", "player_id"]]


def _run_id(history: History, bundle: ModelBundle, req: ForecastRequest, match_ids: list[str]) -> str:
    if req.prediction_run_id:
        if not is_safe_id(req.prediction_run_id):
            raise ValueError("unsafe prediction_run_id")
        return req.prediction_run_id
    payload = {"snapshot": history.snapshot_id, "model": bundle.bundle_id, "origin": req.origin.value,
               "cutoff": req.forecast_cutoff.isoformat(), "generated_at": req.generated_at.isoformat(),
               "matches": match_ids, "candidates": req.candidates}
    h = hashlib.sha256(canonical_json(payload)).hexdigest()[:16]
    return f"{req.origin.value}-{req.forecast_cutoff.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{h}"


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in (*CANONICAL_FIELDS, *EXTRA_FIELDS)})


def forecast(history: History, bundle: ModelBundle, request: ForecastRequest) -> PredictionArtifact:
    man = bundle.manifest
    spec = FeatureSpec()
    if spec.version != man.feature_version:
        raise ValueError(f"bundle feature version {man.feature_version} != {spec.version}")
    matches = _select_matches(history, request)
    match_ids = list(matches["match_id"])
    run_id = _run_id(history, bundle, request, match_ids)
    interval = dict(man.interval)
    common: dict[str, Any] = dict(
        prediction_run_id=run_id, origin=request.origin.value, snapshot_id=history.snapshot_id,
        model_id=bundle.bundle_id, model_name=man.name, model_promoted=man.promoted,
        baseline_model_id=f"{bundle.bundle_id}:baseline_prior5", feature_version=man.feature_version,
        forecast_cutoff=request.forecast_cutoff, generated_at=request.generated_at, interval=interval,
    )
    omissions: list[dict[str, Any]] = []
    if matches.empty:
        intended = len(request.candidates or ())
        for pid, club in request.candidates or ():
            omissions.append({"player_id": pid, "club_id": club, "match_id": None,
                              "reason": "not_in_fixture", "detail": NO_FIXTURE})
        om = pd.DataFrame(omissions, columns=["player_id", "club_id", "match_id", "reason", "detail"])
        return PredictionArtifact(
            PredictionManifest(**common, status=ForecastStatus.UNAVAILABLE.value, reason=NO_FIXTURE,
                               season=request.season, stage_id=request.stage_id, stage_label=None,
                               target_matches={}, intended=intended, predicted=0,
                               omissions=om["reason"].value_counts().to_dict(),
                               warnings=["no scheduled fixture on/after the cutoff in the requested scope"]),
            _empty_rows(), om)

    # candidate universe ----------------------------------------------------------
    sides = pd.concat([
        matches[["match_id", "home_club_id", "away_club_id"]].rename(
            columns={"home_club_id": "club_id", "away_club_id": "opponent_club_id"}),
        matches[["match_id", "away_club_id", "home_club_id"]].rename(
            columns={"away_club_id": "club_id", "home_club_id": "opponent_club_id"}),
    ])
    confirmed = _lineups(history, request, set(match_ids))
    conf_sides = set(zip(confirmed["match_id"], confirmed["club_id"], strict=True))
    if request.candidates is not None:
        cand = pd.DataFrame(list(request.candidates), columns=["player_id", "club_id"])
    else:
        cand = _roster(history, spec, request, set(matches["season"]))[["player_id", "club_id"]]
    rows = cand.merge(sides, on="club_id", how="left")
    for r in _rows(rows[rows["match_id"].isna()]):
        omissions.append({"player_id": r.player_id, "club_id": r.club_id, "match_id": None,
                          "reason": "not_in_fixture", "detail": "club has no target match in scope"})
    rows = rows[rows["match_id"].notna()]
    # confirmed lineups replace the inferred roster for that side
    keep_inferred = [(m, c) not in conf_sides for m, c in zip(rows["match_id"], rows["club_id"], strict=True)]
    for r in _rows(rows[~np.asarray(keep_inferred, dtype=bool)]):
        if (r.match_id, r.club_id, r.player_id) not in set(confirmed.itertuples(index=False, name=None)):
            omissions.append({"player_id": r.player_id, "club_id": r.club_id, "match_id": r.match_id,
                              "reason": "other", "detail": "not named in the announced lineup"})
    rows = rows[np.asarray(keep_inferred, dtype=bool)].assign(selection_status=SelectionStatus.UNCONFIRMED.value)
    conf = confirmed.merge(sides, on=["match_id", "club_id"], how="inner").assign(
        selection_status=SelectionStatus.CONFIRMED.value)
    rows = pd.concat([rows, conf], ignore_index=True).drop_duplicates(["match_id", "player_id"])
    intended = len(rows) + len(omissions)
    # identity resolution
    players = history.players.set_index("player_id") if len(history.players) else pd.DataFrame()
    status = rows["player_id"].map(players["identity_status"]) if len(players) else pd.Series(None, index=rows.index)
    bad = status.isna() | (status != "canonical")
    for r in _rows(rows[bad]):
        omissions.append({"player_id": r.player_id, "club_id": r.club_id, "match_id": r.match_id,
                          "reason": "unresolved_identity",
                          "detail": "identity not in registry" if pd.isna(status.get(r.Index))
                          else f"identity_status={status.get(r.Index)}"})
    rows = rows[~bad].reset_index(drop=True)
    # targets + features ------------------------------------------------------------
    mi = matches.set_index("match_id")
    t = rows.assign(
        season=rows["match_id"].map(mi["season"]).astype(int),
        stage_id=rows["match_id"].map(mi["stage_id"]),
        stage_label=rows["match_id"].map(mi["stage_label"]),
        stage_type=rows["match_id"].map(mi["stage_type"]),
        stage_order=rows["match_id"].map(mi["stage_order"]),
        match_date=rows["match_id"].map(mi["match_date"]),
        venue_id=rows["match_id"].map(mi["venue_id"]),
        forecast_cutoff=pd.Timestamp(request.forecast_cutoff).tz_convert("UTC"),
    )
    ff = build_features(history, t, spec)
    if tuple(man.feature_names) != ff.feature_names:
        raise ValueError("feature schema differs from the bundle's manifest")
    champ = bundle.predictor["champion"]
    base = bundle.predictor["baseline"]
    pred, basis = champ.predict(ff)
    bpred, _ = base.predict(ff)
    hist = ff.history_games
    if not request.allow_cold_start:
        drop = hist < champ.min_history
        for r in _rows(t[drop]):
            omissions.append({"player_id": r.player_id, "club_id": r.club_id, "match_id": r.match_id,
                              "reason": "insufficient_history", "detail": f"history_games < {champ.min_history}"})
        keep = ~drop
        t, pred, basis, bpred, hist = t[keep], pred[keep], basis[keep], bpred[keep], hist[keep]
        ff_keep = np.flatnonzero(keep)
    else:
        ff_keep = np.arange(len(t))
    t = t.reset_index(drop=True)
    q = interval.get("q") if interval.get("available") else None
    calibrated = bool(interval.get("holdout", {}).get("calibrated"))
    lo = np.maximum(pred - q, 0.0) if q is not None else np.full(len(t), np.nan)
    hi = pred + q if q is not None else np.full(len(t), np.nan)
    tz = history.venues.set_index("venue_id")["timezone"] if len(history.venues) else pd.Series(dtype=object)
    sched = {
        r.match_id: (lambda s: None if np.isnan(s) else datetime.fromtimestamp(s, UTC))(
            _utc_start(r.local_start, tz.get(r.venue_id) if r.venue_id is not None else None)
            if r.date_precision == "minute" else np.nan)
        for r in _rows(matches)
    }
    warnings = []
    for b, s in zip(basis, t["selection_status"], strict=True):
        w = []
        if b == COLD_START_NAME:
            w.append(COLD_START_NAME)
        if s == SelectionStatus.UNCONFIRMED.value:
            w.append("selection_unconfirmed")
        if q is not None and not calibrated:
            w.append("interval_uncalibrated")
        warnings.append(w)
    out = pd.DataFrame({
        "prediction_id": [hashlib.sha256(f"{run_id}|{p}|{m}".encode()).hexdigest()[:24]
                          for p, m in zip(t["player_id"], t["match_id"], strict=True)],
        "prediction_run_id": run_id,
        "snapshot_id": history.snapshot_id,
        "model_id": bundle.bundle_id,
        "player_id": t["player_id"],
        "club_id": t["club_id"],
        "match_id": t["match_id"],
        "season": t["season"].astype("int64"),
        "stage_id": t["stage_id"],
        "stage_label": t["stage_label"],
        "scheduled_at": [sched[m] for m in t["match_id"]],
        "forecast_cutoff": pd.Timestamp(request.forecast_cutoff).tz_convert("UTC"),
        "origin": request.origin.value,
        "generated_at": pd.Timestamp(request.generated_at).tz_convert("UTC"),
        "selection_status": t["selection_status"],
        "eligibility_basis": basis,
        "history_games": hist.astype("int64"),
        "predicted_disposals": pred.astype(np.float64),
        "interval_low": lo,
        "interval_high": hi,
        "interval_level": interval.get("level") if q is not None else None,
        "interval_method": interval.get("method") if q is not None else None,
        "warnings": warnings,
        "opponent_club_id": t["opponent_club_id"],
        "venue_id": t["venue_id"],
        "match_date": t["match_date"],
        "recent_mean_5": ff.X["disposals_prior5_mean"].to_numpy()[ff_keep],
        "baseline_model_id": f"{bundle.bundle_id}:baseline_prior5",
        "baseline_prediction": bpred.astype(np.float64),
    })
    out = out.sort_values(["match_id", "club_id", "player_id"], kind="stable").reset_index(drop=True)
    om = pd.DataFrame(omissions, columns=["player_id", "club_id", "match_id", "reason", "detail"])
    first = matches.iloc[0]
    one_stage = matches["stage_id"].nunique() == 1 and matches["season"].nunique() == 1
    target_matches = {
        r.match_id: json_safe({"season": int(r.season), "stage_id": r.stage_id, "stage_label": r.stage_label,
                               "match_date": r.match_date.isoformat(), "home_club_id": r.home_club_id,
                               "away_club_id": r.away_club_id, "status_at_forecast": r.status})
        for r in _rows(matches)
    }
    manifest = PredictionManifest(
        **common, status=ForecastStatus.AVAILABLE.value if len(out) else ForecastStatus.UNAVAILABLE.value,
        reason=None if len(out) else "no_eligible_candidates",
        season=int(first["season"]) if matches["season"].nunique() == 1 else None,
        stage_id=str(first["stage_id"]) if one_stage else None,
        stage_label=str(first["stage_label"]) if one_stage else None,
        target_matches=target_matches, intended=intended, predicted=len(out),
        omissions=om["reason"].value_counts().to_dict(),
        warnings=[] if calibrated or q is None else ["interval not calibrated on holdout"],
    )
    return PredictionArtifact(manifest, out, om, ff)


# ---------------------------------------------------------------------------
# Immutable persistence
# ---------------------------------------------------------------------------


def _rows_bytes_path(rows: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(rows, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def write_artifact(artifact: PredictionArtifact, root: Path) -> Path:
    """Write under ``root/<prediction_run_id>/``; an existing run directory is never
    overwritten (raises ``ArtifactExistsError``)."""
    run_id = artifact.manifest.prediction_run_id
    if not is_safe_id(run_id):
        raise ValueError("unsafe prediction_run_id")
    root.mkdir(parents=True, exist_ok=True)
    final = root / run_id
    if final.exists():
        raise ArtifactExistsError(f"prediction run {run_id} already archived")
    tmp = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    try:
        _rows_bytes_path(artifact.rows, tmp / "rows.parquet")
        om_bytes = canonical_json(json_safe(artifact.omissions.to_dict(orient="records")))
        (tmp / "omissions.json").write_bytes(om_bytes)
        m = artifact.manifest.model_copy(update={
            "rows_sha256": sha256_file(tmp / "rows.parquet"),
            "omissions_sha256": hashlib.sha256(om_bytes).hexdigest(),
        })
        atomic_write_bytes(tmp / "manifest.json", m.model_dump_json(indent=1).encode())
        tmp.rename(final)
        for p in final.iterdir():
            p.chmod(0o444)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return final


def load_artifact(directory: Path) -> PredictionArtifact:
    import pyarrow.parquet as pq

    m = PredictionManifest.model_validate_json((directory / "manifest.json").read_bytes())
    rows_path = directory / m.rows_file
    if sha256_file(rows_path) != m.rows_sha256:
        raise ArtifactIntegrityError("rows file does not match manifest hash")
    om_bytes = (directory / m.omissions_file).read_bytes()
    if hashlib.sha256(om_bytes).hexdigest() != m.omissions_sha256:
        raise ArtifactIntegrityError("omissions file does not match manifest hash")
    rows = pq.read_table(rows_path).to_pandas()
    om = pd.DataFrame(json.loads(om_bytes), columns=["player_id", "club_id", "match_id", "reason", "detail"])
    return PredictionArtifact(m, rows, om)


# ---------------------------------------------------------------------------
# Public view-model conversion
# ---------------------------------------------------------------------------


def _names(df: pd.DataFrame, key: str, col: str) -> dict[str, str]:
    if df is None or df.empty or col not in df:
        return {}
    return {str(k): str(v) for k, v in zip(df[key], df[col], strict=True)}


def match_summary(history: History, match_id: str) -> Any:
    from supercoach_via.publish.view_models import MatchSummary, TeamScore

    clubs = _names(history.clubs, "club_id", "name")
    venues = _names(history.venues, "venue_id", "name")
    m: Any = history.matches.set_index("match_id").loc[match_id]

    def num(v: Any) -> int | None:
        return None if v is None or pd.isna(v) else int(v)

    hs, as_ = num(m.get("home_score")), num(m.get("away_score"))
    winner = None
    if m["status"] == "complete" and hs is not None and as_ is not None and hs != as_:
        winner = m["home_club_id"] if hs > as_ else m["away_club_id"]
    return MatchSummary(
        match_id=match_id, season=int(m["season"]), stage_id=str(m["stage_id"]),
        stage_label=str(m["stage_label"]), stage_type=m["stage_type"],
        round_number=num(m["round_number"]), stage_order=int(m["stage_order"]),
        replay_occurrence=int(m["replay_occurrence"]),
        local_start=m["local_start"] if isinstance(m["local_start"], str) else None,
        match_date=m["match_date"], date_precision=m["date_precision"], status=m["status"],
        venue=venues.get(str(m["venue_id"])) if m["venue_id"] is not None else None,
        home=TeamScore(club_id=m["home_club_id"], name=clubs.get(m["home_club_id"], m["home_club_id"]),
                       goals=num(m.get("home_final_goals")), behinds=num(m.get("home_final_behinds")), score=hs),
        away=TeamScore(club_id=m["away_club_id"], name=clubs.get(m["away_club_id"], m["away_club_id"]),
                       goals=num(m.get("away_final_goals")), behinds=num(m.get("away_final_behinds")), score=as_),
        winner_club_id=winner,
    )


def model_info(bundle: ModelBundle) -> Any:
    from supercoach_via.publish.view_models import ModelInfo

    m = bundle.manifest
    return ModelInfo(
        model_id=m.bundle_id, kind="baseline" if m.kind == "baseline" else "model", name=m.name,
        description=m.description,
        trained_cutoff=datetime.fromisoformat(m.train_cutoff).replace(tzinfo=UTC),
        promoted=m.promoted, promotion_note=m.promotion_note,
    )


def interval_info(bundle: ModelBundle) -> Any:
    from supercoach_via.publish.view_models import IntervalInfo

    i = bundle.manifest.interval
    hold = i.get("holdout") or {}
    return IntervalInfo(available=bool(i.get("available")), level=i.get("level"), method=i.get("method"),
                        calibrated=bool(hold.get("calibrated")),
                        reason=i.get("reason") or hold.get("reason"))


def to_prediction_set(artifact: PredictionArtifact, history: History, bundle: ModelBundle) -> Any:
    """Map an artifact to the public ``PredictionSet`` (full float precision)."""
    from supercoach_via.publish.view_models import OmissionSummary, PredictionRow, PredictionSet

    m = artifact.manifest
    players = _names(history.players, "player_id", "display_name")
    clubs = _names(history.clubs, "club_id", "name")
    venues = _names(history.venues, "venue_id", "name")
    local = _names(history.matches, "match_id", "local_start")

    def f(v: Any) -> float | None:
        return None if v is None or pd.isna(v) else float(v)

    rows = [
        PredictionRow(
            prediction_id=r.prediction_id, prediction_run_id=r.prediction_run_id, snapshot_id=r.snapshot_id,
            model_id=r.model_id, player_id=r.player_id, player_name=players.get(r.player_id, r.player_id),
            club_id=r.club_id, club_name=clubs.get(r.club_id, r.club_id),
            opponent_club_id=r.opponent_club_id,
            opponent_name=clubs.get(r.opponent_club_id) if r.opponent_club_id else None,
            match_id=r.match_id, season=int(r.season), stage_id=r.stage_id, stage_label=r.stage_label,
            scheduled_at=None if r.scheduled_at is None or pd.isna(r.scheduled_at) else r.scheduled_at,
            scheduled_local=local.get(r.match_id) if local.get(r.match_id) not in (None, "None") else None,
            venue=venues.get(r.venue_id) if r.venue_id else None,
            forecast_cutoff=r.forecast_cutoff, origin=r.origin, generated_at=r.generated_at,
            selection_status=r.selection_status, eligibility_basis=r.eligibility_basis,
            history_games=int(r.history_games), recent_mean_5=f(r.recent_mean_5),
            predicted_disposals=float(r.predicted_disposals), interval_low=f(r.interval_low),
            interval_high=f(r.interval_high), interval_level=f(r.interval_level),
            interval_method=r.interval_method if isinstance(r.interval_method, str) else None,
            warnings=list(r.warnings),
        )
        for r in _rows(artifact.rows)
    ]
    om = [OmissionSummary(reason=k, count=int(v), detail=None) for k, v in sorted(m.omissions.items())]
    return PredictionSet(
        season=m.season if m.season is not None else (int(artifact.rows["season"].iloc[0]) if len(rows) else 0),
        stage_id=m.stage_id or "none", stage_label=m.stage_label or "none", status=m.status, reason=m.reason,
        generated_at=m.generated_at, forecast_cutoff=m.forecast_cutoff,
        target_matches=[match_summary(history, mid) for mid in m.target_matches],
        model=model_info(bundle), interval=interval_info(bundle), rows=rows, omissions=om,
    )
