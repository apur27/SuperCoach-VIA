"""Metrics, intervals, champion gate, and scoring/replay of prediction artifacts (PLAN 7.4).

Definitions: error = prediction - actual; MAE = mean |error|; RMSE = sqrt(mean error^2);
bias = mean error; median AE; within-k = fraction with |error| <= k. Headline values are
pooled player-game weighted; mean-of-rounds values are separately labelled. Unknown
actuals are excluded with count/reason and never zero-filled; zero actuals are valid.
Full float precision is kept throughout; rounding is a display concern.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

INTERVAL_METHOD = "split_conformal_abs_residual"
MIN_CALIBRATION = 200
CALIBRATED_BAND = (0.75, 0.85)


def _rows(df: pd.DataFrame) -> Iterator[Any]:
    """Row namedtuples typed as Any (pandas-stubs types attributes as a scalar union)."""
    return iter(df.itertuples())


@dataclass(frozen=True)
class Metrics:
    n: int
    mae: float | None
    rmse: float | None
    bias: float | None
    median_ae: float | None
    within_5: float | None
    within_10: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def metric_block(pred: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> Metrics:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise ValueError("prediction/actual length mismatch")
    if np.isnan(a).any() or np.isnan(p).any():
        raise ValueError("unknown values must be excluded (with a reason) before scoring")
    n = len(p)
    if n == 0:
        return Metrics(0, None, None, None, None, None, None)
    e = p - a
    ae = np.abs(e)
    return Metrics(
        n=n,
        mae=float(ae.mean()),
        rmse=float(math.sqrt(float((e * e).mean()))),
        bias=float(e.mean()),
        median_ae=float(np.median(ae)),
        within_5=float((ae <= 5).mean()),
        within_10=float((ae <= 10).mean()),
    )


def mean_of_groups_mae(pred: np.ndarray, actual: np.ndarray, groups: np.ndarray) -> float | None:
    """Unweighted mean of per-group (e.g. per-round) MAE. Labelled separately from pooled."""
    df = pd.DataFrame({"ae": np.abs(np.asarray(pred, float) - np.asarray(actual, float)), "g": groups})
    if df.empty:
        return None
    return float(df.groupby("g")["ae"].mean().mean())


@dataclass(frozen=True)
class CohortRow:
    dimension: str
    cohort: str
    model: Metrics
    baseline: Metrics | None
    sufficient: bool
    post_hoc: bool = False


def cohort_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
    dims: dict[str, str],
    baseline_col: str | None = None,
    min_n: int = 100,
    post_hoc_dims: frozenset[str] = frozenset(),
) -> list[CohortRow]:
    rows: list[CohortRow] = []
    for dim, col in dims.items():
        for key, g in df.groupby(col, dropna=False, sort=True):
            base = metric_block(g[baseline_col], g[actual_col]) if baseline_col else None
            rows.append(CohortRow(
                dimension=dim,
                cohort=str(key),
                model=metric_block(g[pred_col], g[actual_col]),
                baseline=base,
                sufficient=len(g) >= min_n,
                post_hoc=dim in post_hoc_dims,
            ))
    return rows


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntervalCalibration:
    available: bool
    level: float | None
    method: str | None
    q: float | None
    n_calibration: int
    reason: str | None


def conformal_quantile(abs_residuals: np.ndarray, level: float = 0.8,
                       min_n: int = MIN_CALIBRATION) -> IntervalCalibration:
    """Split-conformal absolute-residual quantile at finite-sample rank ceil((n+1)*level).

    Requires ``min_n`` calibration outcomes; if the rank exceeds n the interval is
    unbounded and reported unavailable. Never derived from MAE.
    """
    r = np.sort(np.asarray(abs_residuals, dtype=float))
    if np.isnan(r).any():
        raise ValueError("calibration residuals contain unknown values")
    n = len(r)
    if n < min_n:
        return IntervalCalibration(False, None, None, None, n, f"insufficient_calibration: {n} < {min_n}")
    rank = math.ceil((n + 1) * level)
    if rank > n:
        return IntervalCalibration(False, None, None, None, n, "quantile_rank_exceeds_sample")
    return IntervalCalibration(True, level, INTERVAL_METHOD, float(r[rank - 1]), n, None)


def apply_interval(pred: np.ndarray, cal: IntervalCalibration) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float)
    if not cal.available or cal.q is None:
        nan = np.full(len(p), np.nan)
        return nan, nan
    return np.maximum(p - cal.q, 0.0), p + cal.q


@dataclass(frozen=True)
class IntervalEvaluation:
    n: int
    coverage: float | None
    median_width: float | None
    calibrated: bool
    reason: str | None


def evaluate_interval(low: np.ndarray, high: np.ndarray, actual: np.ndarray,
                      min_n: int = MIN_CALIBRATION, band: tuple[float, float] = CALIBRATED_BAND) -> IntervalEvaluation:
    lo, hi, a = (np.asarray(x, dtype=float) for x in (low, high, actual))
    ok = ~(np.isnan(lo) | np.isnan(hi) | np.isnan(a))
    n = int(ok.sum())
    if n == 0:
        return IntervalEvaluation(0, None, None, False, "no_interval_or_outcomes")
    cov = float(((a[ok] >= lo[ok]) & (a[ok] <= hi[ok])).mean())
    width = float(np.median(hi[ok] - lo[ok]))
    if n < min_n:
        return IntervalEvaluation(n, cov, width, False, f"insufficient_holdout: {n} < {min_n}")
    if not band[0] <= cov <= band[1]:
        return IntervalEvaluation(n, cov, width, False, f"coverage_outside_band {band}")
    return IntervalEvaluation(n, cov, width, True, None)


# ---------------------------------------------------------------------------
# Champion gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDecision:
    candidate: str
    baseline: str
    passed: bool
    candidate_mae: float | None
    baseline_mae: float | None
    relative_improvement: float | None
    n: int
    failing_cohorts: list[str] = field(default_factory=list)
    reason: str = ""


def champion_gate(
    df: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    cand_col: str,
    base_col: str,
    actual_col: str,
    cohort_dims: dict[str, str],
    min_rel_improvement: float = 0.01,
    cohort_min_n: int = 100,
    cohort_max_rel_worse: float = 0.05,
) -> GateDecision:
    """Matched rows only: the frame must hold both predictions for every row."""
    if df[[cand_col, base_col, actual_col]].isna().any().any():
        raise ValueError("gate requires matched, fully scored rows")
    c = metric_block(df[cand_col], df[actual_col])
    b = metric_block(df[base_col], df[actual_col])
    if c.n == 0 or b.mae is None or c.mae is None:
        return GateDecision(candidate, baseline, False, None, None, None, 0, [], "no scored holdout rows")
    rel = (b.mae - c.mae) / b.mae if b.mae > 0 else 0.0
    failing: list[str] = []
    for row in cohort_metrics(df, pred_col=cand_col, actual_col=actual_col, dims=cohort_dims,
                              baseline_col=base_col, min_n=cohort_min_n):
        if not row.sufficient or row.baseline is None or row.baseline.mae is None or row.model.mae is None:
            continue
        if row.baseline.mae > 0 and (row.model.mae - row.baseline.mae) / row.baseline.mae > cohort_max_rel_worse:
            failing.append(f"{row.dimension}={row.cohort}")
    passed = rel >= min_rel_improvement and not failing
    if passed:
        reason = (f"MAE improves {rel:.2%} >= {min_rel_improvement:.0%}; "
                  f"no sufficient cohort worse by >{cohort_max_rel_worse:.0%}")
    elif rel < min_rel_improvement:
        reason = f"MAE improvement {rel:.2%} < required {min_rel_improvement:.0%}"
    else:
        reason = f"cohorts worse by >{cohort_max_rel_worse:.0%}: {', '.join(failing)}"
    return GateDecision(candidate, baseline, passed, c.mae, b.mae, rel, c.n, failing, reason)


# ---------------------------------------------------------------------------
# Artifact scoring, replay, legacy archive
# ---------------------------------------------------------------------------

METRIC_VERSION = "metrics_v1"


class TargetMismatchError(ValueError):
    """Prediction rows/manifest disagree with the actuals snapshot's fixture identity."""


class OriginError(ValueError):
    """Evaluations of different origins may not be scored or pooled together."""


@dataclass(frozen=True)
class EvaluationArtifact:
    evaluation_id: str
    origin: str
    label: str
    prediction_run_ids: tuple[str, ...]
    model_id: str
    baseline_model_id: str | None
    actuals_snapshot_id: str
    metric_version: str
    populations: dict[str, Any]
    headline: Metrics
    baseline_headline: Metrics | None
    mean_of_rounds_mae: float | None
    cohorts: list[CohortRow]
    interval: IntervalEvaluation
    scored_rows: pd.DataFrame
    notes: list[str]


def _check_targets(rows: pd.DataFrame, manifest_targets: dict[str, dict[str, Any]], matches: pd.DataFrame) -> None:
    mi = matches.set_index("match_id")
    for mid, claim in manifest_targets.items():
        if mid not in mi.index:
            raise TargetMismatchError(f"target match {mid} absent from actuals snapshot")
        m = mi.loc[mid]
        for k in ("season", "stage_id"):
            if str(m[k]) != str(claim[k]):
                raise TargetMismatchError(f"target {mid}: manifest {k}={claim[k]} != actuals {m[k]}")
        if "match_date" in claim and str(m["match_date"]) != str(claim["match_date"]):
            raise TargetMismatchError(f"target {mid}: date moved {claim['match_date']} -> {m['match_date']}")
    for r in _rows(rows[["match_id", "season", "stage_id"]].drop_duplicates()):
        row_claim = manifest_targets.get(r.match_id)
        if row_claim is None or int(row_claim["season"]) != int(r.season) or row_claim["stage_id"] != r.stage_id:
            raise TargetMismatchError(f"row target {r.match_id} disagrees with its manifest")


def _eval_id(parts: dict[str, Any]) -> str:
    import hashlib
    import json

    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]
    return f"eval-{h}"


def evaluate(predictions: Any, actuals: Any, *, min_cohort: int = 100) -> EvaluationArtifact:
    """Score one PredictionArtifact against an identified actuals snapshot (History).

    Never mutates or rewrites the prediction artifact.
    """
    from supercoach_via.ml.models import history_band, volume_band

    man = predictions.manifest
    rows = predictions.rows.copy()
    _check_targets(rows, man.target_matches, actuals.matches)
    pg = actuals.player_games
    mids = set(man.target_matches)
    played = pg[pg["match_id"].isin(mids)]
    status = actuals.matches.set_index("match_id")["status"]
    act = played[["match_id", "player_id", "club_id", "disposals"]].rename(
        columns={"club_id": "actual_club_id", "disposals": "actual"})
    j = rows.merge(act, on=["match_id", "player_id"], how="left", validate="one_to_one", indicator=True)
    joined = j["_merge"] == "both"
    reasons: dict[str, int] = {}
    excl = pd.Series(False, index=j.index)

    def mark(mask: pd.Series, reason: str) -> None:
        nonlocal excl
        new = mask & ~excl
        if new.any():
            reasons[reason] = reasons.get(reason, 0) + int(new.sum())
        excl = excl | new

    mark(joined & (j["match_id"].map(status) != "complete"), "match_not_complete")
    mark(joined & (j["actual_club_id"] != j["club_id"]), "club_mismatch")
    mark(joined & j["actual"].isna(), "unknown_actual")
    missing = int((~joined).sum())
    if missing:
        reasons["no_actual_row_did_not_play_or_unrecorded"] = missing
    scored = j[joined & ~excl].drop(columns="_merge").copy()
    scored["actual"] = scored["actual"].astype(float)
    pops = {
        "intended": int(man.intended), "predicted": len(rows), "joined": int(joined.sum()),
        "played": len(played), "missing": missing, "excluded": int((joined & excl).sum()),
        "exclusion_reasons": reasons,
        "played_not_predicted": int(len(played) - joined.sum()),
    }
    p = scored["predicted_disposals"].to_numpy(dtype=float)
    a = scored["actual"].to_numpy(dtype=float)
    head = metric_block(p, a)
    base = None
    if "baseline_prediction" in scored and len(scored):
        base = metric_block(scored["baseline_prediction"].to_numpy(dtype=float), a)
    scored["season_stage"] = scored["season"].astype(str) + ":" + scored["stage_id"].astype(str)
    scored["history_band"] = history_band(scored["history_games"].to_numpy())
    scored["volume_band"] = volume_band(p)
    scored["actual_band"] = volume_band(a)
    dims = {"season_stage": "season_stage", "club": "club_id", "history_band": "history_band",
            "predicted_volume_band": "volume_band", "actual_band": "actual_band"}
    cohorts = cohort_metrics(scored, pred_col="predicted_disposals", actual_col="actual", dims=dims,
                             baseline_col="baseline_prediction" if base else None, min_n=min_cohort,
                             post_hoc_dims=frozenset({"actual_band"})) if len(scored) else []
    ie = evaluate_interval(scored["interval_low"].to_numpy(dtype=float, na_value=np.nan),
                           scored["interval_high"].to_numpy(dtype=float, na_value=np.nan), a)
    label = {
        "prospective": "Prospective forecasts scored against later settled outcomes",
        "replay": "Replay (retrospective reconstruction from an identified snapshot; legacy rows' "
                  "available_at unknown — assumes each game was available the following UTC day)",
    }.get(man.origin, man.origin)
    eid = _eval_id({"runs": [man.prediction_run_id], "rows": man.rows_sha256 or len(rows),
                    "actuals": actuals.snapshot_id, "metric": METRIC_VERSION})
    notes = [f"headline pooled over {head.n} player-games; mean-of-rounds is a separate figure"]
    return EvaluationArtifact(
        evaluation_id=eid, origin=man.origin, label=label, prediction_run_ids=(man.prediction_run_id,),
        model_id=man.model_id, baseline_model_id=man.baseline_model_id,
        actuals_snapshot_id=actuals.snapshot_id, metric_version=METRIC_VERSION, populations=pops,
        headline=head, baseline_headline=base,
        mean_of_rounds_mae=mean_of_groups_mae(p, a, scored["season_stage"].to_numpy()) if len(scored) else None,
        cohorts=cohorts, interval=ie, scored_rows=scored.reset_index(drop=True), notes=notes,
    )


def score_archive(directory: Any, actuals: Any, *, allow_origins: tuple[Any, ...] | None = None,
                  min_cohort: int = 100) -> EvaluationArtifact:
    """``score``: evaluate an immutable archived artifact (hash-verified on load)."""
    from supercoach_via.domain.schemas import Origin
    from supercoach_via.ml.predict import load_artifact

    allowed = {o.value if hasattr(o, "value") else o for o in (allow_origins or (Origin.PROSPECTIVE,))}
    art = load_artifact(directory)
    if art.manifest.origin not in allowed:
        raise OriginError(f"artifact origin {art.manifest.origin} not in {sorted(allowed)}")
    return evaluate(art, actuals, min_cohort=min_cohort)


def pool_evaluations(evals: list[EvaluationArtifact], *, min_cohort: int = 100) -> EvaluationArtifact:
    """Pool evaluations of ONE origin and one model; mixed origins are refused."""
    if not evals:
        raise ValueError("nothing to pool")
    origins = {e.origin for e in evals}
    if len(origins) != 1:
        raise OriginError(f"refusing to pool origins {sorted(origins)}")
    if len({e.model_id for e in evals}) != 1:
        raise ValueError("refusing to pool different models")
    rows = pd.concat([e.scored_rows for e in evals], ignore_index=True)
    p = rows["predicted_disposals"].to_numpy(dtype=float)
    a = rows["actual"].to_numpy(dtype=float)
    base = metric_block(rows["baseline_prediction"].to_numpy(dtype=float), a) if "baseline_prediction" in rows else None
    pops: dict[str, Any] = {k: sum(int(e.populations[k]) for e in evals)
                            for k in ("intended", "predicted", "joined", "played", "missing", "excluded",
                                      "played_not_predicted")}
    reasons: dict[str, int] = {}
    for e in evals:
        for k, v in e.populations["exclusion_reasons"].items():
            reasons[k] = reasons.get(k, 0) + int(v)
    pops["exclusion_reasons"] = reasons
    dims = {"season_stage": "season_stage", "club": "club_id", "history_band": "history_band",
            "predicted_volume_band": "volume_band", "actual_band": "actual_band"}
    cohorts = cohort_metrics(rows, pred_col="predicted_disposals", actual_col="actual", dims=dims,
                             baseline_col="baseline_prediction" if base else None, min_n=min_cohort,
                             post_hoc_dims=frozenset({"actual_band"})) if len(rows) else []
    ie = evaluate_interval(rows["interval_low"].to_numpy(dtype=float, na_value=np.nan),
                           rows["interval_high"].to_numpy(dtype=float, na_value=np.nan), a)
    runs = tuple(r for e in evals for r in e.prediction_run_ids)
    return EvaluationArtifact(
        evaluation_id=_eval_id({"pooled": [e.evaluation_id for e in evals]}), origin=evals[0].origin,
        label=evals[0].label, prediction_run_ids=runs, model_id=evals[0].model_id,
        baseline_model_id=evals[0].baseline_model_id, actuals_snapshot_id=evals[0].actuals_snapshot_id,
        metric_version=METRIC_VERSION, populations=pops, headline=metric_block(p, a), baseline_headline=base,
        mean_of_rounds_mae=mean_of_groups_mae(p, a, rows["season_stage"].to_numpy()) if len(rows) else None,
        cohorts=cohorts, interval=ie, scored_rows=rows, notes=[f"pooled from {len(evals)} evaluations"],
    )


def replay(history: Any, bundle: Any, *, season: int, generated_at: Any,
           stage_ids: tuple[str, ...] | None = None, min_cohort: int = 100) -> tuple[list[Any], EvaluationArtifact]:
    """Rebuild per-stage replay forecasts (cutoff = 00:00Z on each stage's first match day)
    from one identified snapshot and evaluate them pooled. Clearly labelled ``replay``."""
    from supercoach_via.domain.schemas import Origin
    from supercoach_via.ml.features import match_day_cutoff
    from supercoach_via.ml.predict import ForecastRequest, forecast

    m = history.matches
    m = m[(m["season"] == season) & (m["status"] == "complete") & m["match_date"].notna()]
    if stage_ids is not None:
        m = m[m["stage_id"].isin(stage_ids)]
    stages = m.groupby("stage_id")["match_date"].min().sort_values()
    arts = []
    evals = []
    for stage_id, first_day in stages.items():
        req = ForecastRequest(forecast_cutoff=match_day_cutoff(first_day), generated_at=generated_at,
                              origin=Origin.REPLAY, season=season, stage_id=str(stage_id))
        art = forecast(history, bundle, req)
        if art.rows.empty:
            continue
        arts.append(art)
        evals.append(evaluate(art, history, min_cohort=min_cohort))
    return arts, pool_evaluations(evals, min_cohort=min_cohort)


def legacy_unknown_summary(history: Any) -> dict[str, Any]:
    """Descriptive summary of imported legacy forecast CSVs (origin legacy_unknown).

    Joined only where the importer resolved an unambiguous player_id AND the claimed
    (season, round label, club) maps to exactly one match. Never merged with prospective
    accuracy: legacy target/cutoff mapping is unverified (AUDIT C01: 'next_round_N' files
    may carry the previous round's feature row).
    """
    lp = history.extra.get("legacy_predictions")
    label = ("legacy_unknown archive: imported legacy forecast CSVs, not prospective; target round "
             "and cutoff are unverified claims from filenames")
    if lp is None or lp.empty:
        return {"origin": "legacy_unknown", "label": label, "rows": 0, "joined": 0, "season_from_timestamp": 0,
                "unjoined_reasons": {}, "metrics": None, "files": 0}
    m = history.matches
    pg = history.player_games
    reasons: dict[str, int] = {}
    out = []
    side = pd.concat([
        m[["match_id", "season", "stage_label", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
        m[["match_id", "season", "stage_label", "away_club_id"]].rename(columns={"away_club_id": "club_id"}),
    ])
    side = side[side["match_id"].isin(m[m["status"] == "complete"]["match_id"])]
    counts = side.groupby(["season", "stage_label", "club_id"])["match_id"].agg(list)
    act = pg.set_index(["match_id", "player_id"])["disposals"]
    from_ts = 0
    for r in _rows(lp):
        if r.player_id is None or pd.isna(r.player_id):
            reasons["unresolved_identity"] = reasons.get("unresolved_identity", 0) + 1
            continue
        season = r.claimed_season
        if season is None or pd.isna(season):
            ts = str(r.claimed_timestamp or "")
            if not ts[:4].isdigit():
                reasons["no_claimed_season"] = reasons.get("no_claimed_season", 0) + 1
                continue
            season = int(ts[:4])  # filename generation-timestamp year (labelled derivation)
            from_ts += 1
        key = (int(season), str(r.claimed_round_label), r.club_id)
        cands = counts.get(key) if key in counts.index else None
        if not cands or len(cands) != 1:
            reasons["claimed_round_unmatched_or_ambiguous"] = reasons.get("claimed_round_unmatched_or_ambiguous", 0) + 1
            continue
        k = (cands[0], r.player_id)
        if k not in act.index:
            reasons["did_not_play_claimed_round"] = reasons.get("did_not_play_claimed_round", 0) + 1
            continue
        a = act.loc[k]
        if pd.isna(a):
            reasons["unknown_actual"] = reasons.get("unknown_actual", 0) + 1
            continue
        out.append((float(r.predicted_value), float(a), r.source_path, int(season), str(r.claimed_round_label)))
    df = pd.DataFrame(out, columns=["pred", "actual", "source_path", "season", "round"])
    by_file = []
    for (sp, s, rd), g in df.groupby(["source_path", "season", "round"]):
        by_file.append({"source_path": sp, "claimed_season": int(str(s)), "claimed_round": rd,
                        **metric_block(g["pred"], g["actual"]).as_dict()})
    return {
        "origin": "legacy_unknown", "label": label, "rows": len(lp), "files": int(lp["source_path"].nunique()),
        "joined": len(df), "unjoined_reasons": reasons, "season_from_timestamp": from_ts,
        "metrics": metric_block(df["pred"], df["actual"]).as_dict() if len(df) else None,
        "by_file": by_file,
        "note": "legacy predictions were rounded integers in the CSVs; metrics describe the archive only",
    }


def to_accuracy_report(ev: EvaluationArtifact, *, model_id: str | None = None, season: int | None = None,
                       promotion: str = "", interval_info: Any = None, rows_resource: str | None = None) -> Any:
    from supercoach_via.publish.view_models import (
        AccuracyReport,
        CohortMetric,
        IntervalInfo,
        MetricBlock,
        Populations,
    )

    def mb(m: Metrics | None) -> MetricBlock | None:
        return None if m is None else MetricBlock(**m.as_dict())

    head = mb(ev.headline)
    assert head is not None
    pops = ev.populations
    return AccuracyReport(
        model_id=model_id or ev.model_id, baseline_id=ev.baseline_model_id, season=season,
        origin=ev.origin,
        label=ev.label, headline=head, baseline_headline=mb(ev.baseline_headline),
        mean_of_rounds_mae=ev.mean_of_rounds_mae,
        populations=Populations(**{k: int(pops[k]) for k in ("intended", "predicted", "joined", "played",
                                                             "missing", "excluded")},
                                exclusion_reasons={k: int(v) for k, v in pops["exclusion_reasons"].items()}),
        cohorts=[CohortMetric(dimension=c.dimension + (" (post-hoc)" if c.post_hoc else ""), cohort=c.cohort,
                              model=MetricBlock(**c.model.as_dict()),
                              baseline=mb(c.baseline), sufficient=c.sufficient) for c in ev.cohorts],
        interval=interval_info or IntervalInfo(available=ev.interval.n > 0, level=None, method=None,
                                               calibrated=ev.interval.calibrated, reason=ev.interval.reason),
        interval_coverage=ev.interval.coverage, interval_median_width=ev.interval.median_width,
        promotion=promotion, notes=list(ev.notes), rows_resource=rows_resource,
    )


def write_evaluation(ev: EvaluationArtifact, root: Any) -> Any:
    """Write an evaluation under ``root/<evaluation_id>/`` (never overwrites)."""
    import json
    from pathlib import Path

    from supercoach_via.ml.bundles import json_safe
    from supercoach_via.storage.snapshots import atomic_write_bytes

    root = Path(root)
    d = root / ev.evaluation_id
    if d.exists():
        return d
    d.mkdir(parents=True)
    ev.scored_rows.to_parquet(d / "scored_rows.parquet", index=False)
    summary = {
        "evaluation_id": ev.evaluation_id, "origin": ev.origin, "label": ev.label,
        "prediction_run_ids": list(ev.prediction_run_ids), "model_id": ev.model_id,
        "baseline_model_id": ev.baseline_model_id, "actuals_snapshot_id": ev.actuals_snapshot_id,
        "metric_version": ev.metric_version, "populations": ev.populations,
        "headline": ev.headline.as_dict(),
        "baseline_headline": ev.baseline_headline.as_dict() if ev.baseline_headline else None,
        "mean_of_rounds_mae": ev.mean_of_rounds_mae,
        "cohorts": [{"dimension": c.dimension, "cohort": c.cohort, "model": c.model.as_dict(),
                     "baseline": c.baseline.as_dict() if c.baseline else None,
                     "sufficient": c.sufficient, "post_hoc": c.post_hoc} for c in ev.cohorts],
        "interval": asdict(ev.interval), "notes": ev.notes,
    }
    atomic_write_bytes(d / "evaluation.json", json.dumps(json_safe(summary), indent=1, sort_keys=True).encode())
    return d
