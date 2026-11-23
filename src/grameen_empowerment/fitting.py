"""Regression fitting for the three Table 3 outcomes (plus the optional income-growth variant).

For each supported outcome three models are fitted on the *same eligible rows*:
``M0_intercept_only`` and ``M1_single_<predictor>`` (repository comparisons) and
``M2_table3_full`` (the specification listed in Table 3). Associations only: nothing here
estimates a causal effect of the programme.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from . import modelspec, schema
from .circularity import assert_not_circular
from .incomegrowth import MISSING, NEGATIVE, ZERO, baseline_problem, percent_change
from .modelspec import ModelConfig, ModelSpecError

MIN_OBS_FLOOR = 30
MIN_OBS_PER_PARAM = 10
FEW_CLUSTERS = 10
MAX_ABS_LOGIT_COEF = 25.0

ESTIMATORS = {
    "continuous": ("ols", "Linear regression: the outcome is a documented continuous measure."),
    "binary": ("logit", "Logistic regression: the outcome is a documented 0/1 response."),
    "ordinal": ("ordered_logit", "Ordered logit: the outcome is a documented ordered category."),
}
SCALE = {"ols": "outcome units per predictor unit", "logit": "log-odds", "ordered_logit": "cumulative log-odds"}


class BlockedError(RuntimeError):
    """Raised before fitting when the dataset cannot support any regression."""


@dataclass
class Prepared:
    frame: pd.DataFrame
    sample: pd.DataFrame
    n_total: int
    estimator: str
    rationale: str
    weight_col: str | None
    cluster_col: str | None
    ordinal_levels: list[str] | None
    notes: list[str]


@dataclass
class OutcomeResult:
    outcome_key: str
    coefficients: pd.DataFrame
    fit_statistics: pd.DataFrame
    sample: pd.DataFrame
    notes: list[str]


def _measure_type(cfg: ModelConfig, meta: dict[str, Any]) -> str:
    if cfg.meta_section == "income":
        return "continuous"
    return str(meta[cfg.meta_section]["measure_type"])


def _flag(reason: pd.Series, mask: pd.Series, label: str) -> None:
    reason[(reason == "") & mask.fillna(False)] = label


def prepare(df: pd.DataFrame, cfg: ModelConfig, meta: dict[str, Any], allow_unweighted: bool = False) -> Prepared:
    """Decide which rows are eligible, and why each other row is not (first failing reason wins)."""
    mtype = _measure_type(cfg, meta)
    estimator, rationale = ESTIMATORS[mtype]
    notes: list[str] = []
    reason = pd.Series("", index=df.index, dtype="object")

    if cfg.derived_outcome:
        d = cfg.derived_outcome
        bp = baseline_problem(df[d["baseline"]])
        for lab in (MISSING, ZERO, NEGATIVE):
            _flag(reason, bp == lab, lab)
        raw = percent_change(df[d["baseline"]], df[d["followup"]])
    else:
        raw = df[cfg.outcome]

    levels: list[str] | None = None
    if mtype == "ordinal":
        levels = [str(x) for x in meta[cfg.meta_section]["levels"]]
        s = raw.astype("object")
        _flag(reason, s.isna(), "missing_outcome")
        _flag(reason, ~s.isin(levels), "invalid_outcome_value")
        y = s.where(s.isin(levels))
    else:
        y = pd.to_numeric(raw, errors="coerce")
        _flag(reason, y.isna(), "missing_outcome")
        if mtype == "binary":
            _flag(reason, ~y.isin([0, 1]), "invalid_outcome_value")

    work = pd.DataFrame(index=df.index)
    for p in cfg.predictors:
        if p in cfg.categorical:
            c = df[p].astype("object")
            _flag(reason, c.isna(), f"missing_predictor:{p}")
            _flag(reason, ~c.isin(cfg.categorical[p].levels), f"unknown_category:{p}")
            work[p] = c
        else:
            v = pd.to_numeric(df[p], errors="coerce")
            _flag(reason, v.isna(), f"missing_predictor:{p}")
            work[p] = v

    weight_col = "survey_weight" if "survey_weight" in df.columns else None
    if weight_col and mtype == "ordinal":
        if not allow_unweighted:
            raise ModelSpecError(
                "survey_weight is present but statsmodels' ordered logit does not support survey weights. "
                "Use a continuous or binary autonomy measure, or pass --allow-unweighted to fit without "
                "weights (the output is then labelled unweighted).")
        notes.append("survey weights ignored: ordered logit in statsmodels has no weight support")
        weight_col = None
    if weight_col:
        w = pd.to_numeric(df[weight_col], errors="coerce")
        _flag(reason, w.isna(), "missing_weight")
        _flag(reason, w <= 0, "nonpositive_weight")
        work["_w"] = w
    cluster_col = next((c for c in ("psu", "household_id") if c in df.columns), None)
    if cluster_col:
        g = df[cluster_col].astype("object")
        _flag(reason, g.isna(), "missing_cluster_id")
        work["_g"] = g

    work["y"] = y
    ok = reason == ""
    frame = work[ok].copy()
    counts = reason[~ok].value_counts().sort_index()
    rows = [{"outcome": cfg.outcome_key, "item": "rows_in_file", "n": len(df)}]
    rows += [{"outcome": cfg.outcome_key, "item": f"excluded:{k}", "n": int(v)} for k, v in counts.items()]
    rows.append({"outcome": cfg.outcome_key, "item": "rows_eligible", "n": int(ok.sum())})
    return Prepared(frame, pd.DataFrame(rows), len(df), estimator, rationale, weight_col, cluster_col, levels, notes)


def _check_variation(prep: Prepared, cfg: ModelConfig) -> None:
    f = prep.frame
    n = len(f)
    if f["y"].nunique() < 2:
        raise ModelSpecError(f"outcome '{cfg.outcome}' has no variation among the {n} eligible rows")
    for p in cfg.predictors:
        if p in cfg.categorical:
            spec = cfg.categorical[p]
            present = [lv for lv in spec.levels if (f[p] == lv).any()]
            if spec.reference not in present:
                raise ModelSpecError(f"reference level '{spec.reference}' of '{p}' has no eligible rows; "
                                     "change the reference level in the model config")
            if len(present) < 2:
                raise ModelSpecError(f"categorical predictor '{p}' has only one observed level among eligible rows")
        elif f[p].nunique() < 2:
            raise ModelSpecError(f"predictor '{p}' is constant among the {n} eligible rows")
    n_params = sum(max(1, f[p].nunique() - 1) if p in cfg.categorical else 1 for p in cfg.predictors) + 1
    needed = max(MIN_OBS_FLOOR, MIN_OBS_PER_PARAM * n_params)
    if n < needed:
        raise ModelSpecError(f"only {n} eligible rows for {n_params} parameters; at least {needed} are required")


def build_design(frame: pd.DataFrame, cfg: ModelConfig, predictors: tuple[str, ...]) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for p in predictors:
        if p in cfg.categorical:
            spec = cfg.categorical[p]
            for lv in spec.levels:
                if lv != spec.reference and (frame[p] == lv).any():
                    cols[f"{p}[{lv}]"] = (frame[p] == lv).astype(float)
        else:
            cols[p] = frame[p].astype(float)
    return pd.DataFrame(cols, index=frame.index)


def _check_rank(X: pd.DataFrame, with_constant: bool) -> None:
    M = sm.add_constant(X, has_constant="add") if with_constant else X
    if M.shape[1] and np.linalg.matrix_rank(M.to_numpy()) < M.shape[1]:
        raise ModelSpecError(f"design matrix is rank deficient (collinear columns among {list(M.columns)})")


def _groups(prep: Prepared) -> np.ndarray | None:
    if not prep.cluster_col:
        return None
    codes, uniques = pd.factorize(prep.frame["_g"])
    if len(uniques) < FEW_CLUSTERS:
        prep.notes.append(f"only {len(uniques)} clusters in '{prep.cluster_col}': robust SEs are unreliable")
    return codes


def _coef_table(res: Any, estimator: str, labels: dict[str, str] | None = None) -> pd.DataFrame:
    ci = res.conf_int()
    ci = pd.DataFrame(np.asarray(ci), index=res.params.index, columns=["ci_low", "ci_high"])
    out = pd.DataFrame({
        "term": [str(t) for t in res.params.index], "estimate": res.params.to_numpy(),
        "std_error": res.bse.to_numpy(), "statistic": res.tvalues.to_numpy(),
        "p_value": res.pvalues.to_numpy(), "ci_low": ci["ci_low"].to_numpy(), "ci_high": ci["ci_high"].to_numpy(),
    })
    out["term_type"] = np.where(out["term"] == "const", "intercept", "predictor")
    out["term"] = out["term"].replace({"const": "intercept"})
    if labels:
        out["term_type"] = np.where(out["term"].isin(labels), "threshold", out["term_type"])
        out["term"] = out["term"].replace(labels)
    if estimator in ("logit", "ordered_logit"):
        pred = out["term_type"] == "predictor"
        for src, dst in (("estimate", "odds_ratio"), ("ci_low", "or_ci_low"), ("ci_high", "or_ci_high")):
            out[dst] = np.where(pred, np.exp(out[src]), np.nan)
    return out


def _cov_kwargs(groups: np.ndarray | None) -> dict[str, Any]:
    if groups is None:
        return {"cov_type": "HC1"}
    return {"cov_type": "cluster", "cov_kwds": {"groups": groups}}


def _fit_linear_or_logit(prep: Prepared, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    Xc = sm.add_constant(X, has_constant="add")
    y = prep.frame["y"].astype(float)
    w = prep.frame["_w"].astype(float) if prep.weight_col else None
    groups = _groups(prep)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if prep.estimator == "ols":
            model = sm.WLS(y, Xc, weights=w) if w is not None else sm.OLS(y, Xc)
        else:
            model = sm.GLM(y, Xc, family=sm.families.Binomial(), freq_weights=w)
        res = model.fit(**_cov_kwargs(groups))
    if any(issubclass(c.category, PerfectSeparationWarning) for c in caught) or (
            prep.estimator == "logit" and (~np.isfinite(res.params)).any() or
            prep.estimator == "logit" and (res.params.abs() > MAX_ABS_LOGIT_COEF).any()):
        raise ModelSpecError("perfect or quasi-perfect separation in the logistic model; "
                             "simplify the specification or collect more data")
    weighted = w is not None
    n, k = len(y), len(res.params)
    stats: dict[str, float] = {
        "llf": float(res.llf),
        "aic": np.nan if weighted else float(-2 * res.llf + 2 * k),
        "bic": np.nan if weighted else float(-2 * res.llf + k * np.log(n)),
    }
    if prep.estimator == "ols":
        stats.update(r2=float(res.rsquared), adj_r2=float(res.rsquared_adj), rmse=float(np.sqrt(res.mse_resid)))
    else:
        stats["mcfadden_r2"] = 0.0 if X.shape[1] == 0 else float(1 - res.llf / float(res.llnull))
    return _coef_table(res, prep.estimator), stats


def _threshold_labels(levels: list[str]) -> dict[str, str]:
    return {f"{i}/{i + 1}": f"threshold_param:{levels[i]}|{levels[i + 1]}" for i in range(len(levels) - 1)}


def _ordered_intercept_only(prep: Prepared, levels: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    y = prep.frame["y"]
    counts = np.array([(y == lv).sum() for lv in levels], dtype=float)
    p = counts / counts.sum()
    cuts = np.log(np.cumsum(p)[:-1] / (1 - np.cumsum(p)[:-1]))
    params = np.concatenate([[cuts[0]], np.log(np.diff(cuts))]) if len(cuts) > 1 else cuts
    llf = float(np.sum(counts * np.log(p)))
    k, n = len(levels) - 1, float(counts.sum())
    table = pd.DataFrame({
        "term": list(_threshold_labels(levels).values()), "estimate": params, "std_error": np.nan,
        "statistic": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "term_type": "threshold",
        "odds_ratio": np.nan, "or_ci_low": np.nan, "or_ci_high": np.nan,
    })
    stats = {"llf": llf, "aic": 2 * k - 2 * llf, "bic": k * np.log(n) - 2 * llf, "mcfadden_r2": 0.0}
    return table, stats


def _fit_ordered(prep: Prepared, X: pd.DataFrame, llnull: float) -> tuple[pd.DataFrame, dict[str, float]]:
    observed = [lv for lv in prep.ordinal_levels or [] if (prep.frame["y"] == lv).any()]
    if len(observed) < 2:
        raise ModelSpecError("ordinal outcome has fewer than two observed levels among eligible rows")
    if len(observed) < len(prep.ordinal_levels or []):
        prep.notes.append(f"unused outcome levels dropped: {sorted(set(prep.ordinal_levels) - set(observed))}")
    if X.shape[1] == 0:
        return _ordered_intercept_only(prep, observed)
    yc = pd.Series(pd.Categorical(prep.frame["y"], categories=observed, ordered=True), index=prep.frame.index)
    groups = _groups(prep)
    kwargs: dict[str, Any] = {"method": "bfgs", "disp": False, "maxiter": 1000}
    if groups is not None:
        kwargs.update(cov_type="cluster", cov_kwds={"groups": groups})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = OrderedModel(yc, X, distr="logit").fit(**kwargs)
    if not bool(res.mle_retvals.get("converged", True)):
        raise ModelSpecError("ordered logit did not converge")
    table = _coef_table(res, "ordered_logit", _threshold_labels(observed))
    stats = {"llf": float(res.llf), "aic": float(res.aic), "bic": float(res.bic),
             "mcfadden_r2": float(1 - res.llf / llnull)}
    return table, stats


def fit_outcome(df: pd.DataFrame, cfg: ModelConfig, meta: dict[str, Any], *, allow_unweighted: bool = False) -> OutcomeResult:
    """Fit M0, M1 and M2 on identical rows; raises ModelSpecError with an actionable message."""
    if cfg.outcome_key == "social_engagement":
        assert_not_circular(meta)
    prep = prepare(df, cfg, meta, allow_unweighted=allow_unweighted)
    _check_variation(prep, cfg)
    plan = [("M0_intercept_only", "repository_comparison", ()),
            (cfg.single_predictor_id, "repository_comparison", (cfg.comparison_predictor,)),
            ("M2_table3_full", "table3_specification", cfg.predictors)]
    coefs, fits = [], []
    llnull = np.nan
    for model_id, role, preds in plan:
        X = build_design(prep.frame, cfg, preds)
        _check_rank(X, with_constant=prep.estimator != "ordered_logit")
        if prep.estimator == "ordered_logit":
            table, stats = _fit_ordered(prep, X, llnull)
            if model_id == "M0_intercept_only":
                llnull = stats["llf"]
        else:
            table, stats = _fit_linear_or_logit(prep, X)
        common = {"outcome": cfg.outcome_key, "outcome_variable": cfg.outcome, "model_id": model_id,
                  "model_role": role, "estimator": prep.estimator, "estimator_rationale": prep.rationale,
                  "estimate_scale": SCALE[prep.estimator], "n_obs": len(prep.frame),
                  "weights_used": bool(prep.weight_col), "cluster_used": prep.cluster_col or "none",
                  "cov_type": "cluster" if prep.cluster_col else ("HC1" if prep.estimator != "ordered_logit" else "nonrobust")}
        coefs.append(table.assign(**common))
        fits.append(pd.DataFrame([{**common, "n_params": len(table), **stats}]))
    notes = list(dict.fromkeys(prep.notes))
    if prep.weight_col:
        notes.append("weights used as analytic/frequency weights; stratification is not modelled and AIC/BIC are not reported")
    return OutcomeResult(cfg.outcome_key, pd.concat(coefs, ignore_index=True), pd.concat(fits, ignore_index=True), prep.sample, notes)


@dataclass
class FitSummary:
    coefficients: pd.DataFrame
    fit_statistics: pd.DataFrame
    sample_accounting: pd.DataFrame
    status: pd.DataFrame
    files: list[Path]


def _stamp(table: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    dtype = meta.get("dataset_type", "unknown")
    out = table.copy()
    out.insert(0, "dataset_id", meta.get("dataset_id", "unnamed"))
    out.insert(0, "dataset_type", dtype)
    out.insert(2, "is_synthetic", dtype == "synthetic")
    return out


def fit_dataset(df: pd.DataFrame, meta: dict[str, Any], outdir: Path, *, outcomes: list[str] | None = None,
                configs: dict[str, ModelConfig] | None = None, allow_unweighted: bool = False) -> FitSummary:
    """Validate, fit every supported outcome, and write tidy CSVs to ``outdir``."""
    report = schema.validate_microdata(df, meta)
    if not report.ok:
        raise BlockedError("regression blocked:\n" + report.render())
    configs = configs or modelspec.load_all_configs()
    wanted = outcomes or [o for o in configs if o != "income_growth" or o in report.supported_outcomes]
    status, coefs, fits, samples = [], [], [], []
    for key in wanted:
        if key not in configs:
            status.append({"outcome": key, "status": "blocked", "message": "no model config for this outcome"})
            continue
        if key not in report.supported_outcomes:
            status.append({"outcome": key, "status": "blocked", "message": "; ".join(report.blocked_outcomes.get(key, ["unsupported"]))})
            continue
        try:
            res = fit_outcome(df, configs[key], meta, allow_unweighted=allow_unweighted)
        except (ModelSpecError, ValueError) as exc:
            status.append({"outcome": key, "status": "blocked", "message": str(exc)})
            continue
        coefs.append(res.coefficients)
        fits.append(res.fit_statistics)
        samples.append(res.sample)
        status.append({"outcome": key, "status": "fitted", "message": " | ".join(res.notes)})
    if not coefs:
        raise BlockedError("regression blocked: no outcome could be fitted\n" + "\n".join(
            f"  {s['outcome']}: {s['message']}" for s in status))
    summary = FitSummary(
        _stamp(pd.concat(coefs, ignore_index=True), meta), _stamp(pd.concat(fits, ignore_index=True), meta),
        _stamp(pd.concat(samples, ignore_index=True), meta), _stamp(pd.DataFrame(status), meta), [])
    outdir.mkdir(parents=True, exist_ok=True)
    for name, table in (("coefficients", summary.coefficients), ("fit_statistics", summary.fit_statistics),
                        ("sample_accounting", summary.sample_accounting), ("model_status", summary.status)):
        path = outdir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
        summary.files.append(path)
    return summary
