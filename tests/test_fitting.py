from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from grameen_empowerment import modelspec
from grameen_empowerment.descriptive import descriptive_tables, group_summary
from grameen_empowerment.fitting import BlockedError, fit_dataset, fit_outcome
from grameen_empowerment.modelspec import ModelSpecError


def _coef(res, outcome, model, term):
    c = res.coefficients
    return float(c[(c["outcome"] == outcome) & (c["model_id"] == model) & (c["term"] == term)]["estimate"].iloc[0])


def test_ols_matches_numpy_least_squares(plain_frame, plain_meta, configs):
    res = fit_outcome(plain_frame, configs["decision_autonomy"], plain_meta)
    X = np.column_stack([np.ones(len(plain_frame)), plain_frame["training_participation"], plain_frame["group_membership"]])
    beta, *_ = np.linalg.lstsq(X, plain_frame["autonomy"].to_numpy(), rcond=None)
    got = [_coef(res, "decision_autonomy", "M2_table3_full", t) for t in ("intercept", "training_participation", "group_membership")]
    assert got == pytest.approx(list(beta), abs=1e-8)
    assert got[1] == pytest.approx(4.0, abs=0.4)  # recovers the planted effect


def test_three_models_use_identical_rows_and_roles(plain_frame, plain_meta, configs):
    df = plain_frame.copy()
    df.loc[:9, "loan_tenure_months"] = np.nan  # missing only in a predictor of the full model
    res = fit_outcome(df, configs["income_level"], plain_meta)
    fits = res.fit_statistics
    assert set(fits["n_obs"]) == {390}
    assert list(fits["model_id"]) == ["M0_intercept_only", "M1_single_loan_size", "M2_table3_full"]
    assert list(fits["model_role"]) == ["repository_comparison", "repository_comparison", "table3_specification"]
    acc = res.sample.set_index("item")["n"]
    assert acc["rows_in_file"] - acc["excluded:missing_predictor:loan_tenure_months"] == acc["rows_eligible"] == 390


def test_exclusion_reasons_are_counted_and_exclusive(plain_frame, plain_meta, configs):
    df = plain_frame.copy()
    df.loc[:4, "income_followup"] = np.nan
    df.loc[5:9, "loan_size"] = np.nan
    df.loc[10:12, "education_level"] = "phd"
    df.loc[13, "education_level"] = None
    res = fit_outcome(df, configs["income_level"], plain_meta)
    acc = res.sample.set_index("item")["n"]
    assert acc["excluded:missing_outcome"] == 5
    assert acc["excluded:missing_predictor:loan_size"] == 5
    assert acc["excluded:unknown_category:education_level"] == 3
    assert acc["excluded:missing_predictor:education_level"] == 1
    assert acc["rows_eligible"] == len(df) - 14


def test_zero_baseline_rows_are_excluded_from_income_growth(synth_df, synth_meta, configs):
    zero = int((synth_df["income_baseline"] == 0).sum())
    assert zero > 0
    res = fit_outcome(synth_df, configs["income_growth"], synth_meta, allow_unweighted=True)
    acc = res.sample.set_index("item")["n"]
    assert acc["excluded:zero_baseline_income"] == zero
    assert set(res.fit_statistics["n_obs"]) == {int(acc["rows_eligible"])}
    assert np.isfinite(res.coefficients["estimate"]).all()


def test_estimator_follows_the_documented_measure_type(plain_frame, plain_meta, configs):
    rng = np.random.default_rng(3)
    meta = copy.deepcopy(plain_meta)
    df = plain_frame.copy()
    assert fit_outcome(df, configs["decision_autonomy"], meta).fit_statistics["estimator"].iloc[0] == "ols"
    latent = 0.8 * df["training_participation"] + 0.5 * df["group_membership"] + rng.logistic(size=len(df))
    df["autonomy"] = (latent > 0.5).astype(float)
    meta["autonomy"]["measure_type"] = "binary"
    res = fit_outcome(df, configs["decision_autonomy"], meta)
    assert res.fit_statistics["estimator"].iloc[0] == "logit"
    assert {"odds_ratio", "or_ci_low"} <= set(res.coefficients.columns)
    assert _coef(res, "decision_autonomy", "M2_table3_full", "training_participation") > 0
    df["autonomy"] = pd.cut(latent, [-9, -0.5, 1.0, 9], labels=["low", "medium", "high"]).astype(str)
    meta["autonomy"].update(measure_type="ordinal", levels=["low", "medium", "high"])
    res = fit_outcome(df, configs["decision_autonomy"], meta)
    assert res.fit_statistics["estimator"].iloc[0] == "ordered_logit"
    assert any(t.startswith("threshold_param") for t in res.coefficients["term"])
    assert (res.fit_statistics["mcfadden_r2"] >= 0).all()


def test_weighted_ols_matches_the_closed_form(plain_frame, plain_meta, configs):
    rng = np.random.default_rng(5)
    df = plain_frame.copy()
    df["survey_weight"] = rng.uniform(0.5, 2.0, len(df))
    res = fit_outcome(df, configs["decision_autonomy"], plain_meta)
    assert bool(res.fit_statistics["weights_used"].all())
    X = np.column_stack([np.ones(len(df)), df["training_participation"], df["group_membership"]])
    W = np.diag(df["survey_weight"].to_numpy())
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ df["autonomy"].to_numpy())
    got = [_coef(res, "decision_autonomy", "M2_table3_full", t) for t in ("intercept", "training_participation", "group_membership")]
    assert got == pytest.approx(list(beta), abs=1e-8)
    assert res.fit_statistics["aic"].isna().all()  # AIC is not defined under weights


def test_clusters_are_used_when_present(plain_frame, plain_meta, configs):
    df = plain_frame.assign(household_id=[f"H{i // 3}" for i in range(len(plain_frame))])
    res = fit_outcome(df, configs["decision_autonomy"], plain_meta)
    assert set(res.fit_statistics["cluster_used"]) == {"household_id"}
    assert set(res.fit_statistics["cov_type"]) == {"cluster"}


def test_few_clusters_are_noted(plain_frame, plain_meta, configs):
    df = plain_frame.assign(psu=[f"V{i % 4}" for i in range(len(plain_frame))])
    assert any("clusters" in n for n in fit_outcome(df, configs["decision_autonomy"], plain_meta).notes)


def test_ordinal_with_weights_is_blocked_unless_unweighted_allowed(synth_meta, configs):
    from grameen_empowerment import synthetic

    df = synthetic.generate(autonomy_type="ordinal")
    meta = synthetic.metadata(len(df), synthetic.DEFAULT_SEED, "ordinal")
    with pytest.raises(ModelSpecError, match="survey weights"):
        fit_outcome(df, configs["decision_autonomy"], meta)
    res = fit_outcome(df, configs["decision_autonomy"], meta, allow_unweighted=True)
    assert not res.fit_statistics["weights_used"].any()
    assert any("ignored" in n for n in res.notes)


def test_rank_deficiency_is_detected(plain_frame, plain_meta, configs):
    df = plain_frame.copy()
    df["loan_tenure_months"] = 2.0 * df["loan_size"]
    with pytest.raises(ModelSpecError, match="rank deficient"):
        fit_outcome(df, configs["income_level"], plain_meta)


def test_absent_variation_is_detected(plain_frame, plain_meta, configs):
    df = plain_frame.assign(training_participation=1.0)
    with pytest.raises(ModelSpecError, match="constant"):
        fit_outcome(df, configs["decision_autonomy"], plain_meta)
    with pytest.raises(ModelSpecError, match="no variation"):
        fit_outcome(plain_frame.assign(autonomy=5.0), configs["decision_autonomy"], plain_meta)


def test_missing_reference_level_is_detected(plain_frame, plain_meta, configs):
    df = plain_frame[plain_frame["education_level"] != "none"]
    with pytest.raises(ModelSpecError, match="reference level"):
        fit_outcome(df, configs["income_level"], plain_meta)


def test_single_level_categorical_is_detected(plain_frame, plain_meta, configs):
    with pytest.raises(ModelSpecError, match="only one observed level"):
        fit_outcome(plain_frame.assign(education_level="none"), configs["income_level"], plain_meta)


def test_perfect_separation_is_reported(plain_frame, plain_meta, configs):
    meta = copy.deepcopy(plain_meta)
    meta["autonomy"]["measure_type"] = "binary"
    df = plain_frame.assign(autonomy=plain_frame["training_participation"])
    with pytest.raises(ModelSpecError, match="separation"):
        fit_outcome(df, configs["decision_autonomy"], meta)


def test_too_few_rows_is_reported(plain_frame, plain_meta, configs):
    with pytest.raises(ModelSpecError, match="eligible rows"):
        fit_outcome(plain_frame.head(25), configs["decision_autonomy"], plain_meta)


def test_binary_outcome_values_are_checked(plain_frame, plain_meta, configs):
    meta = copy.deepcopy(plain_meta)
    meta["autonomy"]["measure_type"] = "binary"
    df = plain_frame.assign(autonomy=np.where(np.arange(len(plain_frame)) % 2, 1.0, 7.0))
    with pytest.raises(ModelSpecError):
        fit_outcome(df, configs["decision_autonomy"], meta)


def test_circular_engagement_outcome_cannot_be_fitted(plain_frame, plain_meta, configs):
    meta = copy.deepcopy(plain_meta)
    meta["social_engagement"]["components"] = ["leadership_role", "x"]
    with pytest.raises(ValueError, match="circular"):
        fit_outcome(plain_frame, configs["social_engagement"], meta)


def test_fit_dataset_writes_tidy_synthetic_labelled_tables(tmp_path, synth_df, synth_meta):
    summary = fit_dataset(synth_df, synth_meta, tmp_path, allow_unweighted=True)
    assert {p.name for p in summary.files} == {"coefficients.csv", "fit_statistics.csv", "sample_accounting.csv", "model_status.csv"}
    for name in ("coefficients", "fit_statistics", "sample_accounting", "model_status"):
        t = pd.read_csv(tmp_path / f"{name}.csv")
        assert set(t["dataset_type"]) == {"synthetic"}
        assert t["is_synthetic"].all()
    coefs = pd.read_csv(tmp_path / "coefficients.csv")
    assert {"estimate", "std_error", "ci_low", "ci_high", "p_value", "n_obs", "estimator", "estimator_rationale"} <= set(coefs.columns)
    assert (summary.status["status"] == "fitted").all()


def test_fit_dataset_reports_blocked_outcomes_and_keeps_going(tmp_path, plain_frame, plain_meta):
    summary = fit_dataset(plain_frame.drop(columns=["autonomy"]), plain_meta, tmp_path)
    status = summary.status.set_index("outcome")["status"]
    assert status["decision_autonomy"] == "blocked"
    assert status["income_level"] == "fitted"


def test_fit_dataset_blocks_when_nothing_can_be_fitted(tmp_path, plain_frame, plain_meta):
    with pytest.raises(BlockedError):
        fit_dataset(plain_frame, plain_meta, tmp_path, outcomes=["nonexistent"])


def test_synthetic_and_real_results_are_distinguished_by_label(tmp_path, plain_frame, plain_meta):
    summary = fit_dataset(plain_frame, plain_meta, tmp_path)
    assert set(summary.coefficients["dataset_type"]) == {"original_microdata"}
    assert not summary.coefficients["is_synthetic"].any()


def test_config_validation():
    base = {"outcome_key": "k", "paper_reference": "r", "outcome": "y", "predictors": ["a", "b"],
            "comparison_predictor": "a", "meta_section": "income"}
    assert modelspec.parse_config(base).single_predictor_id == "M1_single_a"
    with pytest.raises(ModelSpecError):
        modelspec.parse_config({**base, "comparison_predictor": "zzz"})
    with pytest.raises(ModelSpecError):
        modelspec.parse_config({k: v for k, v in base.items() if k != "outcome"})
    with pytest.raises(ModelSpecError):
        modelspec.parse_config({**base, "categorical": {"a": {"reference": "x", "levels": ["y", "z"]}}})
    with pytest.raises(ModelSpecError):
        modelspec.parse_config({**base, "categorical": {"q": {"reference": "x", "levels": ["x"]}}})


def test_one_config_per_table3_outcome(configs):
    assert {"income_level", "decision_autonomy", "social_engagement"} <= set(configs)
    assert configs["income_level"].predictors == ("loan_size", "loan_tenure_months", "education_level")
    assert configs["decision_autonomy"].predictors == ("training_participation", "group_membership")
    assert configs["social_engagement"].predictors == ("community_meeting_attendance", "leadership_role")
    assert configs["income_level"].derived_outcome is None  # measured income, not growth


def test_descriptive_overall_is_computed_from_rows_not_group_means():
    df = pd.DataFrame({"v": [10.0] * 9 + [40.0], "g": ["a"] * 9 + ["b"]})
    out = group_summary(df, "v", "g").set_index("group")
    assert out.loc["Overall", "n"] == 10
    assert out.loc["Overall", "mean"] == pytest.approx(13.0)
    assert out.loc["Overall", "mean"] != pytest.approx((out.loc["a", "mean"] + out.loc["b", "mean"]) / 2)


def test_descriptive_tables_use_table5_age_boundaries(synth_df, synth_meta):
    tables = descriptive_tables(synth_df, synth_meta)
    assert {"income_growth_pct_by_sector", "autonomy_level_by_age_group", "engagement_by_education"} <= set(tables)
    groups = set(tables["autonomy_level_by_age_group"]["group"])
    assert {"Under 30 years", "30-45 years", "Over 45 years", "Overall"} == groups
