from __future__ import annotations

import copy

import pandas as pd
import pytest

from grameen_empowerment import published, schema
from grameen_empowerment.circularity import (
    CircularOutcomeError,
    assert_not_circular,
    circularity_problems,
)
from grameen_empowerment.fitting import BlockedError, fit_dataset


def test_bundled_synthetic_fixture_is_valid(report_ok):
    assert report_ok.ok
    assert set(report_ok.supported_outcomes) == {"income_level", "income_growth", "decision_autonomy", "social_engagement"}


def test_published_aggregates_are_refused_not_fitted(tmp_path):
    agg = published.load_values()
    assert schema.looks_like_aggregate(agg)
    rep = schema.validate_microdata(agg, {"dataset_type": "synthetic"})
    assert not rep.ok
    assert any("published aggregates" in e for e in rep.errors)
    with pytest.raises(BlockedError, match="aggregates"):
        fit_dataset(agg, {"dataset_type": "synthetic", "dataset_id": "x"}, tmp_path)


def test_aggregate_dataset_type_cannot_be_fitted(plain_frame, plain_meta):
    plain_meta["dataset_type"] = "published_aggregate"
    assert any("aggregate type" in e for e in schema.validate_microdata(plain_frame, plain_meta).errors)


def test_unknown_dataset_type_is_an_error(plain_frame, plain_meta):
    plain_meta["dataset_type"] = "made_up"
    assert not schema.validate_microdata(plain_frame, plain_meta).ok


def test_missing_required_column_blocks_only_the_affected_outcome(plain_frame, plain_meta):
    rep = schema.validate_microdata(plain_frame.drop(columns=["loan_size"]), plain_meta)
    assert rep.ok
    assert "income_level" in rep.blocked_outcomes
    assert any("loan_size" in r for r in rep.blocked_outcomes["income_level"])
    assert "decision_autonomy" in rep.supported_outcomes


def test_no_supported_outcome_is_an_actionable_error(plain_meta):
    df = pd.DataFrame({"borrower_id": ["a", "b"], "unrelated": [1, 2]})
    rep = schema.validate_microdata(df, plain_meta)
    assert not rep.ok
    assert any("ACCESS_ORIGINAL_DATA" in e for e in rep.errors)


def test_duplicate_or_missing_borrower_ids_are_errors(plain_frame, plain_meta):
    dup = plain_frame.copy()
    dup.loc[1, "borrower_id"] = dup.loc[0, "borrower_id"]
    assert any("unique" in e for e in schema.validate_microdata(dup, plain_meta).errors)
    miss = plain_frame.copy()
    miss.loc[0, "borrower_id"] = None
    assert any("missing" in e for e in schema.validate_microdata(miss, plain_meta).errors)


def test_income_documentation_is_required(plain_frame, plain_meta):
    meta = copy.deepcopy(plain_meta)
    del meta["income"]["currency"]
    rep = schema.validate_microdata(plain_frame, meta)
    assert any("currency" in r for r in rep.blocked_outcomes["income_level"])


def test_income_growth_requires_comparable_baseline(synth_df, synth_meta):
    meta = copy.deepcopy(synth_meta)
    meta["income"]["baseline_comparable"] = False
    rep = schema.validate_microdata(synth_df, meta)
    assert any("baseline_comparable" in r for r in rep.blocked_outcomes["income_growth"])
    assert "income_level" in rep.supported_outcomes


def test_autonomy_measure_must_be_documented(plain_frame, plain_meta):
    meta = copy.deepcopy(plain_meta)
    meta["autonomy"]["measure_type"] = "mystery"
    assert "decision_autonomy" in schema.validate_microdata(plain_frame, meta).blocked_outcomes
    meta = copy.deepcopy(plain_meta)
    meta["autonomy"]["measure_type"] = "ordinal"
    assert any("levels" in r for r in schema.validate_microdata(plain_frame, meta).blocked_outcomes["decision_autonomy"])


def test_synthetic_data_must_be_marked(synth_df, synth_meta):
    rep = schema.validate_microdata(synth_df.drop(columns=["is_synthetic"]), synth_meta)
    assert any("is_synthetic" in e for e in rep.errors)


def test_non_positive_weights_warn(synth_df, synth_meta):
    df = synth_df.copy()
    df.loc[0, "survey_weight"] = 0
    assert any("survey_weight" in w for w in schema.validate_microdata(df, synth_meta).warnings)


def test_circular_engagement_outcome_is_flagged_and_blocked(plain_frame, plain_meta):
    meta = copy.deepcopy(plain_meta)
    meta["social_engagement"]["components"] = ["social_group_membership", "community_meeting_attendance", "leadership_role"]
    assert any("circular" in p for p in circularity_problems(meta))
    rep = schema.validate_microdata(plain_frame, meta)
    assert any("circular" in r for r in rep.blocked_outcomes["social_engagement"])
    with pytest.raises(CircularOutcomeError):
        assert_not_circular(meta)


def test_engagement_components_must_be_declared(plain_frame, plain_meta):
    meta = copy.deepcopy(plain_meta)
    del meta["social_engagement"]["components"]
    assert any("components" in r for r in schema.validate_microdata(plain_frame, meta).blocked_outcomes["social_engagement"])
    assert circularity_problems({}) == ["social_engagement metadata section is missing"]


def test_independent_engagement_outcome_passes(plain_meta):
    assert circularity_problems(plain_meta) == []


def test_schema_markdown_lists_every_field():
    md = schema.schema_markdown()
    assert all(f"`{k}`" in md for k in schema.FIELD_DICTIONARY)


def test_meta_must_be_a_mapping(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        schema.load_meta(p)
