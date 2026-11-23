from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grameen_empowerment import agegroups, normalization
from grameen_empowerment.incomegrowth import ZERO, baseline_problem, percent_change


def test_figure4_labels_are_table6_values_divided_by_110():
    shares = normalization.normalized_shares({"a": 25, "b": 35, "c": 50})
    assert sum([25, 35, 50]) == 110
    assert [normalization.printed_label(v) for v in shares.values()] == [22.7, 31.8, 45.5]
    assert sum(shares.values()) == pytest.approx(100.0)


def test_printed_pie_labels_match_the_recomputation():
    comp = normalization.compare_with_printed()
    assert comp["matches_printed_label"].all()
    assert set(comp["sum_of_table6_values"]) == {110.0}


def test_pie_labels_are_not_a_borrower_distribution():
    assert "not the distribution of borrowers" in normalization.CAPTION


def test_normalization_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        normalization.normalized_shares({"a": 0, "b": 0})


@pytest.mark.parametrize(("age", "label"), [
    (18, "Under 30 years"), (29, "Under 30 years"), (29.9, "Under 30 years"), (30, "30-45 years"),
    (44, "30-45 years"), (45, "30-45 years"), (45.5, "Over 45 years"), (46, "Over 45 years"),
    (80, "Over 45 years")])
def test_age_boundaries_follow_table5(age, label):
    assert agegroups.assign_age_group(age) == label


def test_age_45_belongs_to_30_45():
    assert agegroups.assign_age_group(45) == "30-45 years"


@pytest.mark.parametrize("bad", [np.nan, -1, 121])
def test_missing_or_implausible_ages_are_not_guessed(bad):
    assert pd.isna(agegroups.assign_age_group(bad))


def test_age_series_keeps_index():
    out = agegroups.assign_age_group(pd.Series([29, 30, 45, 46], index=[5, 6, 7, 8]))
    assert list(out) == ["Under 30 years", "30-45 years", "30-45 years", "Over 45 years"]
    assert list(out.index) == [5, 6, 7, 8]


def test_percent_change_is_undefined_at_zero_baseline():
    b = pd.Series([100.0, 0.0, -5.0, np.nan, 200.0])
    f = pd.Series([150.0, 50.0, 10.0, 10.0, 100.0])
    pc = percent_change(b, f)
    assert pc.iloc[0] == pytest.approx(50.0)
    assert pc.iloc[4] == pytest.approx(-50.0)
    assert pc.iloc[1:4].isna().all()


def test_baseline_problem_names_the_reason():
    reasons = baseline_problem(pd.Series([0.0, 5.0, -1.0, np.nan]))
    assert list(reasons) == [ZERO, "", "negative_baseline_income", "missing_baseline_income"]
