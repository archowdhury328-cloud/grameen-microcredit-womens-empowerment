from __future__ import annotations

import pandas as pd
import pytest

from grameen_empowerment import published


@pytest.fixture()
def df() -> pd.DataFrame:
    return published.load_values()


def test_transcription_matches_the_paper(df):
    assert published.validate_values(df) == []


@pytest.mark.parametrize(("table", "label", "value"), [
    ("Table 4", "Small Business", 32), ("Table 4", "Agriculture", 22), ("Table 4", "Overall", 28),
    ("Table 5", "Under 30 years", 40), ("Table 5", "30-45 years", 30), ("Table 5", "Over 45 years", 20),
    ("Table 6", "No formal education", 25), ("Table 6", "Primary education", 35),
    ("Table 6", "Secondary education", 50),
])
def test_exact_reported_values_on_page_2585(df, table, label, value):
    row = df[(df["source_table"] == table) & (df["group_label"] == label)].iloc[0]
    assert float(row["value"]) == value
    assert int(row["printed_page"]) == 2585


def test_overall_is_its_own_value_not_the_mean_of_groups(df):
    t4 = df[df["source_table"] == "Table 4"].set_index("group_label")["value"].astype(float)
    assert t4["Overall"] == 28
    assert (t4["Small Business"] + t4["Agriculture"]) / 2 == 27  # an unweighted mean would be wrong
    assert t4["Overall"] != 27


def test_table6_keeps_education_and_location_on_each_row(df):
    t6 = df[df["source_table"] == "Table 6"]
    assert dict(zip(t6["group_label"], t6["secondary_label"], strict=True)) == {
        "No formal education": "Rural", "Primary education": "Mixed", "Secondary education": "Urban"}


@pytest.mark.parametrize(("rid", "value", "page"), [("N-01", 28, 2582), ("N-04", 35, 2583), ("N-07", 40, 2583)])
def test_narrative_values_have_their_own_location_and_wording(df, rid, value, page):
    row = df[df["record_id"] == rid].iloc[0]
    assert float(row["value"]) == value
    assert int(row["printed_page"]) == page
    assert row["source_location"]
    assert row["wording"]


def test_denominators_are_marked_unavailable(df):
    assert set(df["denominator_n"]) == {"unavailable"}


def test_percent_increase_is_kept_distinct_from_percentage_point(df):
    assert set(df["change_basis"]) <= published.CHANGE_BASES
    assert "percentage_point_change" not in set(df["change_basis"])  # the paper never states one
    assert set(df[df["source_table"] == "Table 4"]["change_basis"]) == {"percent_increase"}
    assert set(df[df["source_table"] == "Table 5"]["change_basis"]) == {"not_stated"}


def test_table3_misreference_is_recorded(df):
    assert "Table 3" in df[df["record_id"] == "N-07"].iloc[0]["notes"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.assign(value=d["value"].where(d["record_id"] != "T4-01", 33)),
    lambda d: d.assign(printed_page=d["printed_page"].where(d["record_id"] != "T5-02", 2586)),
    lambda d: d.assign(secondary_label=d["secondary_label"].where(d["record_id"] != "T6-01", "Urban")),
    lambda d: d.assign(denominator_n=100),
    lambda d: d[d["record_id"] != "N-04"],
    lambda d: pd.concat([d, d.iloc[[0]]]),
])
def test_validation_catches_each_kind_of_transcription_error(df, mutate):
    assert published.validate_values(mutate(df.copy()))


def test_missing_columns_are_reported(df):
    assert published.validate_values(df.drop(columns=["wording"]))[0].startswith("missing columns")


def test_check_published_raises_on_bad_file(tmp_path, df):
    bad = df.copy()
    bad.loc[0, "value"] = 99
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="do not match"):
        published.check_published(p)


def test_table_values_returns_printed_order(df):
    assert list(published.table_values(df, "Table 5")["group_label"]) == [
        "Under 30 years", "30-45 years", "Over 45 years"]
