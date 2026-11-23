"""Published aggregates transcribed from the paper, and their validation.

The paper prints percentages only. It gives no individual incomes, group sizes, variances,
coefficients, or significance levels, so none are inferred here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths

REQUIRED_COLUMNS = [
    "record_id", "source_type", "source_table", "source_location", "printed_page",
    "group_dimension", "group_label", "secondary_dimension", "secondary_label", "measure",
    "unit", "value", "change_basis", "denominator_n", "denominator_status", "wording", "notes",
]
CHANGE_BASES = {"percent_increase", "percentage_point_change", "level_percent_unspecified", "not_stated"}

# Values read off the printed paper (page 2585 tables; pages 2582-2583 narrative).
EXPECTED_TABLE_VALUES: dict[str, dict[str, float]] = {
    "Table 4": {"Small Business": 32.0, "Agriculture": 22.0, "Overall": 28.0},
    "Table 5": {"Under 30 years": 40.0, "30-45 years": 30.0, "Over 45 years": 20.0},
    "Table 6": {"No formal education": 25.0, "Primary education": 35.0, "Secondary education": 50.0},
}
EXPECTED_TABLE6_LOCATION = {
    "No formal education": "Rural", "Primary education": "Mixed", "Secondary education": "Urban",
}
EXPECTED_NARRATIVE: dict[str, tuple[float, int]] = {
    "N-01": (28.0, 2582), "N-02": (32.0, 2582), "N-03": (22.0, 2582), "N-04": (35.0, 2583),
    "N-05": (40.0, 2583), "N-06": (30.0, 2583), "N-07": (40.0, 2583),
}


def load_values(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or paths.published_values_csv(), keep_default_na=False)


def table_values(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Rows of one printed table, in printed order."""
    return df[df["source_table"] == table].reset_index(drop=True)


def validate_values(df: pd.DataFrame) -> list[str]:
    """Return a list of problems; an empty list means the transcription matches the paper."""
    problems: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df["record_id"].duplicated().any():
        problems.append("duplicate record_id values")
    for table, expected in EXPECTED_TABLE_VALUES.items():
        sub = table_values(df, table)
        got = {r.group_label: float(r.value) for r in sub.itertuples()}
        if got != expected:
            problems.append(f"{table}: expected {expected}, found {got}")
        if (sub["printed_page"].astype(int) != 2585).any():
            problems.append(f"{table}: printed page must be 2585")
    for r in table_values(df, "Table 6").itertuples():
        if EXPECTED_TABLE6_LOCATION.get(r.group_label) != r.secondary_label:
            problems.append(f"Table 6 row '{r.group_label}' lost its location label")
    for rid, (value, page) in EXPECTED_NARRATIVE.items():
        hit = df[df["record_id"] == rid]
        if hit.empty:
            problems.append(f"missing narrative record {rid}")
            continue
        if float(hit.iloc[0]["value"]) != value or int(hit.iloc[0]["printed_page"]) != page:
            problems.append(f"{rid}: expected {value} on page {page}")
    if (df["denominator_n"].astype(str) != "unavailable").any():
        problems.append("denominators must be marked 'unavailable'; the paper reports none")
    if not set(df["change_basis"]).issubset(CHANGE_BASES):
        problems.append(f"unknown change_basis: {set(df['change_basis']) - CHANGE_BASES}")
    if (df["source_location"].astype(str).str.len() == 0).any() or (df["wording"] == "").any():
        problems.append("every row needs a source location and wording")
    return problems


def check_published(path: Path | None = None) -> pd.DataFrame:
    df = load_values(path)
    problems = validate_values(df)
    if problems:
        raise ValueError("published values do not match the paper:\n- " + "\n- ".join(problems))
    return df
