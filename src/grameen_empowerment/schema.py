"""Row-level data schema and validator.

A dataset is a CSV with one row per borrower plus a ``.meta.yaml`` sidecar that documents how
each measure was collected. The regression command refuses to run unless the required fields
for an outcome are present *and* documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import DATASET_TYPES

ROW_LEVEL_DATASET_TYPES = ("synthetic", "original_microdata", "independent_bids")
EDUCATION_LEVELS = ["none", "primary", "secondary", "higher"]
AGGREGATE_MARKERS = {"source_table", "group_label", "printed_page", "denominator_n"}

# Fields common to every outcome. Optional design fields are used when present.
ID_FIELD = "borrower_id"
DESIGN_FIELDS = {"household_id": "repeated households (cluster)", "psu": "primary sampling unit (cluster)",
                 "survey_weight": "survey weight", "strata": "stratum (recorded; not used in variance)"}

# outcome key -> (outcome field, predictor fields, metadata section)
OUTCOME_FIELDS: dict[str, dict[str, Any]] = {
    "income_level": {"outcome": "income_followup",
                     "predictors": ["loan_size", "loan_tenure_months", "education_level"],
                     "meta": "income"},
    "income_growth": {"outcome": "income_followup",
                      "predictors": ["loan_size", "loan_tenure_months", "education_level"],
                      "meta": "income", "also": ["income_baseline"]},
    "decision_autonomy": {"outcome": "autonomy",
                          "predictors": ["training_participation", "group_membership"],
                          "meta": "autonomy"},
    "social_engagement": {"outcome": "social_engagement",
                          "predictors": ["community_meeting_attendance", "leadership_role"],
                          "meta": "social_engagement"},
}
MEASURE_TYPES = {"continuous", "binary", "ordinal"}

FIELD_DICTIONARY: dict[str, str] = {
    "borrower_id": "string; unique pseudonymous row id (no names, no national ids)",
    "household_id": "string; households with several borrowers share one id (cluster)",
    "psu": "string; primary sampling unit, e.g. village (cluster)",
    "strata": "string; sampling stratum (recorded only)",
    "survey_weight": "float > 0; documented survey weight",
    "age": "float years; used only for descriptive age groups",
    "sector": "small_business | agriculture; used only for descriptive comparison",
    "income_baseline": "float >= 0; income before the loan, same unit/period/currency as follow-up",
    "income_followup": "float >= 0; measured income after the loan",
    "loan_size": "float > 0; principal in the currency declared in the metadata",
    "loan_tenure_months": "float > 0; loan tenure in months",
    "education_level": "none | primary | secondary | higher (ordered categories)",
    "training_participation": "0/1; took part in training",
    "group_membership": "0/1; member of a lending group",
    "autonomy": "decision-making autonomy measure; type declared in metadata (continuous|binary|ordinal)",
    "community_meeting_attendance": "float >= 0; meetings attended in the reference period",
    "leadership_role": "0/1; holds a leadership role",
    "social_engagement": "independently measured engagement index; components declared in metadata",
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    supported_outcomes: list[str] = field(default_factory=list)
    blocked_outcomes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"errors: {len(self.errors)}  warnings: {len(self.warnings)}"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  WARNING {w}" for w in self.warnings]
        lines.append(f"  supported outcomes: {self.supported_outcomes or 'none'}")
        for k, v in self.blocked_outcomes.items():
            lines.append(f"  blocked {k}: " + "; ".join(v))
        return "\n".join(lines)


def load_meta(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: metadata must be a YAML mapping")
    return meta


def looks_like_aggregate(df: pd.DataFrame) -> bool:
    """True for tables of published percentages (or anything without per-borrower rows)."""
    return bool(AGGREGATE_MARKERS & set(df.columns)) or ID_FIELD not in df.columns


def _meta_gaps(outcome: str, meta: dict[str, Any]) -> list[str]:
    """Documentation that must exist before an outcome may be fitted."""
    gaps: list[str] = []
    spec = OUTCOME_FIELDS[outcome]
    section = meta.get(spec["meta"])
    if not isinstance(section, dict):
        return [f"metadata section '{spec['meta']}' is missing"]
    if spec["meta"] == "income":
        for key in ("unit_level", "reference_period", "currency"):
            if not section.get(key):
                gaps.append(f"income.{key} must be documented")
        if section.get("unit_level") not in (None, "individual", "household"):
            gaps.append("income.unit_level must be 'individual' or 'household'")
        if outcome == "income_growth" and section.get("baseline_comparable") is not True:
            gaps.append("income.baseline_comparable must be true: baseline and follow-up need the "
                        "same unit level, reference period and currency")
    else:
        if section.get("measure_type") not in MEASURE_TYPES:
            gaps.append(f"{spec['meta']}.measure_type must be one of {sorted(MEASURE_TYPES)}")
        if not section.get("definition"):
            gaps.append(f"{spec['meta']}.definition must document how the outcome was measured")
        if section.get("measure_type") == "ordinal" and not section.get("levels"):
            gaps.append(f"{spec['meta']}.levels must list the ordered categories")
    return gaps


def validate_microdata(df: pd.DataFrame, meta: dict[str, Any]) -> ValidationReport:
    """Check structure, documentation and per-outcome support."""
    rep = ValidationReport()
    dtype = meta.get("dataset_type")
    if dtype not in DATASET_TYPES:
        rep.errors.append(f"metadata dataset_type must be one of {DATASET_TYPES}, found {dtype!r}")
    elif dtype not in ROW_LEVEL_DATASET_TYPES:
        rep.errors.append(f"dataset_type {dtype!r} is an aggregate type and cannot be fitted")
    if looks_like_aggregate(df):
        rep.errors.append("this table looks like published aggregates, not borrower-level rows "
                          "(no borrower_id, or aggregate columns present); refusing to fit")
        return rep
    if df[ID_FIELD].isna().any() or df[ID_FIELD].astype(str).str.strip().eq("").any():
        rep.errors.append("borrower_id has missing values")
    if df[ID_FIELD].duplicated().any():
        rep.errors.append("borrower_id must be unique per row")
    if dtype == "synthetic" and "is_synthetic" not in df.columns:
        rep.errors.append("synthetic data must carry an is_synthetic column")
    if "survey_weight" in df and (pd.to_numeric(df["survey_weight"], errors="coerce") <= 0).any():
        rep.warnings.append("survey_weight has non-positive values; those rows will be excluded")
    for outcome, spec in OUTCOME_FIELDS.items():
        needed = [spec["outcome"], *spec["predictors"], *spec.get("also", [])]
        absent = [c for c in needed if c not in df.columns]
        reasons = [f"missing column '{c}'" for c in absent] + _meta_gaps(outcome, meta)
        if not reasons and outcome == "social_engagement":
            reasons += _circularity_gaps(meta)
        if reasons:
            rep.blocked_outcomes[outcome] = reasons
        else:
            rep.supported_outcomes.append(outcome)
    if not rep.supported_outcomes and rep.ok:
        rep.errors.append("no outcome has all required fields documented; see docs/ACCESS_ORIGINAL_DATA.md "
                          "and data/schema.md for the required columns")
    return rep


def _circularity_gaps(meta: dict[str, Any]) -> list[str]:
    from .circularity import circularity_problems

    return circularity_problems(meta)


def schema_markdown() -> str:
    """Render the schema as the Markdown that ships in docs/."""
    lines = ["| field | description |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in FIELD_DICTIONARY.items()]
    return "\n".join(lines)
