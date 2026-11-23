"""Percent income change with explicit handling of zero and negative baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZERO, NEGATIVE, MISSING = "zero_baseline_income", "negative_baseline_income", "missing_baseline_income"


def baseline_problem(baseline: pd.Series) -> pd.Series:
    """Reason a baseline cannot anchor a percent change, or '' when it can."""
    b = pd.to_numeric(baseline, errors="coerce")
    out = pd.Series("", index=b.index, dtype="object")
    out[b.isna()] = MISSING
    out[b == 0] = ZERO
    out[b < 0] = NEGATIVE
    return out


def percent_change(baseline: pd.Series, followup: pd.Series) -> pd.Series:
    """100 * (follow-up - baseline) / baseline; NaN wherever the baseline is not strictly positive.

    A percent change is undefined at a zero baseline. The row is excluded, never imputed,
    and the exclusion is reported by the model code.
    """
    b = pd.to_numeric(baseline, errors="coerce")
    f = pd.to_numeric(followup, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = 100.0 * (f - b) / b
    return pc.where(b > 0)
