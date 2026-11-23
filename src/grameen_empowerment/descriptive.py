"""Descriptive group comparisons computed from row-level data.

The 'overall' figure is computed from all rows. It is never derived from the group means,
because that would silently assume equal group sizes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import agegroups
from .incomegrowth import percent_change


def _summ(frame: pd.DataFrame, value: str, weight: str | None) -> dict[str, float]:
    v = frame[value].astype(float)
    row: dict[str, float] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan,
                             "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan}
    if weight:
        w = frame[weight].astype(float)
        row["weighted_mean"] = float(np.average(v, weights=w)) if len(v) and w.sum() > 0 else np.nan
    return row


def group_summary(df: pd.DataFrame, value: str, group: str, weight: str | None = None) -> pd.DataFrame:
    """Per-group n/mean/sd plus an 'Overall' row from all complete rows."""
    cols = [value, group] + ([weight] if weight else [])
    work = df[cols].dropna(subset=[value, group]).copy()
    rows = []
    for key, sub in work.groupby(group, sort=True):
        rows.append({"group": key, **_summ(sub, value, weight)})
    rows.append({"group": "Overall", **_summ(work, value, weight)})
    return pd.DataFrame(rows)


def descriptive_tables(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Three comparisons mirroring Tables 4-6, computed from borrower rows."""
    wcol = "survey_weight" if "survey_weight" in df.columns else None
    out: dict[str, pd.DataFrame] = {}
    work = df.copy()
    if {"income_baseline", "income_followup", "sector"} <= set(work.columns):
        work["income_growth_pct"] = percent_change(work["income_baseline"], work["income_followup"])
        keep = [c for c in ("income_growth_pct", "sector", wcol) if c]
        out["income_growth_pct_by_sector"] = group_summary(work[keep], "income_growth_pct", "sector", wcol)
    amt = (meta.get("autonomy") or {}).get("measure_type")
    if "age" in work.columns and "autonomy" in work.columns and amt in ("continuous", "binary"):
        work["age_group"] = agegroups.assign_age_group(work["age"])
        work["autonomy"] = pd.to_numeric(work["autonomy"], errors="coerce")
        keep = [c for c in ("autonomy", "age_group", wcol) if c]
        out["autonomy_level_by_age_group"] = group_summary(work[keep], "autonomy", "age_group", wcol)
    if {"education_level", "social_engagement"} <= set(work.columns):
        keep = [c for c in ("social_engagement", "education_level", wcol) if c]
        out["engagement_by_education"] = group_summary(work[keep], "social_engagement", "education_level", wcol)
    return out
