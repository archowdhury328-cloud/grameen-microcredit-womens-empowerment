"""The Figure 4 discrepancy.

Figure 4 prints pie labels of 22.7%, 31.8% and 45.5%. They are Table 6's 25, 35 and 50 divided
by their sum, 110. A pie forces its slices to total 100%, so the picture re-expresses three
independent index values as shares of their own sum. It does not describe how borrowers are
distributed across education levels, which the paper never reports.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from . import paths


def normalized_shares(values: Mapping[str, float]) -> dict[str, float]:
    """Each value divided by the sum of all values, in percent (unrounded)."""
    total = float(sum(values.values()))
    if total <= 0:
        raise ValueError("cannot normalise: values must have a positive sum")
    return {k: 100.0 * float(v) / total for k, v in values.items()}


def printed_label(value: float) -> float:
    """Round to one decimal, as the pie labels were printed."""
    return round(float(value), 1)


def compare_with_printed(labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Table 6 values, their normalised shares, and the labels printed in Figure 4."""
    df = labels if labels is not None else pd.read_csv(paths.figure4_labels_csv())
    shares = normalized_shares(dict(zip(df["education_level"], df["table6_value_percent"], strict=True)))
    out = df.copy()
    out["sum_of_table6_values"] = float(df["table6_value_percent"].sum())
    out["recomputed_share_percent"] = [shares[k] for k in out["education_level"]]
    out["recomputed_label"] = out["recomputed_share_percent"].map(printed_label)
    out["matches_printed_label"] = out["recomputed_label"] == out["printed_pie_label_percent"]
    return out


CAPTION = (
    "Figure 4 as printed re-expresses Table 6's three index values (25, 35, 50) as shares of their "
    "sum, 110 (25/110 = 22.7%, 35/110 = 31.8%, 50/110 = 45.5%). The slices are not the distribution "
    "of borrowers across education levels; the paper reports no such distribution."
)
