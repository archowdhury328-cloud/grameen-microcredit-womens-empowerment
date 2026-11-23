"""Age groups using the boundaries printed in Table 5: under 30, 30-45, over 45."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("Under 30 years", "30-45 years", "Over 45 years")
MIN_AGE, MAX_AGE = 0, 120


def assign_age_group(age: pd.Series | float | int) -> pd.Series | str | float:
    """30 and 45 both belong to '30-45 years'; below 30 is 'Under 30'; above 45 is 'Over 45'.

    Missing or implausible ages (outside 0-120) return NaN rather than being guessed.
    """
    scalar = np.isscalar(age)
    s = pd.Series([age] if scalar else age, dtype="float64")
    valid = s.between(MIN_AGE, MAX_AGE)
    out = pd.Series(np.select([s < 30, s <= 45], [LABELS[0], LABELS[1]], default=LABELS[2]),
                    index=s.index, dtype="object")
    out[~valid | s.isna()] = np.nan
    return out.iloc[0] if scalar else out
