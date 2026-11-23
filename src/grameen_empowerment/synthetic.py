"""Seeded generator for the small fictional fixture.

Nothing here is calibrated to the paper's percentages. Coefficients are arbitrary, chosen only
so that the model code has something to estimate and the edge cases (zero baseline income,
missing values, repeated households, boundary ages) are present. Every row carries
``is_synthetic = True`` and the sidecar metadata says ``dataset_type: synthetic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import __version__

DEFAULT_SEED = 20241225
DEFAULT_N = 240
BOUNDARY_AGES = (29, 30, 45, 46)  # exercise the Table 5 boundaries
ORDINAL_LEVELS = ["low", "medium", "high"]
ENGAGEMENT_COMPONENTS = ["social_group_participation", "local_market_participation", "political_forum_participation"]


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, autonomy_type: str = "continuous") -> pd.DataFrame:
    """Return a fictional borrower table. Same ``(n, seed, autonomy_type)`` gives the same table."""
    if autonomy_type not in {"continuous", "binary", "ordinal"}:
        raise ValueError("autonomy_type must be continuous, binary or ordinal")
    rng = np.random.default_rng(seed)
    ids = [f"SYN{i:05d}" for i in range(1, n + 1)]
    n_households = max(2, int(n * 0.85))
    hh = rng.integers(1, n_households + 1, size=n)
    age = rng.integers(20, 61, size=n)
    age[: len(BOUNDARY_AGES)] = BOUNDARY_AGES[: min(n, len(BOUNDARY_AGES))]
    education = rng.choice(["none", "primary", "secondary", "higher"], size=n, p=[0.35, 0.35, 0.25, 0.05])
    sector = rng.choice(["small_business", "agriculture"], size=n, p=[0.6, 0.4])
    loan = np.round(np.exp(rng.normal(np.log(20000), 0.5, n)), -2)
    tenure = rng.choice([6, 12, 18, 24, 36], size=n).astype(float)
    training = rng.binomial(1, 0.4, n)
    group = rng.binomial(1, 0.7, n)

    edu_rank = pd.Series(education).map({"none": 0, "primary": 1, "secondary": 2, "higher": 3}).to_numpy()
    base = np.round(np.exp(rng.normal(np.log(6000), 0.45, n)), 0)
    base[rng.random(n) < 0.05] = 0.0  # zero-baseline edge case
    growth = 0.04 + 0.000004 * loan + 0.002 * tenure + 0.03 * edu_rank
    follow = base * (1 + growth) + rng.normal(0, 600, n)
    follow = np.where(base == 0, 1500 + rng.normal(0, 400, n) + 0.05 * loan, follow)
    follow = np.round(np.clip(follow, 0, None), 0)

    latent = -0.3 + 0.7 * training + 0.5 * group + rng.normal(0, 1, n)
    if autonomy_type == "continuous":
        autonomy: Any = np.round(np.clip(50 + 12 * latent + rng.normal(0, 4, n), 0, 100), 1)
    elif autonomy_type == "binary":
        autonomy = (latent > 0).astype(float)
    else:
        cuts = np.quantile(latent, [1 / 3, 2 / 3])
        autonomy = np.array(ORDINAL_LEVELS, dtype=object)[np.digitize(latent, cuts)]

    z = rng.normal(0, 1, n)
    attendance = rng.poisson(np.exp(1.0 + 0.35 * z + 0.2 * group)).astype(float)
    leader = rng.binomial(1, _logistic(-1.4 + 0.5 * z))
    comps = {c: np.clip(np.round(2 + 0.9 * z + rng.normal(0, 1.1, n)), 0, 4) for c in ENGAGEMENT_COMPONENTS}
    engagement = np.round(100 * np.mean([v / 4 for v in comps.values()], axis=0), 1)

    w = np.exp(rng.normal(0, 0.35, n))
    df = pd.DataFrame({
        "borrower_id": ids, "household_id": [f"H{h:04d}" for h in hh],
        "psu": [f"V{v:02d}" for v in rng.integers(1, 13, n)],
        "survey_weight": np.round(w / w.mean(), 4), "age": age.astype(float), "sector": sector,
        "income_baseline": base, "income_followup": follow, "loan_size": loan,
        "loan_tenure_months": tenure, "education_level": education,
        "training_participation": training.astype(float), "group_membership": group.astype(float),
        "autonomy": autonomy, "community_meeting_attendance": attendance,
        "leadership_role": leader.astype(float), **comps, "social_engagement": engagement,
    })
    for col, frac in (("loan_tenure_months", 0.03), ("education_level", 0.02), ("autonomy", 0.02),
                      ("community_meeting_attendance", 0.02)):
        df.loc[rng.random(n) < frac, col] = np.nan
    df["is_synthetic"] = True
    return df


def metadata(n: int, seed: int, autonomy_type: str = "continuous") -> dict[str, Any]:
    autonomy: dict[str, Any] = {
        "measure_type": autonomy_type,
        "definition": "FICTIONAL decision-autonomy score generated from training and group membership plus noise.",
    }
    if autonomy_type == "ordinal":
        autonomy["levels"] = ORDINAL_LEVELS
    return {
        "dataset_id": "synthetic_fixture",
        "dataset_type": "synthetic",
        "description": "FICTIONAL borrowers. Not derived from the paper's percentages and not tuned to them.",
        "generator": {"seed": seed, "n": n, "autonomy_type": autonomy_type, "package_version": __version__},
        "income": {"unit_level": "individual", "reference_period": "monthly", "currency": "BDT",
                   "baseline_comparable": True},
        "education": {"coding": ["none", "primary", "secondary", "higher"], "reference": "none"},
        "autonomy": autonomy,
        "social_engagement": {
            "measure_type": "continuous",
            "definition": "FICTIONAL index (0-100): mean intensity of social-group, local-market and "
                          "political-forum participation. Excludes meeting attendance and leadership.",
            "components": ENGAGEMENT_COMPONENTS,
        },
    }


def write_fixture(outdir: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                  autonomy_type: str = "continuous", stem: str = "synthetic_fixture") -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = outdir / f"{stem}.csv", outdir / f"{stem}.meta.yaml"
    df = generate(n, seed, autonomy_type)
    num = df.select_dtypes("float").columns
    df[num] = df[num].round(4) + 0.0  # -0.0 and -1e-17 become 0.0, so bytes do not depend on numpy version
    df.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(metadata(n, seed, autonomy_type), fh, sort_keys=False)
    return csv_path, meta_path
