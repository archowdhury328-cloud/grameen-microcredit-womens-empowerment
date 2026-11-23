from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grameen_empowerment import modelspec, schema, synthetic


@pytest.fixture(scope="session")
def synth_df() -> pd.DataFrame:
    return synthetic.generate()


@pytest.fixture(scope="session")
def synth_meta() -> dict:
    return synthetic.metadata(synthetic.DEFAULT_N, synthetic.DEFAULT_SEED)


@pytest.fixture()
def plain_frame() -> pd.DataFrame:
    """Large, clean frame with known coefficients and no weight or cluster columns."""
    rng = np.random.default_rng(7)
    n = 400
    df = pd.DataFrame({
        "borrower_id": [f"P{i}" for i in range(n)],
        "loan_size": rng.normal(20, 4, n),
        "loan_tenure_months": rng.choice([6, 12, 24], n).astype(float),
        "education_level": rng.choice(["none", "primary", "secondary"], n),
        "training_participation": rng.binomial(1, 0.5, n).astype(float),
        "group_membership": rng.binomial(1, 0.5, n).astype(float),
        "community_meeting_attendance": rng.poisson(3, n).astype(float),
        "leadership_role": rng.binomial(1, 0.3, n).astype(float),
    })
    df["income_followup"] = 100 + 3.0 * df["loan_size"] + 0.5 * df["loan_tenure_months"] + rng.normal(0, 2, n)
    df["autonomy"] = 10 + 4.0 * df["training_participation"] + 2.0 * df["group_membership"] + rng.normal(0, 1, n)
    df["social_engagement"] = 5 + 1.5 * df["community_meeting_attendance"] + rng.normal(0, 1, n)
    return df


@pytest.fixture()
def plain_meta() -> dict:
    return {
        "dataset_id": "plain", "dataset_type": "original_microdata",
        "income": {"unit_level": "individual", "reference_period": "monthly", "currency": "BDT",
                   "baseline_comparable": False},
        "autonomy": {"measure_type": "continuous", "definition": "test score"},
        "social_engagement": {"measure_type": "continuous", "definition": "test index", "components": ["a", "b"]},
    }


@pytest.fixture()
def configs():
    return modelspec.load_all_configs()


@pytest.fixture()
def report_ok(synth_df, synth_meta):
    return schema.validate_microdata(synth_df, synth_meta)
