"""Repository paths. ``GRAMEEN_ROOT`` overrides the root (used by the clean-checkout test)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("GRAMEEN_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def config_dir() -> Path:
    return repo_root() / "configs"


def published_values_csv() -> Path:
    return data_dir() / "published" / "paper_values.csv"


def figure4_labels_csv() -> Path:
    return data_dir() / "published" / "figure4_printed_labels.csv"


def data_dictionary_csv() -> Path:
    return data_dir() / "published" / "data_dictionary.csv"


def synthetic_fixture() -> tuple[Path, Path]:
    base = data_dir() / "synthetic"
    return base / "synthetic_fixture.csv", base / "synthetic_fixture.meta.yaml"


def original_data() -> tuple[Path, Path]:
    base = data_dir() / "original"
    return base / "borrowers.csv", base / "borrowers.meta.yaml"


def qualitative_dir() -> Path:
    return data_dir() / "qualitative"
