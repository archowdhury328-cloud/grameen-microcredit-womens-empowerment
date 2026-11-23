"""Model configurations: one YAML per Table 3 outcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths


class ModelSpecError(ValueError):
    """The data cannot support the requested model. The message says what to fix."""


@dataclass(frozen=True)
class Categorical:
    reference: str
    levels: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    outcome_key: str
    paper_reference: str
    outcome: str
    predictors: tuple[str, ...]
    comparison_predictor: str
    meta_section: str
    categorical: dict[str, Categorical] = field(default_factory=dict)
    derived_outcome: dict[str, str] | None = None

    @property
    def single_predictor_id(self) -> str:
        return f"M1_single_{self.comparison_predictor}"


def parse_config(raw: dict[str, Any]) -> ModelConfig:
    required = ("outcome_key", "paper_reference", "outcome", "predictors", "comparison_predictor", "meta_section")
    absent = [k for k in required if k not in raw]
    if absent:
        raise ModelSpecError(f"model config is missing keys: {absent}")
    predictors = tuple(raw["predictors"])
    if raw["comparison_predictor"] not in predictors:
        raise ModelSpecError("comparison_predictor must be one of the predictors")
    cats = {}
    for name, spec in (raw.get("categorical") or {}).items():
        if name not in predictors:
            raise ModelSpecError(f"categorical variable '{name}' is not a predictor")
        levels = tuple(spec["levels"])
        if spec["reference"] not in levels:
            raise ModelSpecError(f"reference level '{spec['reference']}' of '{name}' is not in its levels")
        cats[name] = Categorical(spec["reference"], levels)
    return ModelConfig(
        outcome_key=raw["outcome_key"], paper_reference=raw["paper_reference"], outcome=raw["outcome"],
        predictors=predictors, comparison_predictor=raw["comparison_predictor"],
        meta_section=raw["meta_section"], categorical=cats, derived_outcome=raw.get("derived_outcome"),
    )


def load_config(path: Path) -> ModelConfig:
    with open(path, encoding="utf-8") as fh:
        return parse_config(yaml.safe_load(fh))


def load_all_configs(directory: Path | None = None) -> dict[str, ModelConfig]:
    base = directory or (paths.config_dir() / "models")
    configs = (load_config(p) for p in sorted(base.glob("*.yaml")))
    return {c.outcome_key: c for c in configs}
