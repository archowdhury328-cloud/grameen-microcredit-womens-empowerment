"""Importer for the World Bank / BIDS "Long-Term Impact of Microcredit Impacts 1991-1992" survey.

Catalog 1317 (ref BGD_1991_LTIMCI_v01_M, https://microdata.worldbank.org/catalog/1317). It covers
Grameen Bank, BRAC and the BRDB RD-12 programme plus non-programme villages, so a result is only
Grameen-specific after filtering to a documented programme code. This is a different survey from
the paper's secondary sources: nothing imported here reproduces or tests the paper.

The microdata are licensed (no redistribution) and are never committed to Git. This module reads
files the user downloaded and cannot run without them. Variable names below were read from the
catalog's public data dictionary; *coding of values must be checked against the downloaded
dictionary* before any mapping is trusted, which is why the programme code is left unset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CATALOG_URL = "https://microdata.worldbank.org/catalog/1317"
DOI = "https://doi.org/10.48529/anw3-f530"
CITATION = ("World Bank, Bangladesh Institute of Development Studies. Long-Term Impact of Microcredit Impacts "
            "1991-1992, Ref. BGD_1991_LTIMCI_v01_M. Dataset downloaded from {url} on {date}.")
KEY = ["nh", "pid"]

# target field -> (file, variable(s), status, note). Names verified against the public dictionary.
FIELD_MAP: list[dict[str, str]] = [
    {"target": "household_id", "file": "any", "variable": "nh", "status": "supported", "note": "Household Number"},
    {"target": "survey_weight", "file": "03br1", "variable": "weight", "status": "supported",
     "note": "Household sampling weight (one row per household)."},
    {"target": "education_level", "file": "02r1", "variable": "edslg", "status": "supported_verify_coding",
     "note": "ED: Highest grade completed. Recode to none/primary/secondary/higher only after reading the value labels."},
    {"target": "training_participation", "file": "02r1", "variable": "edvrt", "status": "supported_verify_coding",
     "note": "EDV: Ever get tech training. Not the same construct as the paper's 'participation in training'."},
    {"target": "group_membership", "file": "11cr1", "variable": "csgpc;csgpn", "status": "supported_verify_coding",
     "note": "Currently member of programme (csgpc) and which programme (csgpn). Programme code must be supplied."},
    {"target": "loan_size", "file": "11ar1", "variable": "csbprin", "status": "supported_verify_coding",
     "note": "Loan principal; summed per borrower (loan-level file). Currency and units per dictionary."},
    {"target": "loan_tenure_months", "file": "11ar1", "variable": "csbtpbd/csbtpbm/csbtpby", "status": "unsupported",
     "note": "Only a due date is recorded; tenure needs loan-date arithmetic with documented conventions. Not implemented."},
    {"target": "income_followup", "file": "04ar1/04br1/05*/06*", "variable": "several", "status": "unsupported",
     "note": "No single income variable. Wage, self-employment and farm income sit in different modules and need a documented construction."},
    {"target": "autonomy", "file": "none found", "variable": "none", "status": "unsupported",
     "note": "No household decision-making module appears in the 12 listed questionnaire sections."},
    {"target": "social_engagement", "file": "none found", "variable": "none", "status": "unsupported",
     "note": "No community-participation module appears in the 12 listed questionnaire sections."},
]
SUPPORTED_FILES = {"rosterr1": "roster", "02r1": "education", "03br1": "weights", "11cr1": "groups", "11ar1": "loans"}


class ConfigIncomplete(ValueError):
    """The mapping file leaves a required, data-dependent value unset."""


@dataclass(frozen=True)
class BidsMapping:
    grameen_progid: Any
    member_codes: tuple[Any, ...]
    yes_codes: tuple[Any, ...]


def load_mapping(path: Path) -> BidsMapping:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    missing = [k for k in ("grameen_progid", "member_codes", "yes_codes") if not raw.get(k) and raw.get(k) != 0]
    if missing:
        raise ConfigIncomplete(
            f"mapping file {path} leaves {missing} unset. Read the value labels in the downloaded data "
            "dictionary (or the catalog's data dictionary pages) and fill them in; until then no estimate may "
            "be labelled Grameen-specific.")
    return BidsMapping(raw["grameen_progid"], tuple(_as_list(raw["member_codes"])), tuple(_as_list(raw["yes_codes"])))


def _as_list(x: Any) -> list[Any]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _read(directory: Path, name: str) -> pd.DataFrame:
    for ext, reader in ((".csv", pd.read_csv), (".dta", pd.read_stata)):
        p = directory / f"{name}{ext}"
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"{name}.csv or {name}.dta not found in {directory}; download it from {CATALOG_URL}")


def import_bids(directory: Path, mapping: BidsMapping) -> pd.DataFrame:
    """Person-level frame for round 1 with supported fields only, filtered to the Grameen programme."""
    edu = _read(directory, "02r1")[["nh", "pid", "edslg", "edvrt"]]
    groups = _read(directory, "11cr1")
    for col in ("nh", "csgpid", "csgpc", "csgpn"):
        if col not in groups.columns:
            raise ValueError(f"11cr1 lacks expected variable '{col}'; check it against the data dictionary")
    groups = groups.rename(columns={"csgpid": "pid"})[["nh", "pid", "csgpc", "csgpn"]]
    weights = _read(directory, "03br1")[["nh", "weight"]]
    loans = _read(directory, "11ar1")[["nh", "csbpbid", "csbprin"]].rename(columns={"csbpbid": "pid"})
    loan_total = loans.groupby(KEY, as_index=False)["csbprin"].sum(min_count=1)
    df = (edu.merge(groups, on=KEY, how="left").merge(weights, on="nh", how="left")
          .merge(loan_total, on=KEY, how="left"))
    df["grameen_member"] = df["csgpn"].isin(mapping.member_codes) & df["csgpc"].isin(mapping.yes_codes)
    out = pd.DataFrame({
        "borrower_id": df["nh"].astype(str) + "-" + df["pid"].astype(str), "household_id": df["nh"].astype(str),
        "survey_weight": df["weight"], "education_level": df["edslg"], "training_participation": df["edvrt"],
        "group_membership": df["grameen_member"].astype(float), "loan_size": df["csbprin"],
    })
    return out.reset_index(drop=True)


def attrition_report(wave_a_ids: pd.Series, wave_b_ids: pd.Series, label_a: str = "1991-92", label_b: str = "1998-99") -> pd.DataFrame:
    """Households present in wave A and still present in wave B. A longitudinal calculation needs this."""
    a, b = set(map(str, wave_a_ids.dropna())), set(map(str, wave_b_ids.dropna()))
    kept = a & b
    return pd.DataFrame([{
        "wave_a": label_a, "wave_b": label_b, "households_wave_a": len(a), "households_wave_b": len(b),
        "retained": len(kept), "lost": len(a - b), "new_in_wave_b": len(b - a),
        "retention_rate": len(kept) / len(a) if a else float("nan"),
    }])


def longitudinal_blockers(waves: dict[str, dict[str, Any]]) -> list[str]:
    """Reasons a two-wave calculation cannot proceed yet (empty list = nothing documented as missing)."""
    problems: list[str] = []
    if len(waves) < 2:
        return ["two linked waves are required; the 1998-99 revisit is a separate catalog entry"]
    for label, w in waves.items():
        if not w.get("ids_documented_as_linkable"):
            problems.append(f"wave {label}: household ids are not documented as linkable across waves")
        if not w.get("measure_definition"):
            problems.append(f"wave {label}: measure definition (unit, period, currency) is undocumented")
    defs = {str(w.get("measure_definition")) for w in waves.values()}
    if len(defs) > 1:
        problems.append("measure definitions differ between waves; the measures are not compatible")
    return problems


def field_map_frame() -> pd.DataFrame:
    return pd.DataFrame(FIELD_MAP)
