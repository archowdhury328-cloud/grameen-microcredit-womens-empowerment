"""Qualitative coding and the joint display (Section 3.4 and 3.5 of the paper).

Stages: (1) familiarisation, (2) coding, (3) theme identification, (4) interpretation.

This module never invents interviews, respondents, independent coders, agreement scores or
theme prevalence. The shipped annotations are paraphrases of passages the *paper* reports;
the counts it prints are counts of annotated passages, not frequencies in any population.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths

CODEBOOK_COLUMNS = ["code_id", "theme", "definition", "inclusion_rule", "exclusion_rule",
                    "linked_outcome", "relation_to_quantitative"]
ANNOTATION_COLUMNS = ["annotation_id", "source_id", "locator", "evidence_type", "code_id", "theme",
                      "subgroup_key", "respondent_attributes", "paraphrase", "analyst_note", "annotator"]
REQUIRED_NONEMPTY = ["annotation_id", "source_id", "locator", "theme", "analyst_note"]
EVIDENCE_TYPES = {"reported_quote_in_paper", "author_summary_in_paper", "new_primary_data"}
RELATIONS = {"convergent", "divergent", "mixed"}
FORBIDDEN_FRAGMENTS = ("kappa", "agreement", "prevalence", "frequency", "inter_coder", "second_coder", "n_respondents")
INSUFFICIENT = "insufficient evidence"
CAVEAT = ("Single annotator working from passages reported in the paper; counts are annotated passages, "
          "not prevalence; no coder agreement exists.")

# (finding id, outcome, dimension, label, published record ids). 'all' findings use every annotation
# linked to the outcome; subgroup findings need an annotation carrying the matching subgroup_key.
FINDINGS: list[tuple[str, str, str, str, list[str]]] = [
    ("J1", "income_level", "all", "Overall", ["N-01"]),
    ("J2", "income_level", "sector", "Small Business", ["T4-01", "N-02"]),
    ("J3", "income_level", "sector", "Agriculture", ["T4-02", "N-03"]),
    ("J4", "decision_autonomy", "all", "Overall", ["N-04"]),
    ("J5", "decision_autonomy", "age_group", "Under 30 years", ["T5-01", "N-05"]),
    ("J6", "decision_autonomy", "age_group", "30-45 years", ["T5-02", "N-06"]),
    ("J7", "decision_autonomy", "age_group", "Over 45 years", ["T5-03"]),
    ("J8", "social_engagement", "all", "Overall", ["N-07"]),
    ("J9", "social_engagement", "education_level", "No formal education", ["T6-01"]),
    ("J10", "social_engagement", "education_level", "Primary education", ["T6-02"]),
    ("J11", "social_engagement", "education_level", "Secondary education", ["T6-03"]),
]


def load_codebook(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or paths.qualitative_dir() / "codebook.csv", keep_default_na=False)


def load_annotations(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or paths.qualitative_dir() / "annotations_paper_examples.csv", keep_default_na=False)


def validate_codebook(cb: pd.DataFrame) -> list[str]:
    problems = [f"codebook missing column '{c}'" for c in CODEBOOK_COLUMNS if c not in cb.columns]
    if problems:
        return problems
    if cb["code_id"].duplicated().any() or cb["theme"].duplicated().any():
        problems.append("codebook code_id and theme must be unique")
    if not set(cb["relation_to_quantitative"]).issubset(RELATIONS):
        problems.append(f"relation_to_quantitative must be in {sorted(RELATIONS)}")
    return problems


def validate_annotations(ann: pd.DataFrame, cb: pd.DataFrame) -> list[str]:
    """Import validator for annotation CSVs."""
    problems = [f"missing column '{c}'" for c in ANNOTATION_COLUMNS if c not in ann.columns]
    bad_cols = [c for c in ann.columns if any(f in c.lower() for f in FORBIDDEN_FRAGMENTS)]
    if bad_cols:
        problems.append(f"columns {bad_cols} are not allowed: agreement scores, prevalence and "
                        "respondent counts must not be recorded unless real data supports them")
    if problems:
        return problems
    for col in REQUIRED_NONEMPTY:
        empty = ann[ann[col].astype(str).str.strip() == ""]
        problems += [f"row {i + 2}: '{col}' is empty" for i in empty.index]
    if ann["annotation_id"].duplicated().any():
        problems.append("annotation_id must be unique")
    known = dict(zip(cb["code_id"], cb["theme"], strict=True))
    for i, r in ann.iterrows():
        if r["code_id"] not in known:
            problems.append(f"row {i + 2}: unknown code_id '{r['code_id']}'")
        elif known[r["code_id"]] != r["theme"]:
            problems.append(f"row {i + 2}: theme '{r['theme']}' does not match code '{r['code_id']}'")
        if r["evidence_type"] not in EVIDENCE_TYPES:
            problems.append(f"row {i + 2}: evidence_type must be one of {sorted(EVIDENCE_TYPES)}")
    return problems


def stage1_familiarization(ann: pd.DataFrame) -> pd.DataFrame:
    """Passages read, by source and evidence type."""
    return (ann.groupby(["source_id", "locator", "evidence_type"], as_index=False)
            .agg(annotated_passages=("annotation_id", "count")))


def stage2_coding(ann: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    return ann.merge(cb[["code_id", "linked_outcome", "relation_to_quantitative"]], on="code_id", how="left")[
        ["annotation_id", "source_id", "code_id", "theme", "linked_outcome", "relation_to_quantitative", "evidence_type"]]


def stage3_themes(ann: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in cb.itertuples():
        hits = ann[ann["code_id"] == r.code_id]
        rows.append({"theme": r.theme, "code_id": r.code_id, "linked_outcome": r.linked_outcome,
                     "relation_to_quantitative": r.relation_to_quantitative,
                     "annotated_passages": len(hits), "annotation_ids": ";".join(hits["annotation_id"]),
                     "status": "annotated" if len(hits) else INSUFFICIENT})
    return pd.DataFrame(rows)


def _status(relations: set[str]) -> str:
    if not relations:
        return INSUFFICIENT
    if relations == {"convergent"}:
        return "convergent"
    if relations == {"divergent"}:
        return "divergent"
    return "tension"


def stage4_interpretation(published: pd.DataFrame, ann: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """Joint display linking each quantitative finding to the qualitative evidence, with tensions kept."""
    rel = dict(zip(cb["theme"], cb["relation_to_quantitative"], strict=True))
    outcome_of = dict(zip(cb["theme"], cb["linked_outcome"], strict=True))
    rows = []
    for fid, outcome, dim, label, rec_ids in FINDINGS:
        recs = published[published["record_id"].isin(rec_ids)]
        if recs.empty:
            raise ValueError(f"joint display references unknown published records {rec_ids}")
        linked = ann[ann["theme"].map(outcome_of) == outcome]
        if dim != "all":
            linked = linked[linked["subgroup_key"] == f"{dim}:{label}"]
        relations = {rel[t] for t in linked["theme"]}
        conv = sorted({t for t in linked["theme"] if rel[t] == "convergent"})
        div = sorted({t for t in linked["theme"] if rel[t] == "divergent"})
        mix = sorted({t for t in linked["theme"] if rel[t] == "mixed"})
        first = recs.iloc[0]
        status = _status(relations)
        summary = INSUFFICIENT if status == INSUFFICIENT else "; ".join(
            x for x in (f"convergent: {', '.join(conv)}" if conv else "", f"divergent: {', '.join(div)}" if div else "",
                        f"mixed: {', '.join(mix)}" if mix else "") if x)
        rows.append({
            "finding_id": fid, "outcome": outcome, "subgroup": "all women" if dim == "all" else f"{dim}: {label}",
            "quantitative_finding": f"{first['measure']} = {first['value']:g}% ({first['change_basis']})",
            "source_records": ";".join(recs["record_id"]),
            "source_pages": ";".join(sorted({str(p) for p in recs["printed_page"]})),
            "quantitative_dataset_type": "published_aggregate",
            "qualitative_status": status, "evidence_pattern": summary,
            "supporting_annotation_ids": ";".join(linked["annotation_id"]),
            "caveat": CAVEAT if status != INSUFFICIENT else "No annotated passage matches this finding.",
        })
    return pd.DataFrame(rows)


def run_stages(published: pd.DataFrame, ann: pd.DataFrame, cb: pd.DataFrame) -> dict[str, pd.DataFrame]:
    problems = validate_codebook(cb) + validate_annotations(ann, cb)
    if problems:
        raise ValueError("qualitative inputs failed validation:\n- " + "\n- ".join(problems))
    return {"stage1_familiarization": stage1_familiarization(ann), "stage2_coding": stage2_coding(ann, cb),
            "stage3_themes": stage3_themes(ann, cb), "stage4_joint_display": stage4_interpretation(published, ann, cb)}
