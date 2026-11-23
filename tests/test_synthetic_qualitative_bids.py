from __future__ import annotations

import pandas as pd
import pytest

from grameen_empowerment import agegroups, bids, paths, published, qualitative, synthetic, workflow


# ---------------------------------------------------------------- synthetic data
def test_generator_is_deterministic():
    a, b = synthetic.generate(), synthetic.generate()
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(synthetic.generate(seed=1))


def test_committed_fixture_is_exactly_what_the_generator_writes(tmp_path):
    csv, meta = paths.synthetic_fixture()
    new_csv, new_meta = synthetic.write_fixture(tmp_path)
    assert csv.read_bytes() == new_csv.read_bytes()
    assert meta.read_bytes() == new_meta.read_bytes()


def test_every_row_and_file_is_marked_synthetic(tmp_path):
    df = synthetic.generate()
    assert df["is_synthetic"].all()
    assert df["borrower_id"].str.startswith("SYN").all()
    _, meta_path = synthetic.write_fixture(tmp_path)
    assert "dataset_type: synthetic" in meta_path.read_text(encoding="utf-8")
    assert "FICTIONAL" in meta_path.read_text(encoding="utf-8")


def test_fixture_contains_the_edge_cases_it_promises():
    df = synthetic.generate()
    assert (df["income_baseline"] == 0).any()
    assert df.isna().any().any()
    assert df["household_id"].duplicated().any()
    groups = agegroups.assign_age_group(df["age"].iloc[:4])
    assert list(groups) == ["Under 30 years", "30-45 years", "30-45 years", "Over 45 years"]


@pytest.mark.parametrize("kind", ["continuous", "binary", "ordinal"])
def test_fixture_covers_all_autonomy_measure_types(kind):
    df = synthetic.generate(autonomy_type=kind)
    assert synthetic.metadata(len(df), 1, kind)["autonomy"]["measure_type"] == kind
    assert df["autonomy"].notna().sum() > 200


def test_bad_autonomy_type_is_rejected():
    with pytest.raises(ValueError):
        synthetic.generate(autonomy_type="nonsense")


def test_synthetic_engagement_excludes_both_predictors():
    comps = synthetic.metadata(10, 1)["social_engagement"]["components"]
    assert not {"community_meeting_attendance", "leadership_role"} & set(comps)


def test_synthetic_is_not_tuned_to_the_papers_percentages(synth_df):
    # The paper reports small business 32 / agriculture 22. The fixture must not reproduce them by design.
    g = synth_df.assign(pc=100 * (synth_df["income_followup"] - synth_df["income_baseline"]) / synth_df["income_baseline"].where(lambda s: s > 0))
    means = g.groupby("sector")["pc"].mean()
    assert not (abs(means["small_business"] - 32) < 1 and abs(means["agriculture"] - 22) < 1)


def test_run_synthetic_output_is_labelled_and_stays_in_its_directory(tmp_path):
    summary, written = workflow.run_synthetic(tmp_path / "syn")
    assert all(str(tmp_path / "syn") in str(p) for p in written)
    assert set(summary.coefficients["dataset_type"]) == {"synthetic"}
    report = (tmp_path / "syn" / "synthetic_regression_report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in report.splitlines()[2] or "SYNTHETIC" in report[:400]
    assert "associations only" in report
    assert (tmp_path / "syn" / "synthetic_coefficients.png").exists()
    assert "SYNTHETIC" in (tmp_path / "syn" / "synthetic_coefficients.svg").read_text(encoding="utf-8")


def test_published_and_synthetic_outputs_never_share_a_dataset_label(tmp_path):
    workflow.reproduce_paper(tmp_path / "pub")
    workflow.run_synthetic(tmp_path / "syn")
    pub = pd.read_csv(tmp_path / "pub" / "manifest.csv")
    syn = pd.read_csv(tmp_path / "syn" / "coefficients.csv")
    assert "synthetic" not in set(pub["dataset_type"])
    assert set(syn["dataset_type"]) == {"synthetic"}


# ---------------------------------------------------------------- qualitative
@pytest.fixture()
def cb():
    return qualitative.load_codebook()


@pytest.fixture()
def ann():
    return qualitative.load_annotations()


def test_shipped_files_validate(cb, ann):
    assert qualitative.validate_codebook(cb) == []
    assert qualitative.validate_annotations(ann, cb) == []


def test_codebook_covers_the_six_required_themes(cb):
    assert set(cb["theme"]) == {
        "financial_independence", "household_decision_authority", "community_participation",
        "repayment_stress", "reliance_on_repeated_borrowing", "continued_family_control"}


def test_template_has_the_required_headers_and_no_rows():
    t = pd.read_csv(paths.qualitative_dir() / "annotation_template.csv")
    assert list(t.columns) == qualitative.ANNOTATION_COLUMNS
    assert t.empty


def test_two_worked_examples_come_from_page_2583_and_are_labelled(ann):
    ex = ann[ann["annotation_id"].isin(["A-01", "A-02", "A-03"])]
    assert ex["locator"].str.contains("p. 2583").all()
    assert set(ex["evidence_type"]) == {"reported_quote_in_paper"}
    assert set(ann["annotator"]) == {"release-annotator"}  # no invented coders


@pytest.mark.parametrize("column", ["cohen_kappa", "inter_coder_agreement", "theme_prevalence", "n_respondents"])
def test_invented_agreement_or_prevalence_columns_are_rejected(ann, cb, column):
    assert any("not allowed" in p for p in qualitative.validate_annotations(ann.assign(**{column: 0.8}), cb))


def test_validator_flags_missing_locator_unknown_code_and_mismatched_theme(ann, cb):
    bad = ann.copy()
    bad.loc[0, "locator"] = ""
    bad.loc[1, "code_id"] = "NOPE"
    bad.loc[2, "theme"] = "community_participation"
    bad.loc[3, "evidence_type"] = "invented"
    problems = " | ".join(qualitative.validate_annotations(bad, cb))
    assert "'locator' is empty" in problems
    assert "unknown code_id" in problems
    assert "does not match code" in problems
    assert "evidence_type" in problems


def test_four_stages_run_and_stage_three_counts_are_passages_not_prevalence(ann, cb):
    stages = qualitative.run_stages(published.load_values(), ann, cb)
    assert list(stages) == ["stage1_familiarization", "stage2_coding", "stage3_themes", "stage4_joint_display"]
    themes = stages["stage3_themes"]
    assert "annotated_passages" in themes.columns
    assert not any("prevalence" in c or "frequency" in c for c in themes.columns)


def test_joint_display_preserves_tensions_and_says_insufficient_evidence(ann, cb):
    j = qualitative.run_stages(published.load_values(), ann, cb)["stage4_joint_display"].set_index("finding_id")
    assert j.loc["J1", "qualitative_status"] == "tension"  # income up, repayment stress and reliance reported
    assert "repayment_stress" in j.loc["J1", "evidence_pattern"]
    assert j.loc["J2", "qualitative_status"] == "tension"
    assert j.loc["J8", "qualitative_status"] == "convergent"
    for fid in ("J5", "J6", "J7", "J9", "J10", "J11"):  # age and education subgroups have no matching passage
        assert j.loc[fid, "qualitative_status"] == qualitative.INSUFFICIENT
        assert j.loc[fid, "supporting_annotation_ids"] == ""


def test_joint_display_links_to_published_records_only(ann, cb):
    j = qualitative.run_stages(published.load_values(), ann, cb)["stage4_joint_display"]
    assert set(j["quantitative_dataset_type"]) == {"published_aggregate"}
    assert (j["source_pages"].str.len() > 0).all()


def test_run_stages_refuses_invalid_inputs(ann, cb):
    with pytest.raises(ValueError, match="failed validation"):
        qualitative.run_stages(published.load_values(), ann.assign(kappa=1.0), cb)


def test_codebook_validation_catches_duplicates(cb):
    assert qualitative.validate_codebook(pd.concat([cb, cb.iloc[[0]]]))
    assert qualitative.validate_codebook(cb.assign(relation_to_quantitative="sideways"))
    assert qualitative.validate_codebook(cb.drop(columns=["definition"]))


# ---------------------------------------------------------------- BIDS / World Bank survey
def test_field_map_marks_unsupported_measures_honestly():
    fm = bids.field_map_frame().set_index("target")
    for t in ("income_followup", "autonomy", "social_engagement", "loan_tenure_months"):
        assert fm.loc[t, "status"] == "unsupported"
    assert fm.loc["survey_weight", "status"] == "supported"


def test_mapping_with_unset_programme_code_is_refused(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("grameen_progid: null\nmember_codes: null\nyes_codes: null\n", encoding="utf-8")
    with pytest.raises(bids.ConfigIncomplete, match="Grameen"):
        bids.load_mapping(p)


def test_shipped_mapping_is_unset_by_design():
    with pytest.raises(bids.ConfigIncomplete):
        bids.load_mapping(paths.config_dir() / "bids_mapping.yaml")


def _fictional_download(d):
    pd.DataFrame({"nh": [1, 1, 2, 3], "pid": [1, 2, 1, 1], "edslg": [5, 0, 8, 10], "edvrt": [1, 2, 2, 1]}).to_csv(d / "02r1.csv", index=False)
    pd.DataFrame({"nh": [1, 1, 2, 3], "csgpid": [1, 2, 1, 1], "csgpc": [1, 2, 1, 2], "csgpn": [7, 7, 3, 7]}).to_csv(d / "11cr1.csv", index=False)
    pd.DataFrame({"nh": [1, 2, 3], "weight": [1.5, 0.8, 1.1]}).to_csv(d / "03br1.csv", index=False)
    pd.DataFrame({"nh": [1, 1, 2], "csbpbid": [1, 1, 1], "csbprin": [1000.0, 500.0, 2000.0]}).to_csv(d / "11ar1.csv", index=False)


def test_importer_maps_supported_fields_from_a_fictional_download(tmp_path):
    _fictional_download(tmp_path)
    mapping = bids.BidsMapping(grameen_progid=7, member_codes=(7,), yes_codes=(1,))
    df = bids.import_bids(tmp_path, mapping)
    assert len(df) == 4
    assert set(df["borrower_id"]) == {"1-1", "1-2", "2-1", "3-1"}
    row = df.set_index("borrower_id")
    assert row.loc["1-1", "group_membership"] == 1.0  # programme 7 and currently a member
    assert row.loc["1-2", "group_membership"] == 0.0  # programme 7 but not a current member
    assert row.loc["2-1", "group_membership"] == 0.0  # a different programme
    assert row.loc["1-1", "loan_size"] == 1500.0  # loans summed per borrower
    assert "income_followup" not in df.columns  # unsupported measures are never invented


def test_importer_reports_missing_files_and_variables(tmp_path):
    mapping = bids.BidsMapping(7, (7,), (1,))
    with pytest.raises(FileNotFoundError, match="catalog/1317"):
        bids.import_bids(tmp_path, mapping)
    _fictional_download(tmp_path)
    pd.DataFrame({"nh": [1], "csgpid": [1]}).to_csv(tmp_path / "11cr1.csv", index=False)
    with pytest.raises(ValueError, match="data dictionary"):
        bids.import_bids(tmp_path, mapping)


def test_attrition_report_counts_households():
    r = bids.attrition_report(pd.Series([1, 2, 3, 4]), pd.Series([3, 4, 5])).iloc[0]
    assert (r["households_wave_a"], r["retained"], r["lost"], r["new_in_wave_b"]) == (4, 2, 2, 1)
    assert r["retention_rate"] == 0.5


def test_longitudinal_calculation_needs_documented_linked_compatible_waves():
    assert bids.longitudinal_blockers({"1991-92": {}})
    waves = {"a": {"ids_documented_as_linkable": True, "measure_definition": "monthly BDT"},
             "b": {"ids_documented_as_linkable": True, "measure_definition": "annual BDT"}}
    assert any("not compatible" in p for p in bids.longitudinal_blockers(waves))
    waves["b"]["measure_definition"] = "monthly BDT"
    assert bids.longitudinal_blockers(waves) == []
    assert bids.longitudinal_blockers({"a": {}, "b": {}})
