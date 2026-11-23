from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from grameen_empowerment import cli, paths

REPO = Path(__file__).resolve().parents[1]


def run(args, capsys):
    code = cli.main(args)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_reproduce_paper_writes_png_and_svg_for_every_figure(tmp_path, capsys):
    code, out, _ = run(["reproduce-paper", "--out", str(tmp_path)], capsys)
    assert code == 0 and "published_aggregate" in out
    for stem in ("fig1_study_framework", "fig2_income_growth_by_sector", "fig3_autonomy_by_age_group",
                 "fig4_social_mobility_table6", "fig4_pie_normalization_comparison"):
        assert (tmp_path / f"{stem}.png").stat().st_size > 5000
        assert "<svg" in (tmp_path / f"{stem}.svg").read_text(encoding="utf-8")[:600]


def test_every_figure_states_its_source_and_dataset_type(tmp_path, capsys):
    cli.main(["reproduce-paper", "--out", str(tmp_path)])
    capsys.readouterr()
    for svg in tmp_path.glob("fig*.svg"):
        text = svg.read_text(encoding="utf-8")
        assert "IJCESEN 10(4):2578-2589" in text
        assert "dataset type" in text
    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert {"output", "source", "dataset_type", "command"} <= set(manifest.columns)


def test_figure4_pie_caption_states_the_normalization(tmp_path, capsys):
    cli.main(["reproduce-paper", "--out", str(tmp_path)])
    capsys.readouterr()
    text = (tmp_path / "fig4_pie_normalization_comparison.svg").read_text(encoding="utf-8")
    assert "110" in text and "not the distribution of" in text


def test_run_synthetic_command(tmp_path, capsys):
    code, out, _ = run(["run-synthetic", "--out", str(tmp_path)], capsys)
    assert code == 0 and "SYNTHETIC" in out
    assert (tmp_path / "synthetic_regression_report.md").exists()


def test_validate_data_bundled_and_explicit(capsys):
    code, out, _ = run(["validate-data"], capsys)
    assert code == 0 and "published_values: ok" in out
    csv, meta = paths.synthetic_fixture()
    code, out, _ = run(["validate-data", "--data", str(csv), "--meta", str(meta)], capsys)
    assert code == 0 and "supported outcomes" in out
    code, _, err = run(["validate-data", "--data", str(csv)], capsys)
    assert code == 1 and "--meta" in err


def test_fit_models_without_data_is_blocked_with_an_actionable_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GRAMEEN_ROOT", str(tmp_path))
    code, _, err = run(["fit-models"], capsys)
    assert code == 2
    assert "corresponding author" in err and "never contacts anyone" in err and "import-data" in err


def test_fit_models_refuses_the_published_aggregates(tmp_path, capsys):
    meta = tmp_path / "m.yaml"
    meta.write_text("dataset_id: agg\ndataset_type: synthetic\n", encoding="utf-8")
    code, _, err = run(["fit-models", "--data", str(paths.published_values_csv()), "--meta", str(meta),
                        "--out", str(tmp_path / "o")], capsys)
    assert code == 2 and "aggregates" in err
    assert not (tmp_path / "o" / "coefficients.csv").exists()


def test_fit_models_on_synthetic_goes_to_the_synthetic_folder(tmp_path, monkeypatch, capsys):
    csv, meta = paths.synthetic_fixture()
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    monkeypatch.setenv("GRAMEEN_ROOT", str(tmp_path))
    code, out, _ = run(["fit-models", "--data", str(csv), "--meta", str(meta), "--allow-unweighted"], capsys)
    assert code == 0 and "dataset type synthetic" in out
    assert (tmp_path / "results" / "synthetic" / "coefficients.csv").exists()
    assert not (tmp_path / "results" / "original").exists()


def test_import_data_validates_copies_and_keeps_rows_out_of_git(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GRAMEEN_ROOT", str(tmp_path))
    from grameen_empowerment import synthetic

    df = synthetic.generate(n=120).drop(columns=["is_synthetic"])
    df["borrower_id"] = [f"R{i}" for i in range(len(df))]
    meta = synthetic.metadata(120, 1)
    meta["dataset_type"] = "original_microdata"
    import yaml

    (tmp_path / "in.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (tmp_path / "in.meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    code, out, _ = run(["import-data", "--source", str(tmp_path / "in.csv"), "--meta", str(tmp_path / "in.meta.yaml")], capsys)
    assert code == 0 and "git-ignored" in out
    assert (tmp_path / "data" / "original" / "borrowers.csv").exists()
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "data/original/" in ignore


def test_import_data_rejects_invalid_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GRAMEEN_ROOT", str(tmp_path))
    pd.DataFrame({"borrower_id": ["a"], "x": [1]}).to_csv(tmp_path / "in.csv", index=False)
    (tmp_path / "m.yaml").write_text("dataset_id: x\ndataset_type: original_microdata\n", encoding="utf-8")
    code, _, _ = run(["import-data", "--source", str(tmp_path / "in.csv"), "--meta", str(tmp_path / "m.yaml")], capsys)
    assert code == 2
    assert not (tmp_path / "data" / "original" / "borrowers.csv").exists()


def test_import_bids_is_blocked_until_programme_codes_are_set(tmp_path, capsys):
    code, _, err = run(["import-bids", "--input", str(tmp_path), "--out", str(tmp_path / "o.csv")], capsys)
    assert code == 2 and "Grameen" in err


def test_build_report_assembles_provenance_and_joint_display(tmp_path, capsys):
    code, out, _ = run(["build-report", "--out", str(tmp_path / "reports")], capsys)
    assert code == 0
    text = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")
    for needle in ("published_aggregate", "synthetic", "independent_bids", "NOT run", "insufficient evidence",
                   "Figure 4 discrepancy", "SYNTHETIC"):
        assert needle in text, needle
    assert (tmp_path / "results" / "published" / "fig2_income_growth_by_sector.png").exists()
    assert (tmp_path / "results" / "qualitative" / "stage4_joint_display.csv").exists()


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_bad_input_returns_error_code(tmp_path, capsys):
    code, _, err = run(["validate-data", "--data", str(tmp_path / "missing.csv"), "--meta", str(tmp_path / "m.yaml")], capsys)
    assert code == 1 and "error" in err


@pytest.mark.slow
def test_clean_checkout_reproduces_figures_and_one_synthetic_report(tmp_path):
    """Copy only the tracked source tree to an empty directory and run the published commands there."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for item in ("src", "data", "configs", "pyproject.toml", "README.md"):
        src = REPO / item
        if src.is_dir():
            shutil.copytree(src, checkout / item, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "original"))
        else:
            shutil.copy(src, checkout / item)
    assert not (checkout / "results").exists()
    env = {**os.environ, "PYTHONPATH": str(checkout / "src"), "GRAMEEN_ROOT": str(checkout)}
    for cmd in (["reproduce-paper"], ["run-synthetic"]):
        proc = subprocess.run([sys.executable, "-m", "grameen_empowerment.cli", *cmd], cwd=checkout, env=env,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
    pub, syn = checkout / "results" / "published", checkout / "results" / "synthetic"
    for stem in ("fig2_income_growth_by_sector", "fig3_autonomy_by_age_group", "fig4_social_mobility_table6"):
        assert (pub / f"{stem}.png").exists() and (pub / f"{stem}.svg").exists()
    report = (syn / "synthetic_regression_report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in report and "M2_table3_full" in report
