# Reproducibility and verification

## Environment

- Python 3.10-3.12. Runtime: numpy, pandas, matplotlib, scipy, statsmodels, pyyaml.
- [requirements.lock](../requirements.lock) pins every package (runtime and dev) as installed in a clean virtual
  environment on Python 3.10. `pip install -r requirements.lock && pip install -e . --no-deps` reproduces it.
  [pyproject.toml](../pyproject.toml) holds the looser compatible ranges.
- `reproduce-paper` and `run-synthetic` need no network after installation.
- `GRAMEEN_ROOT` relocates the repository root (data, configs, results); the clean-checkout test uses it.

## Commands and what each writes

| Command | Writes (default) | Source of every number |
|---|---|---|
| `grameen reproduce-paper` | `results/published/` figures 1-4 (PNG, SVG), `figure4_normalization_check.csv`, `published_values.csv`, `manifest.csv` | the publication |
| `grameen run-synthetic` | `results/synthetic/` data copy, `coefficients.csv`, `fit_statistics.csv`, `sample_accounting.csv`, `model_status.csv`, descriptive CSVs, forest plot, `synthetic_regression_report.md` | a synthetic run |
| `grameen validate-data` | nothing | n/a |
| `grameen fit-models` | `results/<synthetic|original|bids>/` by dataset type | the metadata's dataset type |
| `grameen build-report` | `reports/report.md`, `results/qualitative/*.csv` (and refreshes the two folders above) | mixed, each labelled |

`results/published/manifest.csv` lists every output with its source and dataset type.

## Tests

```bash
pip install -e ".[dev]" && pytest          # fails if coverage drops below 80%
```

Tests cover: exact transcribed values and pages; the Figure 4 arithmetic (division by 110); the Table 5 age
boundaries (age 45 is in 30-45); zero-baseline income exclusion; the three models using identical rows with counted
exclusion reasons; closed-form checks of OLS and weighted least squares; estimator choice by measure type; rank
deficiency, absent variation, missing reference level, separation and minimum-size guards; the circular-engagement
block; refusal to fit published aggregates; separation of synthetic from other dataset types (labels, folders, a
byte-for-byte check of the committed fixture against the generator); qualitative validators and joint-display
statuses; the BIDS importer on a fictional fixture; the CLI exit codes; and a **clean-checkout integration test** that
copies only the tracked source tree to an empty directory, runs `reproduce-paper` and `run-synthetic` there, and
asserts the three quantitative figures (PNG and SVG) and the synthetic report exist.

Last run (Python 3.10.11, Windows, packages as in `requirements.lock`): **137 passed, 0 failed; statement coverage 97.86%** (1,214 statements, 26 missed; the gate is 80%). `ruff check src tests` is clean. Reproduce with `pytest`.

## Known limits

- Figures are produced by matplotlib; PNG bytes can differ slightly across platforms and library versions. The data
  behind them and the SVG text are stable.
- Weighted models use analytic/frequency weights with cluster-robust errors; stratification is not modelled.
- The ordered-logit threshold standard errors are not reported for the intercept-only baseline.
- The BIDS importer has only been run on a fictional fixture, never on the licensed data.
