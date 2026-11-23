# Grameen microcredit and women's empowerment: reproducible companion

Code, data and generated outputs that accompany

> Karim, F., Haldar, U., Rahman, M. M., Islam, M. A., Kazi, I., Miah, M. A., Siddiqa, K. B., & Chy, M. A. R. (2024).
> *Empowering Women through Microcredit: An Analysis of the Grameen Bank Model's Impact on Socio-Economic Outcomes.*
> International Journal of Computational and Experimental Science and ENgineering (IJCESEN), 10(4), 2578-2589.
> <https://doi.org/10.22399/ijcesen.4074>

## What this repository is, and is not

The paper prints a general regression equation and lists of predictors. It prints percentages, but **no individual
incomes, group sizes, variances, coefficients or significance levels**, and its data-availability statement directs
requests to the corresponding author. So:

- A reader can reproduce the paper's **summary charts** with one command, offline, from the printed numbers.
- A reader can see **exactly which row-level fields** are needed to fit the regressions, and the code refuses to fit
  without them.
- Nothing here claims to have reproduced the paper's effects. No fictional borrower records were generated from the
  printed percentages and fitted to "confirm" them.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock && pip install -e . --no-deps
grameen reproduce-paper          # Figures 1-4 from the published values  -> results/published/
```

The same commands work offline after installation.

| Command | What it does | Dataset type of its output |
|---|---|---|
| `grameen reproduce-paper` | Rebuilds Figures 1-4 (PNG + SVG), the Figure 4 arithmetic check and a manifest | `published_aggregate` |
| `grameen run-synthetic` | Seeded fictional fixture, all supported regressions, descriptive tables, report | `synthetic` |
| `grameen validate-data` | Checks the bundled CSVs, or `--data FILE --meta FILE` for your own rows | n/a |
| `grameen import-data` | Validates your row-level CSV and copies it to `data/original/` (git-ignored) | `original_microdata` |
| `grameen fit-models` | Fits the Table 3 models. **Blocked** with an actionable message if row-level data is absent | by metadata |
| `grameen import-bids` | Imports supported fields from a local World Bank/BIDS download (needs the licensed data) | `independent_bids` |
| `grameen build-report` | Assembles [reports/report.md](reports/report.md) with provenance, figures and the joint display | mixed, labelled |

## Where each paper element lives

| Paper element | Page | Data | Code | Output |
|---|---|---|---|---|
| Equation 1, multiple regression | 2582 | [configs/models/](configs/models/) | [fitting.py](src/grameen_empowerment/fitting.py) | [results/synthetic/coefficients.csv](results/synthetic/coefficients.csv) (synthetic only) |
| Table 1, data features | 2584 | [data_dictionary.csv](data/published/data_dictionary.csv) | [published.py](src/grameen_empowerment/published.py) | dictionary only; no values are published |
| Table 2, variables and indicators | 2584-2585 | [data_dictionary.csv](data/published/data_dictionary.csv) | [circularity.py](src/grameen_empowerment/circularity.py) | dictionary only |
| Table 3, regression variables | 2585 | [configs/models/](configs/models/) | [modelspec.py](src/grameen_empowerment/modelspec.py) | [docs/MODELS.md](docs/MODELS.md) |
| Table 4, income growth by sector | 2585 | [paper_values.csv](data/published/paper_values.csv) rows `T4-*` | [figures.py](src/grameen_empowerment/figures.py) | [fig2 PNG](results/published/fig2_income_growth_by_sector.png), [SVG](results/published/fig2_income_growth_by_sector.svg) |
| Table 5, autonomy by age group | 2585 | rows `T5-*` | [agegroups.py](src/grameen_empowerment/agegroups.py) | [fig3 PNG](results/published/fig3_autonomy_by_age_group.png), [SVG](results/published/fig3_autonomy_by_age_group.svg) |
| Table 6, social mobility | 2585 | rows `T6-*` | [figures.py](src/grameen_empowerment/figures.py) | [fig4 PNG](results/published/fig4_social_mobility_table6.png), [SVG](results/published/fig4_social_mobility_table6.svg) |
| Figure 1, study design | 2584 | none (conceptual) | [figures.py](src/grameen_empowerment/figures.py) | [fig1 PNG](results/published/fig1_study_framework.png), [SVG](results/published/fig1_study_framework.svg) |
| Figure 2 | 2585 | `T4-*` | `figure2_income_by_sector` | [fig2](results/published/fig2_income_growth_by_sector.png) |
| Figure 3 | 2586 | `T5-*` | `figure3_autonomy_by_age` | [fig3](results/published/fig3_autonomy_by_age_group.png) |
| Figure 4 (pie as printed) | 2586 | [figure4_printed_labels.csv](data/published/figure4_printed_labels.csv) | [normalization.py](src/grameen_empowerment/normalization.py) | [comparison PNG](results/published/fig4_pie_normalization_comparison.png), [check CSV](results/published/figure4_normalization_check.csv) |
| Narrative figures (28%, 35%, 40%) | 2582-2583 | rows `N-*` | [published.py](src/grameen_empowerment/published.py) | [published_values.csv](results/published/published_values.csv) |
| Section 3.4 thematic analysis | 2582 | [data/qualitative/](data/qualitative/) | [qualitative.py](src/grameen_empowerment/qualitative.py) | [results/qualitative/](results/qualitative/) |

Tools named in the paper (Section 3.5, p. 2582): SPSS and Stata for regression, NVivo for thematic analysis. This
repository does not use them. The regression runs on documented open-source Python (`statsmodels` OLS/WLS/GLM and
ordered logit); the qualitative stages run on plain CSV files with `pandas`.

## The four data sources, kept apart

1. **Published aggregates** (default). [data/published/](data/published/): Tables 4-6, the narrative percentages, the
   Figure 4 printed labels and the Tables 1-2 dictionary. Every row has source table, printed page, group label,
   measure, unit and value; denominators are marked `unavailable`. See [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).
2. **Original supporting data** (not included). Schema, validator and import command are provided; access is by
   request to the paper's corresponding author. See [docs/ACCESS_ORIGINAL_DATA.md](docs/ACCESS_ORIGINAL_DATA.md).
   This repository never sends a request on anyone's behalf and never stores borrower records.
3. **Runnable synthetic example.** [data/synthetic/](data/synthetic/): 240 fictional borrowers from a seeded
   generator ([synthetic.py](src/grameen_empowerment/synthetic.py)). Not calibrated to the paper. Every row, file,
   table and figure is marked synthetic.
4. **Independent Bangladesh comparison.** World Bank / BIDS *Long-Term Impact of Microcredit Impacts 1991-1992*,
   catalog 1317. A different survey, covering Grameen Bank, BRAC and RD-12. See [docs/BIDS_ACCESS.md](docs/BIDS_ACCESS.md).

## Result provenance

| Result | Came from | Status in this release |
|---|---|---|
| Figures 1-4, Tables 4-6 values | the publication | reproduced by `reproduce-paper` |
| Regression coefficients and fit statistics in `results/synthetic/` | a **synthetic run** | demonstrates the code only |
| Qualitative joint display | passages the publication reports, one annotator | [results/qualitative/](results/qualitative/) |
| World Bank / BIDS estimates | independent microdata | **not run**: needs a local licensed download |
| Fits to the authors' own data | original microdata | **not run**: data held by the corresponding author |

## Models, qualitative coding, and honest limits

- [docs/MODELS.md](docs/MODELS.md): one config per Table 3 outcome; intercept-only and single-predictor comparisons on
  identical rows; observation counts and exclusion reasons; weights, clusters, zero-baseline income, the
  engagement-circularity check. Associations only; no causal claim.
- [docs/QUALITATIVE.md](docs/QUALITATIVE.md): codebook, annotation template, validator, joint display, and the two
  worked examples from p. 2583. No interviews, coders, agreement scores or theme prevalence are invented.
- [docs/FIGURE4.md](docs/FIGURE4.md): why the printed pie cannot show how borrowers are spread across education.
- The paper's equation prints `Y = b0 + b1X1 + ... + b2X2 + ... + bnXn + e` with `b2X2` after the first ellipsis.
  This release implements the standard `Y = b0 + sum_j bj Xj + e`.
- Section 4.1 (p. 2583) and Section 5 (p. 2587) cite "Table 3" for the 40% community-participation figure; Table 3
  lists regression variables and contains no such value. Recorded in `paper_values.csv` row `N-07`.

## Verification

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the test and coverage results, the locked environment and
the clean-checkout procedure.

## Citing, maintainer, and assistance

Cite this software with [CITATION.cff](CITATION.cff); cite the paper with all eight authors as above.
Maintainer: **Md Asikur Rahman Chy** ([GitHub](https://github.com/archowdhury328-cloud),
[Google Scholar](https://scholar.google.com/citations?user=mAM8u_UAAAAJ&hl=en)).

Portions of the code, tests and documentation were written with an AI coding assistant. The commit history records
this software release only; it cannot establish who ran the paper's original 2024 analyses.

No licence file is included yet: the repository owner has not chosen one. Until one is added, default copyright applies.
