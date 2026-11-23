# Regression models

Associations only. Nothing here estimates a causal effect of the programme, and there is no supervised-learning
leaderboard.

## One config per Table 3 outcome

| Config | Outcome | Paper-listed predictors | Comparison predictor |
|---|---|---|---|
| [income_level.yaml](../configs/models/income_level.yaml) | measured follow-up income | loan size, loan tenure, borrower education | `loan_size` |
| [decision_autonomy.yaml](../configs/models/decision_autonomy.yaml) | decision-making autonomy | training participation, group membership | `training_participation` |
| [social_engagement.yaml](../configs/models/social_engagement.yaml) | social engagement | community-meeting attendance, leadership roles | `community_meeting_attendance` |
| [income_growth.yaml](../configs/models/income_growth.yaml) | percent income change (optional variant) | as `income_level` | `loan_size` |

Equation 1 (p. 2582) is implemented as `Y = b0 + sum_j bj Xj + e`. The printed form places `b2X2` after the first
ellipsis, which looks like a typesetting slip.

## Three models, same rows

For every supported outcome (`fit_outcome` in [fitting.py](../src/grameen_empowerment/fitting.py)):

| Model id | Role | Terms |
|---|---|---|
| `M0_intercept_only` | repository comparison | intercept |
| `M1_single_<predictor>` | repository comparison | the config's comparison predictor |
| `M2_table3_full` | Table 3 specification | every paper-listed predictor |

All three are fitted on the **same eligible rows**: a row is eligible only if the outcome, every full-model
predictor, the weight and the cluster id are all present and valid. Output tables report the row count and the count
excluded for each reason (`sample_accounting.csv`): `missing_outcome`, `missing_predictor:<name>`,
`unknown_category:<name>`, `invalid_outcome_value`, `missing_weight`, `nonpositive_weight`, `missing_cluster_id`,
`zero_baseline_income`, `negative_baseline_income`, `missing_baseline_income`. Each row is counted under its first
failing reason only, so the counts add up: `rows_in_file - exclusions = rows_eligible`. Row-level exclusion lists are
never written, so original microdata cannot leak through them.

## Estimator follows the documented measure

Declared in the metadata sidecar (`<section>.measure_type`) and echoed in every output row with its rationale:

| `measure_type` | Estimator |
|---|---|
| `continuous` | linear regression (OLS, or WLS with survey weights) |
| `binary` | logistic regression (GLM binomial) |
| `ordinal` | ordered logit (needs `levels`); statsmodels cannot apply survey weights, so it is blocked unless `--allow-unweighted` is passed, and the output is then labelled unweighted |

Income is always continuous. The metadata must record income's `unit_level` (individual or household),
`reference_period` and `currency`; the model is blocked without them.

## Survey design

- `survey_weight`, when present and positive, is applied (WLS / frequency weights). AIC and BIC are then not reported,
  because they are not defined under weights.
- `psu` (preferred) or `household_id` defines clusters; standard errors are cluster-robust. Repeated households
  therefore do not overstate precision. Fewer than 10 clusters triggers a note.
- Stratification is recorded but **not** modelled in the variance, and finite-population corrections are not applied.
  For full design-based inference use a survey package such as R `survey`.
- Without clusters, HC1 robust errors are used.

## Checks before any model is fitted

Absent variation in the outcome or a predictor; categorical reference level missing from the eligible rows (or only
one level observed); rank-deficient design matrix; perfect or quasi-perfect separation in logistic models;
non-convergence in ordered logit; too few rows (at least 30 and 10 per parameter). Each raises an error that says
what to change.

## Income growth and the zero baseline

A model of income *growth* needs `income_baseline` and `income_followup` documented as comparable
(`income.baseline_comparable: true`: same unit level, reference period and currency). Percent change is
`100 * (follow-up - baseline) / baseline`. It is **undefined at a zero baseline**, so those rows are excluded and
counted under `zero_baseline_income`; they are never imputed. Negative and missing baselines are excluded the same way.

## Social engagement is checked for circularity

Table 2 (p. 2584) lists *membership in social groups, attendance at community meetings, and leadership roles* as
engagement indicators, while Table 3 uses attendance and leadership as predictors. An index built from them would be
regressed on its own components. The metadata must therefore list `social_engagement.components`. If either predictor
is among them the model is blocked as circular and an independently measured engagement outcome is required
([circularity.py](../src/grameen_empowerment/circularity.py)). The synthetic fixture measures engagement from
social-group, local-market and political-forum participation, which excludes both predictors.

## Output tables (tidy CSV)

`coefficients.csv` (term, estimate, std_error, statistic, p_value, ci_low, ci_high, plus odds ratios for logistic and
ordered models), `fit_statistics.csv` (log-likelihood, AIC/BIC where defined, R-squared or McFadden pseudo R-squared,
n_obs, weights_used, cluster_used, cov_type), `sample_accounting.csv`, `model_status.csv`. Every row carries
`dataset_type`, `dataset_id` and `is_synthetic`. Ordered-logit thresholds use the statsmodels parameterisation (first
cut raw, later cuts as log-increments) and are labelled `threshold_param:`.
