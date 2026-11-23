# Access note: the paper's original supporting data

The paper's data-availability statement (p. 2588) says the data "are available on request from the corresponding
author" and "are not publicly available due to privacy or ethical restrictions". The corresponding author is named on
the paper's first page (marked with an asterisk). Contact details are printed there; they are not repeated here.

## What this repository does and does not do

- It ships a schema ([DATA_DICTIONARY.md](DATA_DICTIONARY.md) section 4), a validator (`grameen validate-data`) and a
  local import command (`grameen import-data`). Together they list the exact fields needed to fit the regressions.
- It **never** sends a request, email or message on anyone's behalf.
- It **never** stores borrower records, identifiers or private source texts in Git. `data/original/` is git-ignored,
  and a test asserts the ignore rule is present.
- `grameen fit-models` is **blocked** without row-level data and says why; it also refuses a table of published
  percentages if one is passed by mistake.

## If you obtain the data

1. Make sure you may hold and process it under the terms you were given.
2. Build a CSV with the fields in the schema and a `.meta.yaml` sidecar (copy
   [data/synthetic/synthetic_fixture.meta.yaml](../data/synthetic/synthetic_fixture.meta.yaml) and set
   `dataset_type: original_microdata`; remove the "FICTIONAL" text and describe the real measures).
3. Document how autonomy was measured (continuous, binary or ordinal), the income unit, reference period and currency,
   the education coding, and the components of any engagement index.
4. `grameen validate-data --data your.csv --meta your.meta.yaml`
5. `grameen import-data --source your.csv --meta your.meta.yaml`
6. `grameen fit-models` writes to `results/original/` (git-ignored by convention; review before sharing any table,
   since small cells can identify people).

## What the data must contain to support each outcome

| Outcome | Required fields |
|---|---|
| Income level | `income_followup`, `loan_size`, `loan_tenure_months`, `education_level`; income documented |
| Income growth | the above plus `income_baseline`, documented as comparable |
| Decision autonomy | `autonomy` (type documented), `training_participation`, `group_membership` |
| Social engagement | `social_engagement` measured independently of the two predictors, `community_meeting_attendance`, `leadership_role` |
