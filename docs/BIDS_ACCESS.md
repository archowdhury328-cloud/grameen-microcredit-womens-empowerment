# Independent comparison: World Bank / BIDS microcredit survey

**This is a different survey from the paper's sources.** Using it does not reproduce, test or validate the paper.

| | |
|---|---|
| Title | Long-Term Impact of Microcredit Impacts 1991-1992 |
| Reference | BGD_1991_LTIMCI_v01_M |
| Catalog | <https://microdata.worldbank.org/catalog/1317> |
| DOI | <https://doi.org/10.48529/anw3-f530> |
| Producers | Shahidur R. Khandker (World Bank); Bangladesh Institute of Development Studies |
| Design | 1,798 households from 87 villages in 29 thanas; first surveyed 1991-92 in three rounds, revisited 1998-99 |
| Programmes | Grameen Bank, BRAC, and the BRDB RD-12 project, plus non-programme villages |
| Version | v01, edited anonymous datasets for public distribution |

Read from the catalog pages on 2026-11-24. Re-check them: they are the authority, not this note.

## Access terms (summary of the "Get Microdata" page)

- Data are obtained from the Microdata Library after registration. **They may not be redistributed or sold** without
  written agreement, so **nothing from this survey is committed here** (everything in `data/external/` except its README is git-ignored).
- Statistical and scientific research only; no attempt to identify respondents or institutions.
- The terms also say no attempt may be made to *"create links among datasets provided by the Microdata Library, or
  among data from the Microdata Library and other datasets that could identify individuals or organizations."*
  Linking the 1991-92 rounds to the 1998-99 revisit by anonymised household id is how the survey is normally used, but
  read the terms yourself, and ask the Microdata Library (`microdata@worldbank.org`) if unsure.
- Cite as: *World Bank, Bangladesh Institute of Development Studies. Long-Term Impact of Microcredit Impacts 1991-1992,
  Ref. BGD_1991_LTIMCI_v01_M. Dataset downloaded from [url] on [date].*

## Dictionary inspected before mapping

Public data-dictionary pages list 101 files (rounds `r1`-`r3`). Variables below were read from those pages; the coding
of their values was **not** visible without the download and must be checked in the downloaded dictionary.

| Target field | Source | Status |
|---|---|---|
| `household_id` | `nh` (household number) | supported |
| `survey_weight` | `weight` in `03br1` (household sampling weight) | supported |
| `education_level` | `edslg` in `02r1` (highest grade completed) | supported; recode only after reading labels |
| `training_participation` | `edvrt` in `02r1` (ever received technical training) | supported; not the same construct as the paper's "participation in training" |
| `group_membership` | `csgpc` (currently a member) and `csgpn` (which programme) in `11cr1` | supported; programme code must be supplied |
| `loan_size` | `csbprin` in `11ar1` (loan principal), summed per borrower | supported; check currency and units |
| `loan_tenure_months` | only due-date fields (`csbtpbd/m/y`) | **unsupported** |
| `income_followup` | no single income variable; wage, self-employment and farm income are in different modules | **unsupported** |
| `autonomy` | no household decision-making module among the 12 questionnaire sections | **unsupported** |
| `social_engagement` | no community-participation module among the 12 sections | **unsupported** |

The survey is therefore usable here only for the *predictor-side* fields and weights. It cannot support the paper's
three outcomes without a documented construction that this repository does not claim to provide.

## Grameen-specific estimates

A result is Grameen-specific only after filtering to a documented programme identifier. The shipped
[configs/bids_mapping.yaml](../configs/bids_mapping.yaml) leaves `grameen_progid`, `member_codes` and `yes_codes`
**unset on purpose**; `grameen import-bids` refuses to run until you fill them in from the value labels in your
download. The village file also carries `gbh` and `gbt` indicators (Grameen households present in the village;
village in a Grameen thana), which are area-level, not household participation.

## Longitudinal calculations

The 1998-99 revisit is a separate catalog entry. A two-wave calculation needs documented linked waves, compatible
measures and an attrition report. [bids.py](../src/grameen_empowerment/bids.py) provides `attrition_report` and
`longitudinal_blockers`, which list what is undocumented. Nothing is computed across waves by default.

## Local run

```bash
# after downloading the round-1 files you are licensed to use into a folder (not inside Git):
#   02r1, 03br1, 11ar1, 11cr1  (.csv or .dta)
grameen import-bids --input /path/to/download --out data/external/bids_person_level.csv
```

Status in this release: **not run**, because the licensed data are not present on the build machine.
The importer is tested only against a tiny fictional fixture that mimics the variable names.
