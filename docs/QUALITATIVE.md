# Qualitative coding and the joint display

The paper (Section 3.4, p. 2582) describes four thematic-analysis stages and says NVivo was used. This repository
implements the same four stages on plain CSV files, with no proprietary software.

| Stage | Function | Output |
|---|---|---|
| 1. Familiarisation | `stage1_familiarization` | passages read, by source and evidence type |
| 2. Coding | `stage2_coding` | each passage's code, linked outcome and relation to the quantitative result |
| 3. Theme identification | `stage3_themes` | codes grouped into the six themes, with passage counts |
| 4. Interpretation | `stage4_interpretation` | the **joint display** linking each quantitative finding to qualitative evidence |

Files: [codebook.csv](../data/qualitative/codebook.csv),
[annotation_template.csv](../data/qualitative/annotation_template.csv) (headers only),
[annotations_paper_examples.csv](../data/qualitative/annotations_paper_examples.csv). Outputs:
[results/qualitative/](../results/qualitative/), and section 5 of [reports/report.md](../reports/report.md).

## Themes

financial independence, household decision authority, community participation, repayment stress, reliance on repeated
borrowing, continued family control of funds. Each has a definition, an inclusion rule, an exclusion rule, a linked
outcome and a relation to the quantitative result (convergent, divergent or mixed).

## Annotation format and validator

Every annotation needs a `source_id`, a page or passage `locator`, a `theme` (consistent with its `code_id`) and an
`analyst_note`. `grameen`'s validator (`qualitative.validate_annotations`) also rejects any column that records
agreement scores, prevalence, frequency or respondent counts, so such figures cannot be added by accident.

## Worked examples: evidence reported in the paper

These are brief paraphrases of passages on **p. 2583**. They are evidence *reported in the paper*, not interviews
collected by this repository.

**Example 1: business income paired with repayment pressure** (annotations A-01, A-02).
Table 4 reports a 32% income increase for small business (p. 2585). One reported Dhaka participant says she now runs
her own shop and contributes to the family's needs (A-01, financial independence, convergent). One reported Sylhet
participant says much of what her business earns goes to repaying the loan, so the money does not feel like hers
(A-02, repayment stress, divergent). The display keeps both: an income gain and repayment strain are reported
together, and the paper does not net them out.

**Example 2: managing a shop without controlling its earnings** (annotation A-03).
A reported Sylhet participant says she manages the shop but her husband handles the money and insists on overseeing
it, though she feels she should be in charge (continued family control, divergent from the autonomy gain). The paper
does not say this is the same woman as in A-02, so the two are not merged.

Other annotations: A-04 (Chittagong participant, community participation), A-05 and A-06 (the authors' summary of
interviews on household finance control and on repayment strain in agriculture), A-07 (p. 2584, the authors' summary of
dependency on continued loans). Annotations A-05 to A-07 are **author summaries**, labelled as such.

## Reading the joint display

`qualitative_status` is one of `convergent`, `divergent`, `tension` (both) or **`insufficient evidence`** (no annotated
passage matches the finding). With the shipped annotations, the age-group and education-group results (Tables 5 and 6)
have no matching passage, because the paper reports no respondent attributes for its quotations, so they read
`insufficient evidence`. That is the correct reading, not a gap to fill.

## What is not here, deliberately

No interviews, respondents, independent coders, agreement scores or theme prevalence are invented. The single
annotator is labelled `release-annotator`. Counts in stage 3 are counts of annotated passages in this file, not
frequencies in any population. To add real data, append rows to a copy of the template with
`evidence_type: new_primary_data`, run the validator, and report coder agreement only if a second coder genuinely
exists.
