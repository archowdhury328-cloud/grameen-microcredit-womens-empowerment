# The Figure 4 discrepancy

Table 6 (p. 2585) reports a social-mobility index for three education levels:

| Education | Location (as printed) | Table 6 value |
|---|---|---|
| No formal education | Rural | 25% |
| Primary education | Mixed | 35% |
| Secondary education | Urban | 50% |

Figure 4 (p. 2586) is a pie titled *Social Mobility by Education Level* with labels **22.7%, 31.8%, 45.5%**. Those
labels are the Table 6 values divided by their sum, 25 + 35 + 50 = **110**:

    25 / 110 = 22.73%  ->  22.7%
    35 / 110 = 31.82%  ->  31.8%
    50 / 110 = 45.45%  ->  45.5%

(`grameen reproduce-paper` writes [figure4_normalization_check.csv](../results/published/figure4_normalization_check.csv)
and a test asserts all three labels match.)

## Why the pie cannot be read as a distribution of borrowers

A pie forces slices to total 100%. Here it re-expresses three independent index values as shares of their own sum.
The result describes the relative size of the three index values, not how borrowers are spread across education
levels. The paper reports no borrower counts or shares by education, so no such distribution exists to draw.

## What this repository plots instead

- **Primary chart** ([fig4_social_mobility_table6](../results/published/fig4_social_mobility_table6.png)): a dot plot
  of the three Table 6 values, each row keeping both its education level and its location label.
- **Comparison only** ([fig4_pie_normalization_comparison](../results/published/fig4_pie_normalization_comparison.png)):
  the pie as printed, with the normalisation stated in the caption.

## What Table 6 cannot identify

Each education level is paired with exactly one location (Rural, Mixed, Urban). With three rows, education and
location are perfectly confounded, so separate education and location effects cannot be estimated from these values.
