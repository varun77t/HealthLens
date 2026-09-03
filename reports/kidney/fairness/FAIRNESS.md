# Subgroup performance — Chronic Kidney Disease Presence Prediction

> **For research and educational purposes only.** This is a description of how one model behaved on one small held-out sample. It is not an audit, and it does not establish that the model is fair or safe to use.

Held-out test set: **n = 80**. Every figure below comes from the shipped pipeline's predictions on those rows; no model was retrained.

## How to read this

A subgroup is marked **reliable** only if it holds at least 20 rows with at least 10 positives and 10 negatives. Rates carry **Wilson 95% intervals**; ROC-AUC carries a Hanley–McNeil interval.

A gap is called a **material difference** only when it clears two separate bars: the intervals must be disjoint (so it is not sampling noise), **and** the gap must be at least 0.05. The second bar matters as much as the first — with 50,961 test rows the intervals become narrow enough that almost any difference separates, so statistical clarity stops implying importance. Small-sample noise and large-sample over-sensitivity are opposite failures and both are named here.

Equal metrics across groups would **not** demonstrate fairness. Only sex and age are available here, the labels carry their own measurement bias, and parity on two attributes says nothing about who a model harms.

### Attributes that could not be analysed

- **sex** — not recorded in this dataset — no sex breakdown is possible, and its absence cannot be worked around without inventing data.

**0 of 4 subgroups meet the minimum counts.**

## By age_band

### At threshold 0.5

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45-54 | 12 | 8 | 4 | 0.6667 | 1.0000 | 0.6756 | 1.0000 | 1.0000 | 0.5101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| 55-64 | 18 | 12 | 6 | 0.6667 | 1.0000 | 0.7575 | 1.0000 | 1.0000 | 0.6097 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| 65+ | 25 | 22 | 3 | 0.8800 | 1.0000 | 0.8513 | 1.0000 | 1.0000 | 0.4385 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| <45 | 24 | 7 | 17 | 0.2917 | 0.7143 | 0.3589 | 0.9178 | 0.9412 | 0.7302 | 0.9895 | 0.9832 | 0.9138 | 1.0000 | False |

Caveats:

- **45-54** — only 12 rows (min 20); only 8 positives (min 10) — recall is not estimable; only 4 negatives (min 10) — specificity is not estimable.
- **55-64** — only 18 rows (min 20); only 6 negatives (min 10) — specificity is not estimable.
- **65+** — only 3 negatives (min 10) — specificity is not estimable.
- **<45** — only 7 positives (min 10) — recall is not estimable.

Disparity assessment:

- inconclusive — recall: 45-54 1.0000 [0.6756, 1.0000] vs <45 0.7143 [0.3589, 0.9178], gap 0.2857. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — specificity: 45-54 1.0000 [0.5101, 1.0000] vs <45 0.9412 [0.7302, 0.9895], gap 0.0588. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — precision: 45-54 1.0000 [0.6756, 1.0000] vs <45 0.8333 [0.4365, 0.9699], gap 0.1667. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — roc_auc: 45-54 1.0000 [1.0000, 1.0000] vs <45 0.9832 [0.9138, 1.0000], gap 0.0168. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.

### At sensitivity-oriented threshold 0.5792

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45-54 | 12 | 8 | 4 | 0.6667 | 1.0000 | 0.6756 | 1.0000 | 1.0000 | 0.5101 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| 55-64 | 18 | 12 | 6 | 0.6667 | 1.0000 | 0.7575 | 1.0000 | 1.0000 | 0.6097 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| 65+ | 25 | 22 | 3 | 0.8800 | 1.0000 | 0.8513 | 1.0000 | 1.0000 | 0.4385 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| <45 | 24 | 7 | 17 | 0.2917 | 0.7143 | 0.3589 | 0.9178 | 1.0000 | 0.8157 | 1.0000 | 0.9832 | 0.9138 | 1.0000 | False |

Caveats:

- **45-54** — only 12 rows (min 20); only 8 positives (min 10) — recall is not estimable; only 4 negatives (min 10) — specificity is not estimable.
- **55-64** — only 18 rows (min 20); only 6 negatives (min 10) — specificity is not estimable.
- **65+** — only 3 negatives (min 10) — specificity is not estimable.
- **<45** — only 7 positives (min 10) — recall is not estimable.

Disparity assessment:

- inconclusive — recall: 45-54 1.0000 [0.6756, 1.0000] vs <45 0.7143 [0.3589, 0.9178], gap 0.2857. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — specificity: 45-54 1.0000 [0.5101, 1.0000] vs 45-54 1.0000 [0.5101, 1.0000], gap 0.0000. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — precision: 45-54 1.0000 [0.6756, 1.0000] vs 45-54 1.0000 [0.6756, 1.0000], gap 0.0000. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — roc_auc: 45-54 1.0000 [1.0000, 1.0000] vs <45 0.9832 [0.9138, 1.0000], gap 0.0168. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.

ROC-AUC is identical across the two threshold blocks above — it does not depend on the cut-off. Only the rate metrics move, and the size of that movement is a direct measure of how much a threshold-based subgroup comparison reflects the threshold rather than the model.

One caveat specific to age: age is itself one of the model's strongest features, so computing ROC-AUC *within* an age band removes most of the variance in that feature. Within-band AUC is therefore not directly comparable to the overall test AUC. Comparing bands against each other remains meaningful, but the bands span different widths of the underlying age range, so some of the difference between them reflects that rather than model behaviour.

## Calibration

Wrapper applied: `raw` — the base model was already the best-calibrated option on training out-of-fold Brier, so the before/after columns are identical by construction

| | Brier | ECE | ROC-AUC |
|---|---|---|---|
| before (raw) | 0.012681 | 0.036171 | 0.9987 |
| after | 0.012681 | 0.036171 | 0.9987 |
| delta | +0.000000 | +0.000000 | +0.0000 |

ECE is the population-weighted mean gap between predicted probability and observed frequency across 10 bins. It isolates calibration, where Brier mixes calibration with discrimination — but ECE alone is blind to ranking, so a model predicting the base rate for everyone would score near-perfectly. Read the two together.

_Brier and ECE are computed on the held-out test set for reporting. The calibration method itself was chosen on training out-of-fold Brier alone; these figures played no part in that choice._

See `figures/calibration_before_after.png` for the reliability curves.

---
*Generated by `scripts/fairness_report.py`. Every number above comes from that run.*
