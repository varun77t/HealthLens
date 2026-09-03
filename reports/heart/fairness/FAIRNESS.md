# Subgroup performance — Heart Disease Presence Prediction

> **For research and educational purposes only.** This is a description of how one model behaved on one small held-out sample. It is not an audit, and it does not establish that the model is fair or safe to use.

Held-out test set: **n = 61**. Every figure below comes from the shipped pipeline's predictions on those rows; no model was retrained.

## How to read this

A subgroup is marked **reliable** only if it holds at least 20 rows with at least 10 positives and 10 negatives. Rates carry **Wilson 95% intervals**; ROC-AUC carries a Hanley–McNeil interval.

A gap is called a **material difference** only when it clears two separate bars: the intervals must be disjoint (so it is not sampling noise), **and** the gap must be at least 0.05. The second bar matters as much as the first — with 50,961 test rows the intervals become narrow enough that almost any difference separates, so statistical clarity stops implying importance. Small-sample noise and large-sample over-sensitivity are opposite failures and both are named here.

Equal metrics across groups would **not** demonstrate fairness. Only sex and age are available here, the labels carry their own measurement bias, and parity on two attributes says nothing about who a model harms.

**1 of 6 subgroups meet the minimum counts.**

## By sex

### At threshold 0.5

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| female | 20 | 7 | 13 | 0.3500 | 0.7143 | 0.3589 | 0.9178 | 1.0000 | 0.7719 | 1.0000 | 0.9890 | 0.9322 | 1.0000 | False |
| male | 41 | 21 | 20 | 0.5122 | 0.9524 | 0.7733 | 0.9915 | 0.8000 | 0.5840 | 0.9193 | 0.9381 | 0.8604 | 1.0000 | True |

Caveats:

- **female** — only 7 positives (min 10) — recall is not estimable.

Disparity assessment:

- inconclusive — recall: male 0.9524 [0.7733, 0.9915] vs female 0.7143 [0.3589, 0.9178], gap 0.2381. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — specificity: female 1.0000 [0.7719, 1.0000] vs male 0.8000 [0.5840, 0.9193], gap 0.2000. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — precision: female 1.0000 [0.5655, 1.0000] vs male 0.8333 [0.6415, 0.9332], gap 0.1667. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — roc_auc: female 0.9890 [0.9322, 1.0000] vs male 0.9381 [0.8604, 1.0000], gap 0.0509. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.

### At sensitivity-oriented threshold 0.6436

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| female | 20 | 7 | 13 | 0.3500 | 0.7143 | 0.3589 | 0.9178 | 1.0000 | 0.7719 | 1.0000 | 0.9890 | 0.9322 | 1.0000 | False |
| male | 41 | 21 | 20 | 0.5122 | 0.8571 | 0.6536 | 0.9502 | 0.9000 | 0.6990 | 0.9721 | 0.9381 | 0.8604 | 1.0000 | True |

Caveats:

- **female** — only 7 positives (min 10) — recall is not estimable.

Disparity assessment:

- inconclusive — recall: male 0.8571 [0.6536, 0.9502] vs female 0.7143 [0.3589, 0.9178], gap 0.1428. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — specificity: female 1.0000 [0.7719, 1.0000] vs male 0.9000 [0.6990, 0.9721], gap 0.1000. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — precision: female 1.0000 [0.5655, 1.0000] vs male 0.9000 [0.6990, 0.9721], gap 0.1000. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — roc_auc: female 0.9890 [0.9322, 1.0000] vs male 0.9381 [0.8604, 1.0000], gap 0.0509. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.

ROC-AUC is identical across the two threshold blocks above — it does not depend on the cut-off. Only the rate metrics move, and the size of that movement is a direct measure of how much a threshold-based subgroup comparison reflects the threshold rather than the model.

## By age_band

### At threshold 0.5

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45-54 | 15 | 4 | 11 | 0.2667 | 1.0000 | 0.5101 | 1.0000 | 1.0000 | 0.7412 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| 55-64 | 27 | 18 | 9 | 0.6667 | 0.8333 | 0.6078 | 0.9416 | 0.7778 | 0.4526 | 0.9368 | 0.8889 | 0.7658 | 1.0000 | False |
| 65+ | 6 | 3 | 3 | 0.5000 | 1.0000 | 0.4385 | 1.0000 | 0.3333 | 0.0615 | 0.7923 | 1.0000 | 1.0000 | 1.0000 | False |
| <45 | 13 | 3 | 10 | 0.2308 | 1.0000 | 0.4385 | 1.0000 | 1.0000 | 0.7225 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |

Caveats:

- **45-54** — only 15 rows (min 20); only 4 positives (min 10) — recall is not estimable.
- **55-64** — only 9 negatives (min 10) — specificity is not estimable.
- **65+** — only 6 rows (min 20); only 3 positives (min 10) — recall is not estimable; only 3 negatives (min 10) — specificity is not estimable.
- **<45** — only 13 rows (min 20); only 3 positives (min 10) — recall is not estimable.

Disparity assessment:

- inconclusive — recall: 45-54 1.0000 [0.5101, 1.0000] vs 55-64 0.8333 [0.6078, 0.9416], gap 0.1667. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — specificity: 45-54 1.0000 [0.7412, 1.0000] vs 65+ 0.3333 [0.0615, 0.7923], gap 0.6667. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — precision: 45-54 1.0000 [0.5101, 1.0000] vs 65+ 0.6000 [0.2307, 0.8824], gap 0.4000. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — roc_auc: 45-54 1.0000 [1.0000, 1.0000] vs 55-64 0.8889 [0.7658, 1.0000], gap 0.1111. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.

### At sensitivity-oriented threshold 0.6436

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45-54 | 15 | 4 | 11 | 0.2667 | 1.0000 | 0.5101 | 1.0000 | 1.0000 | 0.7412 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| 55-64 | 27 | 18 | 9 | 0.6667 | 0.7778 | 0.5479 | 0.9100 | 0.7778 | 0.4526 | 0.9368 | 0.8889 | 0.7658 | 1.0000 | False |
| 65+ | 6 | 3 | 3 | 0.5000 | 1.0000 | 0.4385 | 1.0000 | 1.0000 | 0.4385 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |
| <45 | 13 | 3 | 10 | 0.2308 | 0.6667 | 0.2077 | 0.9385 | 1.0000 | 0.7225 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False |

Caveats:

- **45-54** — only 15 rows (min 20); only 4 positives (min 10) — recall is not estimable.
- **55-64** — only 9 negatives (min 10) — specificity is not estimable.
- **65+** — only 6 rows (min 20); only 3 positives (min 10) — recall is not estimable; only 3 negatives (min 10) — specificity is not estimable.
- **<45** — only 13 rows (min 20); only 3 positives (min 10) — recall is not estimable.

Disparity assessment:

- inconclusive — recall: 45-54 1.0000 [0.5101, 1.0000] vs <45 0.6667 [0.2077, 0.9385], gap 0.3333. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — specificity: 45-54 1.0000 [0.7412, 1.0000] vs 55-64 0.7778 [0.4526, 0.9368], gap 0.2222. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — precision: 45-54 1.0000 [0.5101, 1.0000] vs 55-64 0.8750 [0.6398, 0.9650], gap 0.1250. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.
- inconclusive — roc_auc: 45-54 1.0000 [1.0000, 1.0000] vs 55-64 0.8889 [0.7658, 1.0000], gap 0.1111. At least one of these subgroups is below the minimum class counts, so this gap is NOT evidence of a disparity — and the absence of a gap would not be evidence of equity either. The test set is too small to support any claim about these groups.

ROC-AUC is identical across the two threshold blocks above — it does not depend on the cut-off. Only the rate metrics move, and the size of that movement is a direct measure of how much a threshold-based subgroup comparison reflects the threshold rather than the model.

One caveat specific to age: age is itself one of the model's strongest features, so computing ROC-AUC *within* an age band removes most of the variance in that feature. Within-band AUC is therefore not directly comparable to the overall test AUC. Comparing bands against each other remains meaningful, but the bands span different widths of the underlying age range, so some of the difference between them reflects that rather than model behaviour.

## Calibration

Wrapper applied: `sigmoid`

| | Brier | ECE | ROC-AUC |
|---|---|---|---|
| before (raw) | 0.100300 | 0.123580 | 0.9502 |
| after | 0.089331 | 0.076520 | 0.9535 |
| delta | -0.010969 | -0.047059 | +0.0032 |

ECE is the population-weighted mean gap between predicted probability and observed frequency across 10 bins. It isolates calibration, where Brier mixes calibration with discrimination — but ECE alone is blind to ranking, so a model predicting the base rate for everyone would score near-perfectly. Read the two together.

_Brier and ECE are computed on the held-out test set for reporting. The calibration method itself was chosen on training out-of-fold Brier alone; these figures played no part in that choice._

See `figures/calibration_before_after.png` for the reliability curves.

---
*Generated by `scripts/fairness_report.py`. Every number above comes from that run.*
