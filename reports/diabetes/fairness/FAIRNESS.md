# Subgroup performance — Diabetes Health-Indicator Risk Prediction

> **For research and educational purposes only.** This is a description of how one model behaved on one small held-out sample. It is not an audit, and it does not establish that the model is fair or safe to use.

Held-out test set: **n = 50961**. Every figure below comes from the shipped pipeline's predictions on those rows; no model was retrained.

## How to read this

A subgroup is marked **reliable** only if it holds at least 20 rows with at least 10 positives and 10 negatives. Rates carry **Wilson 95% intervals**; ROC-AUC carries a Hanley–McNeil interval.

A gap is called a **material difference** only when it clears two separate bars: the intervals must be disjoint (so it is not sampling noise), **and** the gap must be at least 0.05. The second bar matters as much as the first — with 50,961 test rows the intervals become narrow enough that almost any difference separates, so statistical clarity stops implying importance. Small-sample noise and large-sample over-sensitivity are opposite failures and both are named here.

Equal metrics across groups would **not** demonstrate fairness. Only sex and age are available here, the labels carry their own measurement bias, and parity on two attributes says nothing about who a model harms.

**6 of 6 subgroups meet the minimum counts.**

## By sex

### At threshold 0.5

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| female | 28819 | 3731 | 25088 | 0.1295 | 0.1471 | 0.1361 | 0.1589 | 0.9841 | 0.9825 | 0.9856 | 0.8419 | 0.8338 | 0.8499 | True |
| male | 22142 | 3316 | 18826 | 0.1498 | 0.1457 | 0.1341 | 0.1581 | 0.9814 | 0.9793 | 0.9832 | 0.8179 | 0.8088 | 0.8270 | True |

Disparity assessment:

- inconclusive — recall: female 0.1471 [0.1361, 0.1589] vs male 0.1457 [0.1341, 0.1581], gap 0.0014. The confidence intervals overlap, so the data is consistent with no difference between these groups. Do not report this as a disparity. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- inconclusive — specificity: female 0.9841 [0.9825, 0.9856] vs male 0.9814 [0.9793, 0.9832], gap 0.0027. The confidence intervals overlap, so the data is consistent with no difference between these groups. Do not report this as a disparity. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- inconclusive — precision: female 0.5791 [0.5474, 0.6102] vs female 0.5791 [0.5474, 0.6102], gap 0.0000. The confidence intervals overlap, so the data is consistent with no difference between these groups. Do not report this as a disparity. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- small but statistically clear — roc_auc: female 0.8419 [0.8338, 0.8499] vs male 0.8179 [0.8088, 0.8270], gap 0.0240. The intervals are disjoint, so the difference is statistically clear — but it is smaller than 0.05, and at this sample size confidence intervals are narrow enough that almost any difference separates. Report it as a measured but small difference, not as a disparity worth acting on.

### At sensitivity-oriented threshold 0.1388

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| female | 28819 | 3731 | 25088 | 0.1295 | 0.7899 | 0.7765 | 0.8026 | 0.7384 | 0.7329 | 0.7438 | 0.8419 | 0.8338 | 0.8499 | True |
| male | 22142 | 3316 | 18826 | 0.1498 | 0.8079 | 0.7941 | 0.8210 | 0.6697 | 0.6630 | 0.6764 | 0.8179 | 0.8088 | 0.8270 | True |

Disparity assessment:

- inconclusive — recall: male 0.8079 [0.7941, 0.8210] vs female 0.7899 [0.7765, 0.8026], gap 0.0180. The confidence intervals overlap, so the data is consistent with no difference between these groups. Do not report this as a disparity. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- **MATERIAL DIFFERENCE** — specificity: female 0.7384 [0.7329, 0.7438] vs male 0.6697 [0.6630, 0.6764], gap 0.0687. The intervals are disjoint and the gap exceeds the practical threshold of 0.05, so this is a real and material difference in how the model performs across these groups. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- inconclusive — precision: female 0.3099 [0.3007, 0.3193] vs male 0.3011 [0.2917, 0.3107], gap 0.0088. The confidence intervals overlap, so the data is consistent with no difference between these groups. Do not report this as a disparity. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- small but statistically clear — roc_auc: female 0.8419 [0.8338, 0.8499] vs male 0.8179 [0.8088, 0.8270], gap 0.0240. The intervals are disjoint, so the difference is statistically clear — but it is smaller than 0.05, and at this sample size confidence intervals are narrow enough that almost any difference separates. Report it as a measured but small difference, not as a disparity worth acting on.

ROC-AUC is identical across the two threshold blocks above — it does not depend on the cut-off. Only the rate metrics move, and the size of that movement is a direct measure of how much a threshold-based subgroup comparison reflects the threshold rather than the model.

## By age_band

### At threshold 0.5

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 18-34 | 4848 | 119 | 4729 | 0.0245 | 0.0084 | 0.0015 | 0.0461 | 0.9998 | 0.9988 | 1.0000 | 0.8583 | 0.8156 | 0.9010 | True |
| 35-49 | 10111 | 645 | 9466 | 0.0638 | 0.0791 | 0.0606 | 0.1025 | 0.9943 | 0.9926 | 0.9956 | 0.8535 | 0.8348 | 0.8722 | True |
| 50-64 | 18110 | 2624 | 15486 | 0.1449 | 0.1669 | 0.1531 | 0.1817 | 0.9808 | 0.9785 | 0.9829 | 0.8241 | 0.8140 | 0.8341 | True |
| 65+ | 17892 | 3659 | 14233 | 0.2045 | 0.1481 | 0.1370 | 0.1600 | 0.9720 | 0.9692 | 0.9746 | 0.7683 | 0.7588 | 0.7778 | True |

Disparity assessment:

- **MATERIAL DIFFERENCE** — recall: 50-64 0.1669 [0.1531, 0.1817] vs 18-34 0.0084 [0.0015, 0.0461], gap 0.1585. The intervals are disjoint and the gap exceeds the practical threshold of 0.05, so this is a real and material difference in how the model performs across these groups. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- small but statistically clear — specificity: 18-34 0.9998 [0.9988, 1.0000] vs 65+ 0.9720 [0.9692, 0.9746], gap 0.0278. The intervals are disjoint, so the difference is statistically clear — but it is smaller than 0.05, and at this sample size confidence intervals are narrow enough that almost any difference separates. Report it as a measured but small difference, not as a disparity worth acting on. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- inconclusive — precision: 50-64 0.5959 [0.5600, 0.6308] vs 35-49 0.4857 [0.3923, 0.5801], gap 0.1102. The confidence intervals overlap, so the data is consistent with no difference between these groups. Do not report this as a disparity. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- **MATERIAL DIFFERENCE** — roc_auc: 18-34 0.8583 [0.8156, 0.9010] vs 65+ 0.7683 [0.7588, 0.7778], gap 0.0900. The intervals are disjoint and the gap exceeds the practical threshold of 0.05, so this is a real and material difference in how the model performs across these groups.

### At sensitivity-oriented threshold 0.1388

| subgroup | n | n_positive | n_negative | prevalence | recall | recall_lo | recall_hi | specificity | specificity_lo | specificity_hi | roc_auc | roc_auc_lo | roc_auc_hi | reliable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 18-34 | 4848 | 119 | 4729 | 0.0245 | 0.2689 | 0.1974 | 0.3549 | 0.9852 | 0.9813 | 0.9883 | 0.8583 | 0.8156 | 0.9010 | True |
| 35-49 | 10111 | 645 | 9466 | 0.0638 | 0.6109 | 0.5727 | 0.6477 | 0.8806 | 0.8739 | 0.8870 | 0.8535 | 0.8348 | 0.8722 | True |
| 50-64 | 18110 | 2624 | 15486 | 0.1449 | 0.7820 | 0.7658 | 0.7974 | 0.7099 | 0.7027 | 0.7170 | 0.8241 | 0.8140 | 0.8341 | True |
| 65+ | 17892 | 3659 | 14233 | 0.2045 | 0.8603 | 0.8487 | 0.8712 | 0.5019 | 0.4937 | 0.5101 | 0.7683 | 0.7588 | 0.7778 | True |

Disparity assessment:

- **MATERIAL DIFFERENCE** — recall: 65+ 0.8603 [0.8487, 0.8712] vs 18-34 0.2689 [0.1974, 0.3549], gap 0.5914. The intervals are disjoint and the gap exceeds the practical threshold of 0.05, so this is a real and material difference in how the model performs across these groups. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- **MATERIAL DIFFERENCE** — specificity: 18-34 0.9852 [0.9813, 0.9883] vs 65+ 0.5019 [0.4937, 0.5101], gap 0.4833. The intervals are disjoint and the gap exceeds the practical threshold of 0.05, so this is a real and material difference in how the model performs across these groups. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- inconclusive — precision: 18-34 0.3137 [0.2318, 0.4091] vs 35-49 0.2585 [0.2372, 0.2811], gap 0.0552. The confidence intervals overlap, so the data is consistent with no difference between these groups. Do not report this as a disparity. Note that this metric is computed at a fixed decision threshold, so part of the gap reflects where that cut-off sits relative to each subgroup's score distribution rather than how well the model ranks people within the group; the threshold-free ROC-AUC comparison is the better guide to ranking quality.
- **MATERIAL DIFFERENCE** — roc_auc: 18-34 0.8583 [0.8156, 0.9010] vs 65+ 0.7683 [0.7588, 0.7778], gap 0.0900. The intervals are disjoint and the gap exceeds the practical threshold of 0.05, so this is a real and material difference in how the model performs across these groups.

ROC-AUC is identical across the two threshold blocks above — it does not depend on the cut-off. Only the rate metrics move, and the size of that movement is a direct measure of how much a threshold-based subgroup comparison reflects the threshold rather than the model.

One caveat specific to age: age is itself one of the model's strongest features, so computing ROC-AUC *within* an age band removes most of the variance in that feature. Within-band AUC is therefore not directly comparable to the overall test AUC. Comparing bands against each other remains meaningful, but the bands span different widths of the underlying age range, so some of the difference between them reflects that rather than model behaviour.

## Calibration

Wrapper applied: `isotonic`

| | Brier | ECE | ROC-AUC |
|---|---|---|---|
| before (raw) | 0.173580 | 0.235611 | 0.8319 |
| after | 0.095927 | 0.002000 | 0.8317 |
| delta | -0.077652 | -0.233611 | -0.0002 |

ECE is the population-weighted mean gap between predicted probability and observed frequency across 10 bins. It isolates calibration, where Brier mixes calibration with discrimination — but ECE alone is blind to ranking, so a model predicting the base rate for everyone would score near-perfectly. Read the two together.

_Brier and ECE are computed on the held-out test set for reporting. The calibration method itself was chosen on training out-of-fold Brier alone; these figures played no part in that choice._

See `figures/calibration_before_after.png` for the reliability curves.

---
*Generated by `scripts/fairness_report.py`. Every number above comes from that run.*
