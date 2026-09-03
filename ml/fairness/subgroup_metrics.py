"""Per-subgroup performance, with the uncertainty attached.

A subgroup table is the easiest place in a project like this to manufacture a finding.
Split 61 patients by sex and one group has 7 positives; its recall is then a fraction with
a single-digit denominator, and the gap against the other group will look substantial
whatever the model does. Reporting that gap as a disparity — or worse, as evidence of
fairness because it happened to be small — is not analysis.

So every rate here carries a **Wilson score interval**, and every subgroup carries an
explicit ``reliable`` flag driven by its class counts. A disparity is only ever called
when the two groups' confidence intervals are disjoint; otherwise the verdict says the
data cannot distinguish them, which on the heart and kidney test sets is the usual answer.

Wilson rather than the normal approximation because the normal interval is badly wrong at
exactly the sample sizes that matter here — with 7 successes out of 7 it produces a
zero-width interval at 1.0, and with small counts it can run outside [0, 1] entirely.

ROC-AUC gets the Hanley–McNeil standard error, which is closed-form (so deterministic, no
bootstrap seed to record) and accounts for both class counts.

Nothing in this module changes a model. It reads a trained pipeline's predictions on the
held-out test set and describes them.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# A rate computed from fewer than this many cases in the relevant class is reported but
# flagged: recall over 7 positives is a number, not an estimate.
MIN_CLASS_N = 10
# Below this many rows a subgroup is not summarised at all beyond its counts.
MIN_SUBGROUP_N = 20
# A gap smaller than this is not worth acting on even when it is statistically clear.
# The large-sample mirror of the small-sample problem: with 50,961 test rows, confidence
# intervals shrink until almost any difference separates, and "statistically distinguishable"
# stops meaning "important". Both failure modes get named rather than only the noisy one.
PRACTICAL_GAP = 0.05

# Metrics computed at a fixed decision threshold. A gap in these partly reflects where the
# threshold happens to sit relative to each subgroup's score distribution, which is a
# property of the cut-off rather than of how well the model ranks people within the group.
THRESHOLD_DEPENDENT = {"recall", "specificity", "precision", "selection_rate"}

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi), NaN if n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    k = float(successes)
    n = float(n)
    denom = n + z * z
    centre = (k + z * z / 2.0) / denom
    half = (z / denom) * math.sqrt(k * (n - k) / n + z * z / 4.0)
    return (max(0.0, centre - half), min(1.0, centre + half))


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley–McNeil standard error of a ROC-AUC estimate."""
    if n_pos <= 0 or n_neg <= 0 or not np.isfinite(auc):
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    num = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    )
    return math.sqrt(max(num, 0.0) / (n_pos * n_neg))


# --- subgroup definitions -------------------------------------------------------------

# BRFSS codes its age question as 13 ordinal levels. Collapsing to four bands keeps each
# band large enough to estimate while staying interpretable; the mapping is exact, not
# approximate, because the underlying codes are themselves ranges.
_BRFSS_AGE_BANDS = {
    **{i: "18-34" for i in (1, 2, 3)},
    **{i: "35-49" for i in (4, 5, 6)},
    **{i: "50-64" for i in (7, 8, 9)},
    **{i: "65+" for i in (10, 11, 12, 13)},
}

_YEAR_BAND_EDGES = [(0, 45, "<45"), (45, 55, "45-54"), (55, 65, "55-64"), (65, 200, "65+")]

SEX_LABELS = {0: "female", 1: "male"}


def age_bands(age: pd.Series, disease: str) -> pd.Series:
    """Map an age column to readable bands.

    Diabetes stores a 13-level BRFSS survey code, not years; heart and kidney store years.
    Treating the survey code as a year value would silently produce nonsense bands.
    """
    if disease == "diabetes":
        return age.map(_BRFSS_AGE_BANDS).astype("object").rename("age_band")
    out = pd.Series(pd.NA, index=age.index, dtype="object", name="age_band")
    for lo, hi, label in _YEAR_BAND_EDGES:
        out[(age >= lo) & (age < hi)] = label
    return out


def sex_groups(sex: pd.Series) -> pd.Series:
    return sex.map(SEX_LABELS).astype("object").rename("sex")


def available_attributes(X: pd.DataFrame, disease: str) -> dict[str, pd.Series]:
    """Protected attributes present in this dataset, as label Series.

    Kidney records no sex variable, so no sex breakdown exists for it — that absence is
    reported rather than worked around.
    """
    out: dict[str, pd.Series] = {}
    sex_col = next((c for c in X.columns if c.lower() == "sex"), None)
    if sex_col is not None:
        out["sex"] = sex_groups(X[sex_col])
    age_col = next((c for c in X.columns if c.lower() == "age"), None)
    if age_col is not None:
        out["age_band"] = age_bands(X[age_col], disease)
    return out


# --- the table --------------------------------------------------------------------------


def subgroup_table(
    y_true, y_prob, groups: pd.Series, *, threshold: float = 0.5
) -> pd.DataFrame:
    """Per-subgroup metrics with Wilson intervals and an explicit reliability flag."""
    y_true = pd.Series(np.asarray(y_true).astype(int)).reset_index(drop=True)
    y_prob = pd.Series(np.asarray(y_prob, dtype=float)).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    y_pred = (y_prob >= threshold).astype(int)

    rows = []
    for name in sorted(groups.dropna().unique(), key=str):
        mask = groups == name
        yt, yp, pr = y_true[mask], y_pred[mask], y_prob[mask]
        n = int(mask.sum())
        n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())

        row: dict = {
            "subgroup": str(name),
            "n": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "prevalence": round(n_pos / n, 4) if n else float("nan"),
            "cm_tp": tp, "cm_fn": fn, "cm_tn": tn, "cm_fp": fp,
        }
        _add_rate(row, "recall", tp, n_pos)
        _add_rate(row, "specificity", tn, n_neg)
        _add_rate(row, "precision", tp, tp + fp)
        _add_rate(row, "selection_rate", int((yp == 1).sum()), n)

        if n_pos > 0 and n_neg > 0:
            auc = float(roc_auc_score(yt, pr))
            se = auc_standard_error(auc, n_pos, n_neg)
            row["roc_auc"] = round(auc, 4)
            row["roc_auc_lo"] = round(max(0.0, auc - Z95 * se), 4)
            row["roc_auc_hi"] = round(min(1.0, auc + Z95 * se), 4)
        else:
            row["roc_auc"] = row["roc_auc_lo"] = row["roc_auc_hi"] = float("nan")

        reasons = []
        if n < MIN_SUBGROUP_N:
            reasons.append(f"only {n} rows (min {MIN_SUBGROUP_N})")
        if n_pos < MIN_CLASS_N:
            reasons.append(f"only {n_pos} positives (min {MIN_CLASS_N}) — recall is not estimable")
        if n_neg < MIN_CLASS_N:
            reasons.append(f"only {n_neg} negatives (min {MIN_CLASS_N}) — specificity is not estimable")
        row["reliable"] = not reasons
        row["caveat"] = "; ".join(reasons)
        rows.append(row)

    return pd.DataFrame(rows)


def _add_rate(row: dict, name: str, successes: int, n: int) -> None:
    if n <= 0:
        row[name] = row[f"{name}_lo"] = row[f"{name}_hi"] = float("nan")
        return
    lo, hi = wilson_interval(successes, n)
    row[name] = round(successes / n, 4)
    row[f"{name}_lo"] = round(lo, 4)
    row[f"{name}_hi"] = round(hi, 4)


# --- disparity ----------------------------------------------------------------------------


def disparity_assessment(table: pd.DataFrame, metric: str = "recall") -> dict:
    """Compare the extreme subgroups on ``metric`` and say whether the gap is supportable.

    A gap is only called a disparity when the two Wilson intervals are **disjoint**. Two
    overlapping intervals mean the data is consistent with no difference at all, however
    large the point estimate gap looks.
    """
    usable = table.dropna(subset=[metric])
    result: dict = {"metric": metric, "n_subgroups": int(len(usable))}
    if len(usable) < 2:
        result.update({
            "conclusive": False,
            "verdict": f"Fewer than two subgroups have a defined {metric}; no comparison is possible.",
        })
        return result

    hi_row = usable.loc[usable[metric].idxmax()]
    lo_row = usable.loc[usable[metric].idxmin()]
    gap = float(hi_row[metric] - lo_row[metric])
    disjoint = float(lo_row[f"{metric}_hi"]) < float(hi_row[f"{metric}_lo"])
    both_reliable = bool(hi_row["reliable"] and lo_row["reliable"])

    result.update({
        "best_subgroup": str(hi_row["subgroup"]),
        "best_value": float(hi_row[metric]),
        "best_ci": [float(hi_row[f"{metric}_lo"]), float(hi_row[f"{metric}_hi"])],
        "worst_subgroup": str(lo_row["subgroup"]),
        "worst_value": float(lo_row[metric]),
        "worst_ci": [float(lo_row[f"{metric}_lo"]), float(lo_row[f"{metric}_hi"])],
        "gap": round(gap, 4),
        "intervals_disjoint": bool(disjoint),
        "all_subgroups_reliable": both_reliable,
        "statistically_clear": bool(disjoint and both_reliable),
        "practically_large": bool(gap >= PRACTICAL_GAP),
        "threshold_dependent": metric in THRESHOLD_DEPENDENT,
        # A finding worth acting on has to clear both bars, not just the statistical one.
        "conclusive": bool(disjoint and both_reliable and gap >= PRACTICAL_GAP),
    })
    result["verdict"] = _disparity_verdict(result, metric)
    return result


def _disparity_verdict(r: dict, metric: str) -> str:
    head = (
        f"{metric}: {r['best_subgroup']} {r['best_value']:.4f} "
        f"[{r['best_ci'][0]:.4f}, {r['best_ci'][1]:.4f}] vs {r['worst_subgroup']} "
        f"{r['worst_value']:.4f} [{r['worst_ci'][0]:.4f}, {r['worst_ci'][1]:.4f}], "
        f"gap {r['gap']:.4f}."
    )
    caveat = (
        " Note that this metric is computed at a fixed decision threshold, so part of the "
        "gap reflects where that cut-off sits relative to each subgroup's score "
        "distribution rather than how well the model ranks people within the group; the "
        "threshold-free ROC-AUC comparison is the better guide to ranking quality."
        if r["threshold_dependent"] else ""
    )

    if not r["all_subgroups_reliable"]:
        return head + (
            " At least one of these subgroups is below the minimum class counts, so this "
            "gap is NOT evidence of a disparity — and the absence of a gap would not be "
            "evidence of equity either. The test set is too small to support any claim "
            "about these groups."
        )
    if not r["intervals_disjoint"]:
        return head + (
            " The confidence intervals overlap, so the data is consistent with no "
            "difference between these groups. Do not report this as a disparity."
        ) + caveat
    if not r["practically_large"]:
        return head + (
            f" The intervals are disjoint, so the difference is statistically clear — but "
            f"it is smaller than {PRACTICAL_GAP}, and at this sample size confidence "
            "intervals are narrow enough that almost any difference separates. Report it "
            "as a measured but small difference, not as a disparity worth acting on."
        ) + caveat
    return head + (
        " The intervals are disjoint and the gap exceeds the practical threshold of "
        f"{PRACTICAL_GAP}, so this is a real and material difference in how the model "
        "performs across these groups."
    ) + caveat
