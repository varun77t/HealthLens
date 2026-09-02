"""Is a candidate 'external' dataset actually independent of the training data?

External validation only means something if the model has not seen the rows. Public
medical datasets are frequently derived from one another — reprocessed, subsetted,
renamed and re-uploaded under a different name — and nothing in the file announces it.
Evaluating on such a dataset produces a number that looks like evidence of
generalisation and is not.

The failure is quiet, which is what makes it dangerous. A contaminated result does not
look broken: it lands near the internal test score, or slightly below it, which reads as
exactly the modest confirmation an honest external validation would give.

So the overlap is measured **before** any metric is computed, and a substantial overlap
blocks the result from being reported as validation.

Matching is exact on the full feature vector after numeric normalisation. Values are cast
to float and rounded before comparison, because the same row can arrive as ``int64`` in
one release and ``float64`` in another — a string comparison of "63" against "63.0" finds
no matches at all and silently reports a contaminated dataset as clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Above this fraction of matched rows, a dataset cannot be described as external.
# Set deliberately low: a handful of coincidentally identical rows is plausible in
# medical data, but a few per cent is not.
MAX_ACCEPTABLE_OVERLAP = 0.05


def row_keys(df: pd.DataFrame, columns: list[str] | None = None, *, decimals: int = 3) -> pd.Series:
    """Exact-match keys for each row, robust to int/float dtype differences.

    Rows containing NaN keep the literal 'nan' token, so a row with a missing value
    matches only another row missing the same value — which is the intended behaviour.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    numeric = df[cols].astype("float64").round(decimals)
    return numeric.apply(lambda r: "|".join(f"{v:.{decimals}f}" for v in r), axis=1)


@dataclass
class OverlapReport:
    n_candidate: int
    n_reference: int
    n_matched: int
    pct_matched: float
    n_label_agreements: int
    n_unseen: int
    pct_unseen: float
    n_in_train: int | None = None
    n_in_test: int | None = None
    pct_in_train: float | None = None
    independent: bool = True
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def dataset_overlap(
    X_candidate: pd.DataFrame,
    y_candidate: pd.Series,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    *,
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    columns: list[str] | None = None,
) -> OverlapReport:
    """Measure how much of ``X_candidate`` already appears in ``X_reference``.

    ``X_train`` / ``X_test`` are the reference dataset's split halves. Supplying them
    separates the two ways contamination hurts: rows in the training split were learned
    from, and rows in the test split are simply not new information.
    """
    cols = list(columns) if columns is not None else list(X_reference.columns)
    k_cand = row_keys(X_candidate, cols)
    k_ref = row_keys(X_reference, cols)

    ref_labels = dict(zip(k_ref, pd.Series(y_reference).to_numpy()))
    matched = k_cand.isin(set(k_ref))
    n_matched = int(matched.sum())
    n_cand = int(len(X_candidate))

    y_cand = pd.Series(y_candidate).to_numpy()
    agreements = sum(
        1
        for pos, (_, k) in enumerate(k_cand.items())
        if matched.iloc[pos] and ref_labels[k] == y_cand[pos]
    )

    report = OverlapReport(
        n_candidate=n_cand,
        n_reference=int(len(X_reference)),
        n_matched=n_matched,
        pct_matched=round(100 * n_matched / n_cand, 2) if n_cand else 0.0,
        n_label_agreements=int(agreements),
        n_unseen=n_cand - n_matched,
        pct_unseen=round(100 * (n_cand - n_matched) / n_cand, 2) if n_cand else 0.0,
    )

    if X_train is not None:
        in_train = k_cand.isin(set(row_keys(X_train, cols)))
        report.n_in_train = int(in_train.sum())
        report.pct_in_train = round(100 * float(in_train.mean()), 2)
    if X_test is not None:
        report.n_in_test = int(k_cand.isin(set(row_keys(X_test, cols))).sum())

    fraction = n_matched / n_cand if n_cand else 0.0
    report.independent = fraction <= MAX_ACCEPTABLE_OVERLAP
    report.reasons = _reasons(report, fraction)
    report.verdict = _verdict(report, fraction)
    return report


def _reasons(report: OverlapReport, fraction: float) -> list[str]:
    if report.independent:
        return []
    reasons = [
        f"{report.n_matched} of {report.n_candidate} rows ({report.pct_matched}%) match a "
        f"row in the training dataset exactly on every feature, against a maximum "
        f"acceptable overlap of {100 * MAX_ACCEPTABLE_OVERLAP:.0f}%"
    ]
    if report.n_matched and report.n_label_agreements == report.n_matched:
        reasons.append(
            f"all {report.n_matched} matched rows also carry the same label, so these are "
            "the same records rather than coincidentally similar patients"
        )
    if report.n_in_train:
        reasons.append(
            f"{report.n_in_train} rows ({report.pct_in_train}%) are in the model's own "
            "training split — the model was fitted on them"
        )
    if report.n_unseen == 0:
        reasons.append("no row in this dataset is unseen by the model")
    return reasons


def _verdict(report: OverlapReport, fraction: float) -> str:
    if report.independent:
        return (
            f"Independent: {report.pct_unseen}% of rows ({report.n_unseen}/"
            f"{report.n_candidate}) are unseen by the model. Usable as external validation."
        )
    if fraction >= 0.95:
        kind = "a subset of the training dataset, not an independent cohort"
    else:
        kind = "substantially overlapping with the training dataset"
    return (
        f"NOT INDEPENDENT — this dataset is {kind}. "
        + " ".join(r.capitalize() + "." for r in report.reasons)
        + " Any metric computed on it measures memorisation as much as generalisation and "
        "must not be published as external validation."
    )


def schema_compatibility(
    X_a: pd.DataFrame, X_b: pd.DataFrame, *, categorical: list[str]
) -> pd.DataFrame:
    """Compare the value domains of two frames column by column.

    Aligning a second dataset to a schema is only safe if the *encodings* agree, not just
    the column names: a 'thal' coded 3/6/7 in one release and 0/1/2 in another will align
    silently and predict nonsense. Returns one row per column with both value domains and
    whether they match.
    """
    rows = []
    for col in X_a.columns:
        if col not in X_b.columns:
            rows.append({"column": col, "in_both": False, "compatible": False,
                         "values_a": "", "values_b": "", "note": "missing from second dataset"})
            continue
        a, b = X_a[col].dropna(), X_b[col].dropna()
        if col in categorical:
            va, vb = sorted(a.unique().tolist()), sorted(b.unique().tolist())
            extra = sorted(set(vb) - set(va))
            rows.append({
                "column": col, "in_both": True,
                "values_a": ", ".join(f"{v:g}" for v in va),
                "values_b": ", ".join(f"{v:g}" for v in vb),
                "compatible": not extra,
                "note": "" if not extra
                        else f"values {extra} absent from the reference encoding",
            })
        else:
            overlaps = not (b.max() < a.min() or b.min() > a.max())
            rows.append({
                "column": col, "in_both": True,
                "values_a": f"[{a.min():g}, {a.max():g}] mean {a.mean():.2f}",
                "values_b": f"[{b.min():g}, {b.max():g}] mean {b.mean():.2f}",
                "compatible": bool(overlaps),
                "note": "" if overlaps else "ranges do not overlap — likely different units",
            })
    return pd.DataFrame(rows)
