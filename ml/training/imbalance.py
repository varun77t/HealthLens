"""Does SMOTE beat class weighting? Measure it rather than assuming.

Both are ways of stopping a classifier from ignoring a rare positive class, and the
usual advice ("use SMOTE for imbalanced data") is asserted far more often than it is
checked. This module runs the same model, on the same folds, under both strategies and
reports the difference.

    class_weight  estimator compensates internally (``class_weight="balanced"``, or
                  ``scale_pos_weight`` for XGBoost, which has no ``class_weight``)
    smote         estimator uses default weights; SMOTE synthesises minority examples

SMOTE runs **inside** the cross-validation pipeline, so it resamples each fold's training
portion only and never the validation fold. Resampling before splitting would leak
synthetic neighbours of validation rows into training and inflate every score.

The comparison is deliberately run for one model family, not the whole zoo: the question
is whether the imbalance strategy changes the answer, and answering it for the selected
model is what actually informs the shipped pipeline.

Read the output with the threshold in mind. Both strategies shift the operating point,
so *recall at 0.5* moves a lot while the threshold-free metrics (ROC-AUC, PR-AUC) often
barely move. PR-AUC is the one to weigh here: on an imbalanced problem it is sensitive
to exactly the minority-class behaviour these methods are meant to improve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from config import CV_FOLDS, RANDOM_STATE
from ml.training.model_zoo import build_model, positive_class_weight
from ml.training.splits import cv_splitter

_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}

STRATEGIES = ("class_weight", "smote")


def compare_imbalance_strategies(
    disease: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    spec,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = CV_FOLDS,
) -> pd.DataFrame:
    """Cross-validate ``model_name`` under each imbalance strategy on the training set.

    Returns one row per strategy with mean/std of each metric across the folds. Nothing
    is fitted on the test set and nothing here changes the shipped model — this is
    evidence for the model card, produced on the training data only.
    """
    splitter = cv_splitter(disease, n_splits=n_splits)
    pos_weight = positive_class_weight(y_train)
    rows = []
    for strategy in STRATEGIES:
        pipe = build_model(
            model_name,
            spec,
            smote=(strategy == "smote"),
            random_state=RANDOM_STATE,
            pos_weight=pos_weight,
        )
        cv = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=groups,
            scoring=_SCORING,
            cv=splitter,
            n_jobs=-1,
            return_train_score=False,
        )
        row = {"strategy": strategy, "model": model_name,
               "fit_seconds_mean": float(np.mean(cv["fit_time"]))}
        for metric in _SCORING:
            vals = cv[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def imbalance_note(table: pd.DataFrame, *, chosen: str = "class_weight") -> str:
    """One sentence on what the comparison actually showed."""
    if table.empty or len(table) < 2:
        return "Imbalance strategies were not compared."
    t = table.set_index("strategy")
    cw, sm = t.loc["class_weight"], t.loc["smote"]
    d_pr = float(sm["pr_auc_mean"] - cw["pr_auc_mean"])
    d_roc = float(sm["roc_auc_mean"] - cw["roc_auc_mean"])
    d_rec = float(sm["recall_mean"] - cw["recall_mean"])
    noise = float(max(cw["pr_auc_std"], sm["pr_auc_std"]))

    verdict = (
        "within fold-to-fold noise, so there is no evidence it helps"
        if abs(d_pr) < noise
        else ("an improvement" if d_pr > 0 else "a degradation")
    )
    return (
        f"SMOTE vs class weighting, same model and same folds: PR-AUC {sm['pr_auc_mean']:.4f} "
        f"vs {cw['pr_auc_mean']:.4f} ({d_pr:+.4f}, fold std {noise:.4f}) — {verdict}. "
        f"ROC-AUC moved {d_roc:+.4f} and recall at threshold 0.5 moved {d_rec:+.4f}; recall "
        "moves most because both strategies shift the operating point rather than the "
        f"ranking. Shipped strategy: {chosen}."
    )
