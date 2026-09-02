"""End-to-end experiment orchestration for one disease.

``run_experiment(disease)`` is the entry point the per-disease training scripts
(Phases 3-5) will call:

    1. load ``(X, y, spec)``            (ml.data.loaders)
    2. hold-out split                  (ml.training.splits.make_split)
    3. cross-validated model comparison on the training set
                                       -> reports/<disease>/model_comparison.csv
    4. RandomizedSearchCV tuning of the top models
    5. documented model selection      -> reports/<disease>/SELECTION.md
    6. refit the chosen pipeline on the full training set and return it
       (serialisation + SHAP + model card happen in the per-disease phase)

**Phase 2 status:** this module is scaffolding. The functions are implemented but
nothing in the repo calls them yet — no model has been trained, no
``model_comparison.csv`` or ``SELECTION.md`` exists. Running this is Phase 3+.

Selection is never by accuracy alone: models are ranked on a blend of ROC-AUC,
PR-AUC and recall (see ``rank_models``), with the final call written up by a human
in ``SELECTION.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from config import CV_FOLDS, RANDOM_STATE, REPORTS_DIR
from ml.data.loaders import load
from ml.training.model_zoo import MODEL_NAMES, build_model
from ml.training.param_space import param_space, search_iter
from ml.training.splits import cv_splitter, make_split

_CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}

# Weights for the composite ranking score (screening-oriented: discrimination first,
# then the minority-class recall). Documented, not clinical.
_RANK_WEIGHTS = {"roc_auc": 0.4, "pr_auc": 0.4, "recall": 0.2}


@dataclass
class ExperimentResult:
    disease: str
    split_meta: dict
    comparison: pd.DataFrame
    tuned: dict = field(default_factory=dict)      # name -> {"best_params", "cv_roc_auc"}
    selected_model: str | None = None
    fitted_pipeline: object | None = None


def cross_validate_models(
    disease: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    groups: np.ndarray | None = None,
    smote: bool = False,
    names: list[str] | None = None,
    n_splits: int = CV_FOLDS,
) -> pd.DataFrame:
    """5-fold stratified CV on the training set for every model in the zoo."""
    from ml.data.loaders import load as _load

    _, _, spec = _load(disease)
    splitter = cv_splitter(disease, n_splits=n_splits)
    rows = []
    for name in names or MODEL_NAMES:
        pipe = build_model(name, spec, smote=smote, random_state=RANDOM_STATE)
        cv = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=groups,
            scoring=_CV_SCORING,
            cv=splitter,
            n_jobs=-1,
            return_train_score=False,
        )
        row = {"model": name, "smote": smote}
        for metric in _CV_SCORING:
            vals = cv[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def rank_models(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add a composite ``rank_score`` column (higher is better)."""
    out = comparison.copy()
    out["rank_score"] = sum(
        _RANK_WEIGHTS[m] * out[f"{m}_mean"] for m in _RANK_WEIGHTS
    )
    return out.sort_values("rank_score", ascending=False).reset_index(drop=True)


def tune_model(
    disease: str,
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    groups: np.ndarray | None = None,
    smote: bool = False,
    n_iter: int | None = None,
    n_splits: int = CV_FOLDS,
) -> RandomizedSearchCV:
    """RandomizedSearchCV (scoring = ROC-AUC) for one model."""
    _, _, spec = load(disease)
    pipe = build_model(name, spec, smote=smote, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_space(name, smote=smote),
        n_iter=n_iter or search_iter(name, disease),
        scoring="roc_auc",
        cv=cv_splitter(disease, n_splits=n_splits),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    fit_params = {"groups": groups} if groups is not None else {}
    search.fit(X_train, y_train, **fit_params)
    return search


def _write_comparison(disease: str, ranked: pd.DataFrame) -> Path:
    out_dir = REPORTS_DIR / disease
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_comparison.csv"
    ranked.round(4).to_csv(path, index=False)
    return path


def _write_selection_md(disease: str, ranked: pd.DataFrame, tuned: dict, selected: str) -> Path:
    out_dir = REPORTS_DIR / disease
    lines = [
        f"# Model selection — {disease}",
        "",
        "Selection criteria (in order): ROC-AUC and PR-AUC (discrimination), then "
        "recall/sensitivity for the positive class, then calibration. **Not** accuracy.",
        "",
        "## Cross-validated comparison (training set, "
        f"{CV_FOLDS}-fold stratified{' group' if disease == 'diabetes' else ''} CV)",
        "",
        ranked.round(4).to_markdown(index=False),
        "",
        "## Tuning",
        "",
    ]
    for name, info in tuned.items():
        lines.append(f"- **{name}**: CV ROC-AUC {info['cv_roc_auc']:.4f}; params `{info['best_params']}`")
    lines += [
        "",
        f"## Selected: `{selected}`",
        "",
        "_Rationale to be completed by a reviewer when this experiment is run "
        "(Phase 3+). This file is generated with the numbers above filled in from "
        "the actual run._",
        "",
        "---",
        "*Generated by `ml.training.experiment.run_experiment`.*",
    ]
    path = out_dir / "SELECTION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_experiment(
    disease: str,
    *,
    smote: bool = False,
    tune_top_k: int = 3,
    write: bool = True,
) -> ExperimentResult:
    """Run the full comparison -> tuning -> selection -> refit sequence.

    Not called anywhere in Phase 2. Invoked by the per-disease training scripts.
    """
    X, y, _ = load(disease)
    split = make_split(disease, X, y)
    X_train, X_test, y_train, y_test = split.apply(X, y)

    comparison = cross_validate_models(
        disease, X_train, y_train, groups=split.groups, smote=smote
    )
    ranked = rank_models(comparison)

    tuned: dict = {}
    for name in ranked["model"].head(tune_top_k):
        search = tune_model(disease, name, X_train, y_train, groups=split.groups, smote=smote)
        tuned[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": float(search.best_score_),
            "estimator": search.best_estimator_,
        }

    selected = max(tuned, key=lambda n: tuned[n]["cv_roc_auc"]) if tuned else ranked.loc[0, "model"]
    fitted = tuned[selected]["estimator"] if selected in tuned else None

    if write:
        _write_comparison(disease, ranked)
        _write_selection_md(
            disease,
            ranked,
            {k: {"best_params": v["best_params"], "cv_roc_auc": v["cv_roc_auc"]} for k, v in tuned.items()},
            selected,
        )

    return ExperimentResult(
        disease=disease,
        split_meta=split.meta,
        comparison=ranked,
        tuned={k: {"best_params": v["best_params"], "cv_roc_auc": v["cv_roc_auc"]} for k, v in tuned.items()},
        selected_model=selected,
        fitted_pipeline=fitted,
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(
        "ml.training.experiment is Phase 3+ scaffolding and is not run in Phase 2.\n"
        "No models are trained yet. See docs/PROJECT_STATUS.md."
        + (f"\n(ignored args: {sys.argv[1:]})" if sys.argv[1:] else "")
    )
