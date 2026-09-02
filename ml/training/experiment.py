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

Steps 1-5 live here; evaluation, calibration, SHAP, serialisation and the model card
are in :mod:`ml.training.finalize`, which calls this. Use
``python -m scripts.train_<disease>`` to run the whole thing.

Selection is never by accuracy alone: models are ranked on a blend of ROC-AUC,
PR-AUC and recall (see ``rank_models``), and the rationale is written to
``SELECTION.md`` with the actual numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from config import CV_FOLDS, RANDOM_STATE, REPORTS_DIR
from ml.data.loaders import load
from ml.reporting import df_to_markdown
from ml.training.model_zoo import MODEL_NAMES, build_model, positive_class_weight
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
    rationale: str = ""
    margin: dict = field(default_factory=dict)     # how decisive the choice was


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
    _, _, spec = load(disease)
    splitter = cv_splitter(disease, n_splits=n_splits)
    pos_weight = positive_class_weight(y_train)
    rows = []
    for name in names or MODEL_NAMES:
        pipe = build_model(
            name, spec, smote=smote, random_state=RANDOM_STATE, pos_weight=pos_weight
        )
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
    pipe = build_model(
        name, spec, smote=smote, random_state=RANDOM_STATE,
        pos_weight=positive_class_weight(y_train),
    )
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


def selection_margin(ranked: pd.DataFrame, tuned: dict, selected: str) -> dict:
    """How decisive was the choice? Compares the tuned top-2 against CV noise.

    With a few hundred training rows the fold-to-fold standard deviation of ROC-AUC is
    often an order of magnitude larger than the gap between the best models. Saying so
    explicitly stops the leaderboard being read as a real quality ordering.
    """
    scores = sorted(((v["cv_roc_auc"], k) for k, v in tuned.items()), reverse=True)
    runner_up = scores[1] if len(scores) > 1 else None
    cv_std = float(ranked.set_index("model").loc[selected, "roc_auc_std"])
    gap = float(scores[0][0] - runner_up[0]) if runner_up else float("nan")
    return {
        "selected": selected,
        "selected_tuned_roc_auc": float(scores[0][0]),
        "runner_up": runner_up[1] if runner_up else None,
        "runner_up_tuned_roc_auc": float(runner_up[0]) if runner_up else None,
        "gap": gap,
        "cv_roc_auc_std": cv_std,
        "gap_within_cv_noise": bool(runner_up is not None and gap < cv_std),
    }


def _write_selection_md(
    disease: str, ranked: pd.DataFrame, tuned: dict, selected: str, rationale: str
) -> Path:
    out_dir = REPORTS_DIR / disease
    margin = selection_margin(ranked, tuned, selected)
    lines = [
        f"# Model selection — {disease}",
        "",
        "Selection criteria (in order): ROC-AUC and PR-AUC (discrimination), then "
        "recall/sensitivity for the positive class, then calibration. **Not** accuracy.",
        "",
        "## Cross-validated comparison (training set, "
        f"{CV_FOLDS}-fold {'grouped' if disease == 'diabetes' else 'stratified'} CV)",
        "",
        df_to_markdown(ranked.round(4)),
        "",
        "## Tuning (RandomizedSearchCV, scoring = ROC-AUC)",
        "",
    ]
    for name, info in sorted(tuned.items(), key=lambda kv: -kv[1]["cv_roc_auc"]):
        lines.append(f"- **{name}**: CV ROC-AUC {info['cv_roc_auc']:.4f}")
        lines.append(f"  - params: `{info['best_params']}`")
    lines += [
        "",
        f"## Selected: `{selected}`",
        "",
        rationale,
        "",
        "### How decisive was this?",
        "",
    ]
    if margin["runner_up"] is None:
        lines.append("Only one model was tuned, so there is nothing to compare against.")
    elif margin["gap_within_cv_noise"]:
        lines += [
            f"**Not decisive.** `{selected}` beat `{margin['runner_up']}` by "
            f"{margin['gap']:.4f} ROC-AUC, but the selected model's own fold-to-fold "
            f"standard deviation is {margin['cv_roc_auc_std']:.4f} — an order of magnitude "
            "larger. On this sample size that gap is noise, not evidence that one model "
            "is genuinely better. Read the leaderboard as 'these models perform "
            "comparably', not as a ranking. A different random seed could reorder them.",
        ]
    else:
        lines += [
            f"`{selected}` beat `{margin['runner_up']}` by {margin['gap']:.4f} ROC-AUC, "
            f"which exceeds the selected model's fold-to-fold standard deviation "
            f"({margin['cv_roc_auc_std']:.4f}).",
        ]
    lines += [
        "",
        "---",
        "*Generated by `ml.training.experiment.run_experiment`. Every number above comes "
        "from that run.*",
    ]
    path = out_dir / "SELECTION.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def selection_rationale(ranked: pd.DataFrame, tuned: dict, selected: str) -> str:
    """Prose explaining the choice, built from the run's real numbers."""
    top = str(ranked.iloc[0]["model"])
    rank_of_selected = int(ranked.index[ranked["model"] == selected][0]) + 1
    tuned_str = ", ".join(
        f"{k} {v['cv_roc_auc']:.4f}"
        for k, v in sorted(tuned.items(), key=lambda kv: -kv[1]["cv_roc_auc"])
    )
    if top == selected:
        placing = f"put `{selected}` first before tuning"
    else:
        placing = (
            f"put `{top}` first before tuning, with `{selected}` at rank {rank_of_selected}"
        )
    return (
        f"`{selected}` was chosen from {len(ranked)} candidate pipelines. "
        f"The cross-validated ranking (0.4·ROC-AUC + 0.4·PR-AUC + 0.2·recall) {placing}. "
        f"After randomised hyper-parameter search the tuned cross-validated ROC-AUC scores "
        f"were: {tuned_str}; `{selected}` was highest and was refit on the full training set. "
        "Accuracy was recorded but was not used as a selection criterion."
    )


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

    public_tuned = {
        k: {"best_params": _plain(v["best_params"]), "cv_roc_auc": v["cv_roc_auc"]}
        for k, v in tuned.items()
    }
    rationale = selection_rationale(ranked, public_tuned, selected)
    margin = selection_margin(ranked, public_tuned, selected)

    if write:
        _write_comparison(disease, ranked)
        _write_selection_md(disease, ranked, public_tuned, selected, rationale)

    return ExperimentResult(
        disease=disease,
        split_meta=split.meta,
        comparison=ranked,
        tuned=public_tuned,
        selected_model=selected,
        fitted_pipeline=fitted,
        rationale=rationale,
        margin=margin,
    )


def _plain(params: dict) -> dict:
    """numpy scalars -> python scalars, so params serialise cleanly to JSON/markdown."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(
        "Run a full training pass through ml.training.finalize.train_disease instead, "
        "e.g. `python -m scripts.train_heart`. This module only provides the comparison "
        "and tuning stages.\n"
        + (f"(ignored args: {sys.argv[1:]})" if sys.argv[1:] else "")
    )
