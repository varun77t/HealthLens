"""End-to-end training run for one disease: experiment -> evaluate -> calibrate -> explain -> ship.

``train_disease(disease)`` is what ``scripts/train_<disease>.py`` calls. It produces
every artifact for that disease and returns a summary dict. All numbers written come
from this run; nothing is defaulted or filled in by hand.

Artifacts
---------
``reports/<disease>/``
    ``model_comparison.csv``     cross-validated comparison of the whole model zoo
    ``SELECTION.md``             which model was chosen and on what evidence
    ``metrics.json``             test-set metrics (0.5 and sensitivity-oriented threshold)
    ``threshold_sweep.csv``      precision/recall/specificity across thresholds
    ``calibration/``             calibration selection (train CV) and test-set comparison
    ``shap_global.json``         mean |SHAP| per original feature
    ``missingness_vs_target.csv`` is a column's missingness itself predictive?
    ``figures/``                 ROC, PR, confusion, calibration, SHAP plots
``models/<disease>/``
    ``pipeline.joblib``          the serving artifact (calibrated if calibration was chosen)
    ``base_pipeline.joblib``     the uncalibrated pipeline the SHAP explainer is built on
    ``metadata.json`` / ``MODEL_CARD.md``

Why two pipelines: ``CalibratedClassifierCV`` wraps and refits the base estimator, which
hides the tree/linear structure SHAP needs. The calibrated wrapper is what produces
probabilities; the base pipeline is what explains them. Platt/isotonic calibration is a
monotonic transform of the base model's score, so the sign and ranking of SHAP
contributions carry over — that caveat is recorded in the model card.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from config import DISEASES, MODELS_DIR, REPORTS_DIR
from ml.data.loaders import load
from ml.eda.profile import missingness_target_association
from ml.evaluation import calibration as calib
from ml.evaluation.curves import all_evaluation_plots
from ml.evaluation.diagnostics import (
    attribution_note,
    complete_case_probe,
    missingness_only_probe,
)
from ml.evaluation.metrics import classification_metrics, threshold_sweep, youden_threshold
from ml.explainability.shap_explainer import (
    DiseaseExplainer,
    plot_beeswarm,
    plot_global_importance,
)
from ml.model_card import build_model_card, write_model_card
from ml.reporting import write_json
from ml.training.experiment import run_experiment
from ml.training.splits import cv_splitter, make_split


def _dataset_stats(X: pd.DataFrame, y: pd.Series, spec) -> dict:
    counts = y.value_counts().sort_index()
    return {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric": len(spec.numeric),
        "n_categorical": len(spec.categorical),
        "n_binary": len(spec.binary),
        "target": str(y.name),
        "class_0": int(counts.get(0, 0)),
        "class_1": int(counts.get(1, 0)),
        "positive_rate": round(float(y.mean()), 4),
        "total_missing_cells": int(X.isna().sum().sum()),
    }


def train_disease(
    disease: str,
    *,
    smote: bool = False,
    tune_top_k: int = 3,
    calibration_inner_cv: int = 3,
    shap_background: int = 200,
) -> dict:
    """Run the full pipeline for ``disease`` and write every artifact."""
    reports = REPORTS_DIR / disease
    figures = reports / "figures"
    models_dir = MODELS_DIR / disease
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, spec = load(disease)
    split = make_split(disease, X, y)
    X_train, X_test, y_train, y_test = split.apply(X, y)

    # 1. Model comparison + tuning + selection (writes model_comparison.csv, SELECTION.md).
    print(f"[{disease}] comparing and tuning models ...")
    exp = run_experiment(disease, smote=smote, tune_top_k=tune_top_k, write=True)
    base_pipeline = exp.fitted_pipeline
    selected = exp.selected_model
    if base_pipeline is None:  # pragma: no cover - tuning always yields an estimator
        raise RuntimeError(f"run_experiment('{disease}') returned no fitted pipeline")
    cv_row = exp.comparison.set_index("model").loc[selected].to_dict()
    print(f"[{disease}] selected: {selected}")

    # 2. Calibration: selected on TRAINING out-of-fold scores only.
    print(f"[{disease}] calibration selection (training CV) ...")
    cv_cal = calib.calibration_cv_scores(
        base_pipeline,
        X_train,
        y_train,
        cv=cv_splitter(disease),
        groups=split.groups,
        inner_cv=calibration_inner_cv,
    )
    choice = calib.pick_calibration(cv_cal)
    print(f"[{disease}] calibration: {choice.method}")

    final_pipeline = (
        base_pipeline
        if choice.method == "raw"
        else calib.fit_calibrated(base_pipeline, X_train, y_train, choice.method,
                                  inner_cv=calibration_inner_cv)
    )

    # Test-set comparison of all three options — reported, never used to choose.
    test_cal = calib.compare_calibration(
        base_pipeline, X_train, y_train, X_test, y_test, inner_cv=calibration_inner_cv
    )

    # 3. Sensitivity-oriented threshold, derived from TRAINING out-of-fold predictions.
    #    Picking it on the test set would make the numbers it produces optimistic, so
    #    the threshold is learned on train and only then applied to test.
    oof = cross_val_predict(
        clone(final_pipeline),
        X_train,
        y_train,
        cv=cv_splitter(disease),
        groups=split.groups,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    alt_t = youden_threshold(y_train, oof)

    # 4. Evaluate the final pipeline on the held-out test set.
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    metrics_050 = classification_metrics(y_test, y_prob, threshold=0.5)
    metrics_alt = classification_metrics(y_test, y_prob, threshold=alt_t)
    metrics_alt["threshold_rule"] = (
        "Youden's J (max sensitivity + specificity - 1) maximised on out-of-fold "
        "predictions over the TRAINING set, then applied unchanged to the test set. "
        "The test set played no part in choosing it."
    )
    sweep = threshold_sweep(y_test, y_prob)

    figure_paths = all_evaluation_plots(
        y_test, y_prob, figures, threshold=0.5, title=f"— {DISEASES[disease]['label']}"
    )

    # 4b. Why does it score this well? Attribute the performance before believing it.
    probe = missingness_only_probe(
        X_train, y_train, X_test, y_test, cv=cv_splitter(disease), groups=split.groups
    )
    miss_assoc = missingness_target_association(X, y)
    diagnostics = {
        "missingness_only_probe": probe,
        "complete_case": complete_case_probe(X, y),
        "attribution_note": attribution_note(probe, metrics_050["roc_auc"]),
    }
    if not miss_assoc.empty:
        miss_assoc.to_csv(reports / "missingness_vs_target.csv", index=False)
        diagnostics["missingness_vs_target_top"] = (
            miss_assoc.head(8).to_dict(orient="records")
        )
    print(f"[{disease}] {diagnostics['attribution_note']}")

    # 5. SHAP on the base (uncalibrated) pipeline.
    print(f"[{disease}] computing SHAP ...")
    explainer = DiseaseExplainer(base_pipeline, spec, X_train, max_background=shap_background)
    importance = explainer.global_importance(X_test)
    additivity = explainer.additivity_error(X_test.head(min(50, len(X_test))))
    shap_bar = plot_global_importance(importance, figures)
    shap_bee = plot_beeswarm(explainer, X_test, figures)
    example = explainer.explain_one(X_test.iloc[[0]])

    write_json(
        reports / "shap_global.json",
        {
            "disease": disease,
            "explainer": explainer.kind,
            "base_value": explainer.base_value,
            "additivity_max_error": additivity,
            "n_rows_explained": int(len(X_test)),
            "global_importance": importance.to_dict(orient="records"),
            "example_local_explanation": example.to_dict(),
        },
    )

    # 6. Persist artifacts.
    pipeline_path = models_dir / "pipeline.joblib"
    base_path = models_dir / "base_pipeline.joblib"
    joblib.dump(final_pipeline, pipeline_path)
    joblib.dump(base_pipeline, base_path)

    sweep.round(4).to_csv(reports / "threshold_sweep.csv", index=False)
    (reports / "calibration").mkdir(parents=True, exist_ok=True)
    cv_cal.round(6).to_csv(reports / "calibration" / "cv_selection_train.csv")
    test_cal.round(6).to_csv(reports / "calibration" / "test_comparison.csv")

    write_json(
        reports / "metrics.json",
        {
            "disease": disease,
            "model": selected,
            "calibration": choice.method,
            "split": split.meta,
            "selection_margin": exp.margin,
            "diagnostics": diagnostics,
            "cross_validation_training_set": cv_row,
            "test_set_threshold_0.5": metrics_050,
            "test_set_alternative_threshold": metrics_alt,
            "calibration_selection_train_cv": cv_cal.reset_index().to_dict(orient="records"),
            "calibration_test_comparison": test_cal.reset_index().to_dict(orient="records"),
        },
    )

    # 7. Model card.
    card = build_model_card(
        disease,
        selected_model=selected,
        hyperparameters=_clean_params(exp.tuned.get(selected, {}).get("best_params", {})),
        calibration_method=choice.method,
        calibration_rationale=choice.rationale,
        selection_rationale=exp.rationale,
        selection_margin=exp.margin,
        dataset_stats=_dataset_stats(X, y, spec),
        split_meta={"strategy": split.strategy, **split.meta},
        cv_summary=cv_row,
        test_metrics=metrics_050,
        alt_threshold_metrics=metrics_alt,
        calibration_table=test_cal,
        diagnostics=diagnostics,
        shap_top_features=importance.head(10).to_dict(orient="records"),
        artifacts={
            "pipeline": pipeline_path.relative_to(MODELS_DIR.parent).as_posix(),
            "base_pipeline": base_path.relative_to(MODELS_DIR.parent).as_posix(),
            "reports": reports.relative_to(MODELS_DIR.parent).as_posix(),
            "figures": [p.name for p in figure_paths + [shap_bar] if p],
        },
    )
    card_json, card_md = write_model_card(disease, card)

    print(f"[{disease}] wrote {pipeline_path} and {card_json}")
    return {
        "disease": disease,
        "selected_model": selected,
        "calibration": choice.method,
        "test_metrics": metrics_050,
        "alt_threshold_metrics": metrics_alt,
        "shap_additivity_max_error": additivity,
        "diagnostics": diagnostics,
        "artifacts": {
            "pipeline": pipeline_path,
            "base_pipeline": base_path,
            "metadata": card_json,
            "model_card": card_md,
            "reports": reports,
        },
    }


def _clean_params(params: dict) -> dict:
    """Make search results JSON-friendly (numpy scalars -> python)."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out
