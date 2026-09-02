"""Candidate model pipelines for the model-comparison stage.

Every model is returned as an :class:`imblearn.pipeline.Pipeline`::

    [ ("preprocess", ColumnTransformer),
      ("smote", SMOTE)          # only when smote=True
      ("clf", <estimator>) ]

Using the imbalanced-learn pipeline means that when SMOTE is enabled it resamples
**inside each CV fold's training portion only** — never the validation fold, never
the held-out test set. That is the whole point of putting it in the pipeline rather
than calling ``fit_resample`` on the data up front.

Class imbalance can be addressed two ways and we keep them mutually exclusive:
    smote=False -> estimator uses ``class_weight="balanced"`` (or scale_pos_weight)
    smote=True  -> estimator uses default weights, SMOTE balances the classes

Nothing here is fitted. ``build_model`` just assembles the estimator object.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config import RANDOM_STATE
from ml.data.feature_spec import FeatureSpec
from ml.preprocessing.build_preprocessor import build_preprocessor

MODEL_NAMES: list[str] = ["logreg", "random_forest", "svc", "xgboost", "lightgbm"]


def _estimator(name: str, *, balanced: bool, random_state: int):
    """Instantiate a bare classifier.

    ``balanced=True`` asks the estimator to compensate for class imbalance itself
    (used when SMOTE is off).
    """
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if balanced else None,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced" if balanced else None,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "svc":
        return SVC(
            probability=True,
            class_weight="balanced" if balanced else None,
            random_state=random_state,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced" if balanced else None,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    raise KeyError(f"Unknown model '{name}'. Options: {MODEL_NAMES}")


def build_model(
    name: str,
    spec: FeatureSpec,
    *,
    smote: bool = False,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Assemble an (unfitted) preprocessing + optional-SMOTE + classifier pipeline."""
    if name not in MODEL_NAMES:
        raise KeyError(f"Unknown model '{name}'. Options: {MODEL_NAMES}")

    steps: list[tuple] = [("preprocess", build_preprocessor(spec))]
    if smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", _estimator(name, balanced=not smote, random_state=random_state)))
    return ImbPipeline(steps=steps)


def build_all(
    spec: FeatureSpec,
    *,
    smote: bool = False,
    random_state: int = RANDOM_STATE,
    names: list[str] | None = None,
) -> dict[str, ImbPipeline]:
    """``{name: pipeline}`` for the whole zoo (or the subset in ``names``)."""
    return {
        n: build_model(n, spec, smote=smote, random_state=random_state)
        for n in (names or MODEL_NAMES)
    }
