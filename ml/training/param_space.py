"""Hyper-parameter search spaces for :class:`~sklearn.model_selection.RandomizedSearchCV`.

Keys are pipeline-step-prefixed (``clf__<param>``) so they apply to the estimator
inside the :func:`ml.training.model_zoo.build_model` pipeline. Distributions are
deliberately modest — the medical datasets are small (n = 303 / 400) and an
aggressive search would just overfit the cross-validation.

``param_space(name)`` returns a dict suitable for ``RandomizedSearchCV(param_distributions=...)``.
``search_iter(name, disease)`` suggests an ``n_iter`` (smaller for the large diabetes set).
"""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform

_SPACES: dict[str, dict] = {
    "logreg": {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear", "saga"],
    },
    "random_forest": {
        "clf__n_estimators": randint(200, 800),
        "clf__max_depth": [None, 4, 6, 8, 12, 20],
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2", 0.5, 0.8],
    },
    "svc": {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__kernel": ["rbf"],
    },
    "xgboost": {
        "clf__n_estimators": randint(200, 800),
        "clf__max_depth": randint(2, 8),
        "clf__learning_rate": loguniform(1e-2, 3e-1),
        "clf__subsample": uniform(0.6, 0.4),          # [0.6, 1.0]
        "clf__colsample_bytree": uniform(0.6, 0.4),   # [0.6, 1.0]
        "clf__min_child_weight": randint(1, 10),
        "clf__reg_lambda": loguniform(1e-2, 1e1),
    },
    "lightgbm": {
        "clf__n_estimators": randint(200, 800),
        "clf__num_leaves": randint(15, 128),
        "clf__max_depth": [-1, 4, 6, 8, 12],
        "clf__learning_rate": loguniform(1e-2, 3e-1),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_samples": randint(5, 60),
        "clf__reg_lambda": loguniform(1e-2, 1e1),
    },
}

# When SMOTE is in the pipeline, also vary its neighbourhood size.
_SMOTE_SPACE = {"smote__k_neighbors": randint(3, 8)}


def param_space(name: str, *, smote: bool = False) -> dict:
    if name not in _SPACES:
        raise KeyError(f"No param space for '{name}'. Options: {sorted(_SPACES)}")
    space = dict(_SPACES[name])
    if smote:
        space.update(_SMOTE_SPACE)
    return space


def search_iter(name: str, disease: str) -> int:
    """Suggested ``n_iter`` for RandomizedSearchCV."""
    if disease == "diabetes":
        # Large dataset: each fit is expensive, keep the search short.
        return 15 if name in {"xgboost", "lightgbm"} else 10
    return 30 if name in {"random_forest", "xgboost", "lightgbm"} else 20
