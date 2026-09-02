"""Leakage-safe preprocessing for each disease.

``build_preprocessor(spec)`` returns an **unfitted** :class:`~sklearn.compose.ColumnTransformer`
that must only ever be used inside a :class:`~sklearn.pipeline.Pipeline`. It is never
``fit`` on the full dataset — the enclosing pipeline fits it on the training folds only,
so imputation statistics and scaling parameters never see held-out data.

Column handling (roles come from the :class:`FeatureSpec`):
    numeric      -> SimpleImputer(median)         -> StandardScaler
    categorical  -> SimpleImputer(most_frequent)  -> OneHotEncoder(handle_unknown="ignore")
    binary       -> SimpleImputer(most_frequent)   (already 0/1, left unscaled)

Scaling the numeric block is harmless for the tree models (they are scale-invariant)
and necessary for LogisticRegression / SVC, so one shared preprocessor covers every
estimator in the zoo.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data.feature_spec import FeatureSpec


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def _categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def _binary_pipeline() -> Pipeline:
    # Binary indicators are already 0/1; only fill the (few) missing entries.
    return Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent"))])


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for ``spec``'s feature roles.

    Parameters
    ----------
    spec : FeatureSpec
        The disease feature specification (``spec.numeric`` / ``spec.categorical`` /
        ``spec.binary`` name the columns of the ``X`` returned by the loader).
    """
    transformers: list[tuple] = []
    if spec.numeric:
        transformers.append(("numeric", _numeric_pipeline(), list(spec.numeric)))
    if spec.categorical:
        transformers.append(("categorical", _categorical_pipeline(), list(spec.categorical)))
    if spec.binary:
        transformers.append(("binary", _binary_pipeline(), list(spec.binary)))

    if not transformers:  # pragma: no cover - every disease has features
        raise ValueError(f"FeatureSpec for '{spec.disease}' declares no feature columns")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def output_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Names of the columns a *fitted* preprocessor emits (for SHAP / model cards)."""
    return list(fitted_preprocessor.get_feature_names_out())
