"""Phase 2 tests: the shared preprocessor is leakage-safe and well-formed.

These fit *transformers* only (imputers / scaler / one-hot) on small slices — no
classifier is trained and no performance metric is produced.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.data.loaders import LOADERS
from ml.preprocessing.build_preprocessor import build_preprocessor, output_feature_names
from ml.training.splits import make_split


@pytest.mark.parametrize("disease", ["heart", "kidney", "diabetes"])
def test_preprocessor_is_unfitted_and_typed(disease, load_dataset):
    _, _, spec = load_dataset(disease)
    pre = build_preprocessor(spec)
    # Unfitted: no learned attributes yet.
    assert not hasattr(pre, "transformers_")
    declared = {c for _, _, cols in pre.transformers for c in cols}
    assert declared == set(spec.features)


@pytest.mark.parametrize("disease", ["heart", "kidney", "diabetes"])
def test_fit_transform_has_no_missing_and_is_finite(disease, load_dataset):
    X, y, spec = load_dataset(disease)
    split = make_split(disease, X, y)
    X_train, X_test, _, _ = split.apply(X, y)

    pre = build_preprocessor(spec)
    Xt_train = pre.fit_transform(X_train)
    Xt_test = pre.transform(X_test)

    assert np.isfinite(Xt_train).all()
    assert np.isfinite(Xt_test).all()
    assert Xt_train.shape[0] == len(X_train)
    assert Xt_test.shape[0] == len(X_test)
    # Feature names are exposed for SHAP / model cards.
    names = output_feature_names(pre)
    assert len(names) == Xt_train.shape[1]


def test_numeric_imputer_uses_train_statistics_only(load_dataset):
    """The median used to fill NaNs must come from the training rows, not the whole set."""
    X, y, spec = load_dataset("kidney")  # kidney has heavy missingness
    split = make_split("kidney", X, y)
    X_train, _, _, _ = split.apply(X, y)

    col = "hemo"  # a numeric column known to contain NaNs
    assert X[col].isna().any()

    pre = build_preprocessor(spec)
    pre.fit(X_train)

    numeric_pipe: Pipeline = dict(pre.named_transformers_)["numeric"]
    imputer: SimpleImputer = numeric_pipe.named_steps["impute"]
    learned = imputer.statistics_[list(spec.numeric).index(col)]

    train_median = float(X_train[col].median())
    full_median = float(X[col].median())
    assert learned == pytest.approx(train_median)
    # Guard the test itself: train vs full medians must actually differ for kidney/hemo,
    # otherwise this assertion proves nothing.
    assert train_median != full_median


def test_onehot_handles_unseen_categories(load_dataset):
    """A category present only in test must not break transform (handle_unknown='ignore')."""
    X, y, spec = load_dataset("heart")
    split = make_split("heart", X, y)
    X_train, X_test, _, _ = split.apply(X, y)
    pre = build_preprocessor(spec)
    pre.fit(X_train)
    # Inject an unseen category into a categorical column of the test frame.
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "thal"] = 999
    out = pre.transform(X_test)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("disease", list(LOADERS))
def test_all_diseases_build(disease, load_dataset):
    _, _, spec = load_dataset(disease)
    assert build_preprocessor(spec) is not None
