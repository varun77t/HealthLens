"""Phase 2 tests: splitting is leakage-safe and model pipelines are well-formed.

No estimator is fitted here. ``build_model`` only *assembles* the pipeline object;
the tests check its structure and that it round-trips through joblib.
"""
from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

from ml.training.model_zoo import MODEL_NAMES, build_model
from ml.training.param_space import param_space, search_iter
from ml.training.splits import cv_splitter, make_split


# --- splitting ---------------------------------------------------------------

@pytest.mark.parametrize("disease", ["heart", "kidney", "diabetes"])
def test_split_is_disjoint_and_covers_all_rows(disease, load_dataset):
    X, y, _ = load_dataset(disease)
    split = make_split(disease, X, y)
    train, test = set(split.train_idx.tolist()), set(split.test_idx.tolist())
    assert not (train & test)
    assert train | test == set(range(len(X)))


@pytest.mark.parametrize("disease", ["heart", "kidney", "diabetes"])
def test_split_is_deterministic(disease, load_dataset):
    X, y, _ = load_dataset(disease)
    a = make_split(disease, X, y)
    b = make_split(disease, X, y)
    assert np.array_equal(a.test_idx, b.test_idx)


@pytest.mark.parametrize("disease", ["heart", "kidney", "diabetes"])
def test_split_roughly_stratified(disease, load_dataset):
    X, y, _ = load_dataset(disease)
    split = make_split(disease, X, y)
    assert split.meta["train_positive_rate"] == pytest.approx(
        split.meta["test_positive_rate"], abs=0.03
    )


def test_diabetes_no_identical_feature_vector_across_split(load_dataset):
    """The core reason diabetes gets a group-aware split: no shared feature rows."""
    X, y, _ = load_dataset("diabetes")
    split = make_split("diabetes", X, y)
    assert split.strategy == "group_shuffle_holdout"
    assert split.meta["groups_shared_across_split"] == 0

    import pandas as pd

    keys = pd.util.hash_pandas_object(X, index=False).to_numpy()
    assert not (set(keys[split.train_idx]) & set(keys[split.test_idx]))


def test_heart_uses_plain_stratified_split(load_dataset):
    X, y, _ = load_dataset("heart")
    split = make_split("heart", X, y)
    assert split.strategy == "stratified_holdout"
    assert split.groups is None


def test_cv_splitter_type_matches_disease():
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    assert isinstance(cv_splitter("diabetes"), GroupKFold)
    assert isinstance(cv_splitter("heart"), StratifiedKFold)


# --- model pipelines --------------------------------------------------------

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model_structure(name, load_dataset):
    _, _, spec = load_dataset("heart")
    pipe = build_model(name, spec)
    assert isinstance(pipe, ImbPipeline)
    assert isinstance(pipe.named_steps["preprocess"], ColumnTransformer)
    assert "clf" in pipe.named_steps
    assert "smote" not in pipe.named_steps  # off by default


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model_with_smote_has_resampler(name, load_dataset):
    _, _, spec = load_dataset("heart")
    pipe = build_model(name, spec, smote=True)
    assert "smote" in pipe.named_steps


def test_smote_and_class_weight_are_mutually_exclusive(load_dataset):
    _, _, spec = load_dataset("heart")
    weighted = build_model("random_forest", spec, smote=False).named_steps["clf"]
    smoted = build_model("random_forest", spec, smote=True).named_steps["clf"]
    assert weighted.class_weight == "balanced"
    assert smoted.class_weight is None


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_is_picklable(name, load_dataset):
    _, _, spec = load_dataset("heart")
    pipe = build_model(name, spec)
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    assert isinstance(restored, ImbPipeline)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_param_space_keys_target_pipeline_steps(name):
    space = param_space(name, smote=True)
    assert space  # non-empty
    for key in space:
        assert key.startswith("clf__") or key.startswith("smote__")
    assert search_iter(name, "diabetes") <= search_iter(name, "heart")
