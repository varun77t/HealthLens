"""Train/test splitting and cross-validation splitters.

Two concerns are handled here so every disease pipeline splits data the same way:

1. **Hold-out split** (``make_split``) — a single train/test partition with
   ``test_size`` from :data:`config.TEST_SIZE` and ``random_state`` from
   :data:`config.RANDOM_STATE`.

2. **Diabetes duplicate rows.** The CDC BRFSS dataset (id 891) contains 25,772 rows
   whose full 21-feature vector is identical to an earlier row — legitimate repeated
   survey profiles, not corrupted data (see ``reports/diabetes/EDA.md``). A naive
   random split can put copies of the same feature vector in *both* train and test,
   letting a model score partly by memorisation. For diabetes we therefore group
   identical feature vectors (``groups`` = hash of the row) and keep every group
   wholly on one side, using :class:`~sklearn.model_selection.GroupShuffleSplit`
   for the hold-out and :class:`~sklearn.model_selection.GroupKFold` (shuffled) for
   cross-validation.

   *Why not* ``StratifiedGroupKFold``: the diabetes data has ~227,900 groups over
   253,680 rows (almost every row is its own group), and that estimator's greedy
   class-balancing loop is ~O(n_groups²) here — a single split takes ≈140 s. With
   groups this granular, random group assignment is already stratified in practice:
   measured train/test positive rates land within 0.002 of the 0.1393 overall rate
   (``make_split`` records the actual numbers in ``SplitResult.meta``).

Heart / kidney / statlog have zero duplicate rows and use a plain stratified split.

Nothing here fits a model or computes a performance metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)

from config import CV_FOLDS, RANDOM_STATE, TEST_SIZE

# Diseases whose identical feature vectors must not straddle the split.
_GROUP_AWARE = {"diabetes"}

# A group-aware split is only "close enough" to stratified when groups are granular.
# If the largest class-rate deviation across folds exceeds this, callers should know.
STRATIFICATION_TOLERANCE = 0.03


@dataclass
class SplitResult:
    disease: str
    strategy: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    groups: np.ndarray | None  # per-row group id aligned to the *train* rows, or None
    meta: dict = field(default_factory=dict)

    def apply(self, X: pd.DataFrame, y: pd.Series):
        """Return ``(X_train, X_test, y_train, y_test)`` positionally indexed."""
        return (
            X.iloc[self.train_idx].reset_index(drop=True),
            X.iloc[self.test_idx].reset_index(drop=True),
            y.iloc[self.train_idx].reset_index(drop=True),
            y.iloc[self.test_idx].reset_index(drop=True),
        )


def _feature_group_ids(X: pd.DataFrame) -> np.ndarray:
    """Integer id per row; rows with an identical feature vector share an id."""
    keys = pd.util.hash_pandas_object(X, index=False)
    return pd.factorize(keys)[0]


def make_split(
    disease: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """Build the hold-out train/test split for ``disease``."""
    n = len(X)
    y_arr = np.asarray(y)

    if disease in _GROUP_AWARE:
        group_ids = _feature_group_ids(X)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y_arr, groups=group_ids))
        train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
        train_groups = group_ids[train_idx]
        leaked = set(group_ids[test_idx]) & set(train_groups)
        meta = {
            "n_total": n,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_duplicate_rows": int(pd.Series(group_ids).duplicated().sum()),
            "n_unique_feature_vectors": int(len(np.unique(group_ids))),
            "groups_shared_across_split": len(leaked),
            "overall_positive_rate": round(float(y_arr.mean()), 4),
            "train_positive_rate": round(float(y_arr[train_idx].mean()), 4),
            "test_positive_rate": round(float(y_arr[test_idx].mean()), 4),
        }
        return SplitResult(
            disease=disease,
            strategy="group_shuffle_holdout",
            train_idx=train_idx,
            test_idx=test_idx,
            groups=train_groups,
            meta=meta,
        )

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
    meta = {
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_duplicate_rows": int(X.duplicated().sum()),
        "overall_positive_rate": round(float(y_arr.mean()), 4),
        "train_positive_rate": round(float(y_arr[train_idx].mean()), 4),
        "test_positive_rate": round(float(y_arr[test_idx].mean()), 4),
    }
    return SplitResult(
        disease=disease,
        strategy="stratified_holdout",
        train_idx=train_idx,
        test_idx=test_idx,
        groups=None,
        meta=meta,
    )


def cv_splitter(disease: str, n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE):
    """CV splitter for model comparison / tuning on the *training* set.

    * group-aware disease -> ``GroupKFold(shuffle=True)`` — pass
      ``groups=SplitResult.groups`` to ``.split`` / ``cross_validate``.
      (Not ``StratifiedGroupKFold``: too slow with near-singleton groups; folds
      come out stratified anyway — see the module docstring.)
    * otherwise -> ``StratifiedKFold(shuffle=True)``.
    """
    if disease in _GROUP_AWARE:
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def is_group_aware(disease: str) -> bool:
    return disease in _GROUP_AWARE
