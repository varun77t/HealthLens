"""SHAP explanations for a fitted disease pipeline.

The models are ``imblearn`` pipelines whose first step is the shared
``ColumnTransformer``. SHAP has to run on the *transformed* matrix, but a user-facing
explanation must talk about the **original** columns (``cp``, ``thal``, ...), not the
one-hot expansions (``cp_1.0``, ``cp_2.0``, ...). :class:`DiseaseExplainer` therefore

    1. transforms X with the pipeline's fitted preprocessor,
    2. runs the explainer appropriate to the estimator
       (``TreeExplainer`` for forests/boosters, ``LinearExplainer`` for logistic
       regression, ``KernelExplainer`` as a model-agnostic fallback),
    3. **sums** the SHAP values of every transformed column back onto the original
       column it came from — summing is exact for one-hot groups, so additivity is
       preserved.

Every number produced here comes from the actual fitted model. Nothing is
approximated by hand or filled in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RANDOM_STATE
from ml.data.feature_spec import FeatureSpec


# ------------------------------------------------------------------------------------
# Mapping transformed columns back to their source column
# ------------------------------------------------------------------------------------


def feature_source_map(preprocessor) -> dict[str, str]:
    """``{transformed_column_name: original_column_name}`` for a fitted ColumnTransformer.

    Uses each transformer's own ``categories_`` rather than string prefixes, because
    prefix matching is ambiguous (``thal`` is a prefix of ``thalach``).
    """
    mapping: dict[str, str] = {}
    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop" or trans is None or not len(cols):
            continue
        out = list(trans.get_feature_names_out(cols))
        if len(out) == len(cols):
            # Pass-through style block (impute / scale keep one column per input).
            for o, c in zip(out, cols):
                mapping[str(o)] = str(c)
            continue
        # One-hot expansion: consume the output names per source column.
        encoder = trans.named_steps["onehot"] if hasattr(trans, "named_steps") else trans
        i = 0
        for c, cats in zip(cols, encoder.categories_):
            for _ in range(len(cats)):
                mapping[str(out[i])] = str(c)
                i += 1
        if i != len(out):  # pragma: no cover - guards an encoder config we do not use
            raise RuntimeError(
                f"Could not map {len(out)} outputs of transformer '{name}' back to sources"
            )
    return mapping


def _positive_class(values) -> np.ndarray:
    """Normalise the many shapes SHAP returns for a binary classifier to ``(n, f)``."""
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) == 2 else values[0])
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] == 2 else arr[:, :, 0]
    return arr


def _scalar_base(expected_value) -> float:
    arr = np.asarray(expected_value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr[1] if arr.size == 2 else arr.flat[0])


# ------------------------------------------------------------------------------------
# Explainer
# ------------------------------------------------------------------------------------


@dataclass
class LocalExplanation:
    base_value: float
    predicted_probability: float
    contributions: list[dict]  # [{feature, value, contribution, description}]

    def to_dict(self) -> dict:
        return {
            "base_value": self.base_value,
            "predicted_probability": self.predicted_probability,
            "contributions": self.contributions,
        }


class DiseaseExplainer:
    """SHAP wrapper around a fitted pipeline for one disease."""

    def __init__(
        self,
        pipeline,
        spec: FeatureSpec,
        X_background: pd.DataFrame,
        *,
        max_background: int = 200,
        random_state: int = RANDOM_STATE,
    ):
        import shap

        self.pipeline = pipeline
        self.spec = spec
        self.preprocess = pipeline.named_steps["preprocess"]
        self.clf = pipeline.named_steps["clf"]
        self.out_names = [str(c) for c in self.preprocess.get_feature_names_out()]
        self.source_map = feature_source_map(self.preprocess)
        self.original_features = list(spec.features)

        bg = X_background
        if len(bg) > max_background:
            bg = bg.sample(max_background, random_state=random_state)
        self._background_t = self._transform(bg)

        module = type(self.clf).__module__
        if module.startswith(("sklearn.ensemble", "xgboost", "lightgbm")):
            self.kind = "tree"
            self.explainer = shap.TreeExplainer(self.clf)
        elif module.startswith("sklearn.linear_model"):
            self.kind = "linear"
            self.explainer = shap.LinearExplainer(self.clf, self._background_t)
        else:
            self.kind = "kernel"
            self.explainer = shap.KernelExplainer(
                lambda a: self.clf.predict_proba(a)[:, 1],
                shap.kmeans(self._background_t, min(25, len(self._background_t))),
            )

    # -- internals ------------------------------------------------------------

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocess.transform(X[self.original_features]))

    def _shap_transformed(self, Xt: np.ndarray) -> np.ndarray:
        if self.kind == "kernel":
            return _positive_class(self.explainer.shap_values(Xt, silent=True))
        return _positive_class(self.explainer.shap_values(Xt))

    @property
    def base_value(self) -> float:
        return _scalar_base(self.explainer.expected_value)

    # -- public ---------------------------------------------------------------

    def shap_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values on the *transformed* columns, ``(n_rows, n_transformed)``."""
        vals = self._shap_transformed(self._transform(X))
        return pd.DataFrame(vals, columns=self.out_names, index=X.index)

    def aggregate_to_original(self, shap_df: pd.DataFrame) -> pd.DataFrame:
        """Sum transformed-column SHAP values onto their original source column."""
        sources = [self.source_map[c] for c in shap_df.columns]
        agg = shap_df.T.groupby(sources).sum().T
        return agg.reindex(columns=self.original_features, fill_value=0.0)

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per original feature, ranked (the global explanation)."""
        agg = self.aggregate_to_original(self.shap_matrix(X))
        out = pd.DataFrame(
            {
                "feature": agg.columns,
                "mean_abs_shap": agg.abs().mean().to_numpy(),
                "mean_shap": agg.mean().to_numpy(),
                "description": [self.spec.describe(c) for c in agg.columns],
            }
        )
        out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    def explain_one(self, x: pd.DataFrame | pd.Series) -> LocalExplanation:
        """Signed per-feature contributions for a single case, on original names."""
        row = x.to_frame().T if isinstance(x, pd.Series) else x
        if len(row) != 1:
            raise ValueError("explain_one expects exactly one row")
        row = row.astype(float)
        agg = self.aggregate_to_original(self.shap_matrix(row)).iloc[0]
        proba = float(self.pipeline.predict_proba(row[self.original_features])[:, 1][0])
        contributions = [
            {
                "feature": feat,
                "value": None if pd.isna(row.iloc[0][feat]) else float(row.iloc[0][feat]),
                "contribution": float(agg[feat]),
                "description": self.spec.describe(feat),
            }
            for feat in sorted(self.original_features, key=lambda f: -abs(agg[f]))
        ]
        return LocalExplanation(
            base_value=self.base_value,
            predicted_probability=proba,
            contributions=contributions,
        )

    def additivity_error(self, X: pd.DataFrame) -> float | None:
        """Max |base + sum(SHAP) - model raw output|, or ``None`` if not checkable.

        A sanity check that the explanation really reconstructs the model, rather than
        being a plausible-looking but unrelated set of numbers.
        """
        Xt = self._transform(X)
        shap_vals = self._shap_transformed(Xt)
        recon = self.base_value + shap_vals.sum(axis=1)
        module = type(self.clf).__module__
        try:
            if self.kind == "linear":
                target = self.clf.decision_function(Xt)
            elif module.startswith("xgboost"):
                target = self.clf.get_booster().inplace_predict(Xt, predict_type="margin")
            elif module.startswith("lightgbm"):
                target = self.clf.predict_proba(Xt, raw_score=True)
            else:  # sklearn ensembles and the kernel fallback both work in probability
                target = self.clf.predict_proba(Xt)[:, 1]
        except Exception:  # pragma: no cover - best effort only
            return None
        return float(np.max(np.abs(recon - np.asarray(target).ravel())))


# ------------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------------


def plot_global_importance(importance: pd.DataFrame, out_dir: Path, *, top_n: int = 20) -> Path:
    """Horizontal bar chart of mean |SHAP| on the original feature names."""
    data = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.32 * len(data))))
    ax.barh(data["feature"], data["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP value|  (impact on model output)")
    ax.set_title("Global feature importance (SHAP)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shap_global_importance.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_beeswarm(explainer: DiseaseExplainer, X: pd.DataFrame, out_dir: Path) -> Path | None:
    """SHAP beeswarm on the transformed columns (distribution, not just magnitude)."""
    import shap

    try:
        vals = explainer.shap_matrix(X)
        fig = plt.figure()
        shap.summary_plot(
            vals.to_numpy(),
            explainer._transform(X),
            feature_names=explainer.out_names,
            show=False,
            plot_size=None,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "shap_beeswarm.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - the bar chart is the required artefact
        plt.close("all")
        return None
