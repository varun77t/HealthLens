"""EDA plotting helpers. Each function saves a PNG and returns its path.

Plots are intentionally plain (matplotlib defaults + a light seaborn theme) so they
read cleanly in both the notebooks and the analytics API later.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: notebooks and CI both work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.data.feature_spec import FeatureSpec

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distribution(y: pd.Series, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = y.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(y):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Target distribution — {y.name}")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.margins(y=0.15)
    return _save(fig, out_dir, "target_distribution.png")


def plot_missingness(X: pd.DataFrame, out_dir: Path) -> Path:
    miss = (X.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(miss) + 1)))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.barh(miss.index[::-1], miss.values[::-1], color="#8172B2")
        ax.set_xlabel("% missing")
        ax.set_title("Missing values by column")
    return _save(fig, out_dir, "missingness.png")


def plot_numeric_histograms(X: pd.DataFrame, spec: FeatureSpec, out_dir: Path) -> Path:
    cols = spec.numeric
    n = len(cols)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, col in zip(axes, cols):
        X[col].dropna().hist(ax=ax, bins=30, color="#4C72B0")
        ax.set_title(col, fontsize=10)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Numeric feature distributions", y=1.02)
    return _save(fig, out_dir, "numeric_histograms.png")


def plot_numeric_boxplots_by_target(X: pd.DataFrame, y: pd.Series, spec: FeatureSpec, out_dir: Path) -> Path:
    cols = spec.numeric
    n = len(cols)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    df = X.copy()
    df["__target__"] = y.values
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x="__target__", y=col, ax=ax, hue="__target__", legend=False)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Numeric features by target class", y=1.02)
    return _save(fig, out_dir, "numeric_boxplots_by_target.png")


def plot_correlation_heatmap(X: pd.DataFrame, spec: FeatureSpec, out_dir: Path, method: str = "spearman") -> Path:
    cols = spec.numeric
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2, 0.9 * len(cols) + 1))
    corr = X[cols].corr(method=method)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title(f"{method.title()} correlation — numeric features")
    return _save(fig, out_dir, "correlation_heatmap.png")


def plot_categorical_rates(X: pd.DataFrame, y: pd.Series, spec: FeatureSpec, out_dir: Path) -> Path:
    cols = [*spec.categorical, *spec.binary]
    if not cols:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No categorical/binary features", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_dir, "categorical_positive_rate.png")
    n = len(cols)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    df = X.copy()
    df["__target__"] = y.values
    for ax, col in zip(axes, cols):
        rate = df.groupby(col)["__target__"].mean()
        ax.bar(rate.index.astype(str), rate.values, color="#55A868")
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("P(target=1)")
        ax.set_ylim(0, 1)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Positive-class rate by category", y=1.02)
    return _save(fig, out_dir, "categorical_positive_rate.png")


def all_eda_plots(X: pd.DataFrame, y: pd.Series, spec: FeatureSpec, out_dir: Path) -> list[Path]:
    return [
        plot_target_distribution(y, out_dir),
        plot_missingness(X, out_dir),
        plot_numeric_histograms(X, spec, out_dir),
        plot_numeric_boxplots_by_target(X, y, spec, out_dir),
        plot_correlation_heatmap(X, spec, out_dir),
        plot_categorical_rates(X, y, spec, out_dir),
    ]
