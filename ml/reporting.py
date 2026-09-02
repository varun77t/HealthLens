"""Small shared helpers for writing report files.

Kept dependency-free on purpose: ``DataFrame.to_markdown`` would pull in ``tabulate``,
and the tables here are simple enough to render directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(v, floatfmt: str = "{:.4f}") -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else floatfmt.format(v)
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return str(v)


def df_to_markdown(df: pd.DataFrame, *, floatfmt: str = "{:.4f}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    frame = df.reset_index() if index else df
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], floatfmt) for c in frame.columns) + " |")
    return "\n".join(lines)


def dict_to_markdown(d: dict, *, key_header: str = "key", value_header: str = "value",
                     floatfmt: str = "{:.4f}") -> str:
    lines = [f"| {key_header} | {value_header} |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| `{k}` | {_fmt(v, floatfmt)} |")
    return "\n".join(lines)


def write_lines(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"{type(o)} is not JSON serialisable")


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path
