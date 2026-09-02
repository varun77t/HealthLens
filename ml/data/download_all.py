"""Populate ``data/raw/`` with every dataset used by the platform.

Usage:
    python -m ml.data.download_all           # fetch anything not already cached
    python -m ml.data.download_all --force   # re-fetch everything from UCI
"""
from __future__ import annotations

import argparse
import sys

from ml.data.loaders import LOADERS
from ml.data.validate import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    exit_code = 0
    for name, loader in LOADERS.items():
        print(f"\n=== {name} ===")
        X, y, spec = loader(force=args.force)
        result = validate(name, X, y, spec)
        print(f"  rows x features : {result.n_rows} x {result.n_features}")
        print(f"  target          : {result.target_name}")
        print(f"  target counts   : {result.target_counts}  (positive rate {result.positive_rate:.4f})")
        missing = X.isna().sum()
        missing = missing[missing > 0]
        if len(missing):
            print(f"  columns with missing values : {missing.to_dict()}")
        else:
            print("  columns with missing values : none")
        if result.problems:
            exit_code = 1
            for p in result.problems:
                print(f"  !! VALIDATION PROBLEM: {p}")
        else:
            print("  validation      : OK")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
