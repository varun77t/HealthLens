"""Ensure the repository root is on sys.path for pytest and notebooks importing `ml`."""
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest

from ml.data.loaders import load as _load


@functools.lru_cache(maxsize=None)
def _cached_load(disease: str):
    """Load a dataset once per test session (the diabetes CSV is 253k rows)."""
    return _load(disease)


@pytest.fixture
def load_dataset():
    """Return a cached ``load(disease) -> (X, y, spec)`` callable.

    Tests must not mutate the returned frames in place; copy first if needed.
    """
    return _cached_load
