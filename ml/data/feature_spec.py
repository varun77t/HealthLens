"""Feature specification for a disease dataset.

A ``FeatureSpec`` records which columns are continuous, which are nominal
categoricals, and which are 0/1 binary indicators, plus the target column name and
a human-readable description for every feature. The modelling choice of where a
borderline column goes (e.g. an ordinal code treated as numeric) is documented in
the ``notes`` field and in each disease's ``reports/<disease>/TARGET.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    disease: str
    target: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    descriptions: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def features(self) -> list[str]:
        """All predictor columns, in numeric -> categorical -> binary order."""
        return [*self.numeric, *self.categorical, *self.binary]

    def describe(self, column: str) -> str:
        return self.descriptions.get(column, "(no description recorded)")

    def kind(self, column: str) -> str:
        if column in self.numeric:
            return "numeric"
        if column in self.categorical:
            return "categorical"
        if column in self.binary:
            return "binary"
        if column == self.target:
            return "target"
        return "unknown"
