"""Response schemas.

Shared across the three diseases because the *shape* of an answer is the same even though
the inputs are not. Several fields exist specifically to stop a number being read as more
than it is:

``flagged`` vs ``risk_band``
    ``flagged`` is the model's actual decision at its own operating threshold. ``risk_band``
    is a presentation label. They are reported separately so the label can never be mistaken
    for the decision.
``explanation.uncalibrated_probability``
    SHAP explains the *base* pipeline; ``probability`` comes from the calibrated wrapper.
    Both are returned rather than pretending the contributions sum to the calibrated number.
``imputed_features`` / ``extrapolated_features``
    What the caller did not supply, and what they supplied outside the training range.
``disclaimer``
    On every response, including errors that still carry a body.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    feature: str
    value: float | None = Field(description="The value used, after imputation.")
    contribution: float = Field(
        description="Signed SHAP contribution on the uncalibrated model score. "
                    "Positive pushes towards the positive class."
    )
    description: str


class Explanation(BaseModel):
    method: str
    explainer: str = Field(description="SHAP explainer used: tree, linear or kernel.")
    base_value: float = Field(
        description="The explainer's expected model output over the background sample."
    )
    uncalibrated_probability: float = Field(
        description="The BASE pipeline's probability — the quantity these contributions "
                    "reconstruct. Not the calibrated `probability` field above."
    )
    note: str
    contributions: list[FeatureContribution]


class RiskBandInfo(BaseModel):
    label: str
    lower: float
    upper: float
    note: str


class PredictionResponse(BaseModel):
    disease: str
    module: str
    positive_class_meaning: str
    probability: float = Field(
        description="Calibrated model-estimated probability of the positive class."
    )
    flagged: bool = Field(
        description="Whether the model would flag this case at its own operating "
                    "threshold. This, not the risk band, is the model's decision."
    )
    threshold: float
    threshold_rule: str
    risk_band: RiskBandInfo
    model_name: str
    calibration: str
    model_version: str = Field(description="Date the model card was generated.")
    n_features_expected: int
    n_features_provided: int
    imputed_features: list[str]
    extrapolated_features: list[dict]
    warnings: list[str]
    explanation: Explanation | None
    disclaimer: str

    model_config = {"protected_namespaces": ()}


class ScenarioRequest(BaseModel):
    """A base case plus the feature overrides to apply to it."""

    features: dict[str, float | None] = Field(
        description="The base case, validated against the same schema as /predict."
    )
    overrides: dict[str, float | None] = Field(
        description="Feature values to change. Validated the same way."
    )


class ScenarioResponse(BaseModel):
    disease: str
    module: str
    label: str = Field(
        description="Always the illustrative-scenario label. Read it before the numbers."
    )
    baseline: PredictionResponse
    modified: PredictionResponse
    changed_features: list[dict]
    probability_delta: float
    band_changed: bool
    flag_changed: bool
    warnings: list[str]
    disclaimer: str


class ModelSummary(BaseModel):
    disease: str
    module: str
    positive_class_meaning: str
    model_name: str
    calibration: str
    version: str
    n_train: int
    n_test: int
    roc_auc_test: float
    operating_threshold: float
    external_validation: dict
    explainer: dict
    limitations: list[str]
    disclaimer: str

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    diseases: dict[str, dict]
    disclaimer: str
