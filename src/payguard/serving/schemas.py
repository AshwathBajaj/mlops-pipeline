"""Request and response models for the scoring API.

A request is one row of the ULB credit-card table: Time, Amount, and V1-V28.
The label is not part of the request.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from payguard import __version__
from payguard.schema import PCA_COLUMNS


class Transaction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Time": 40000.0,
                "Amount": 12.5,
                **{name: 0.0 for name in PCA_COLUMNS},
            }
        },
    )

    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_flag: bool
    threshold: float
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    encoded_value: float


class ExplanationResponse(BaseModel):
    fraud_probability: float
    fraud_flag: bool
    threshold: float
    base_value: float
    space: str
    contributions: list[FeatureContribution]
    note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    service: str = "payguard"
    version: str = __version__
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None
