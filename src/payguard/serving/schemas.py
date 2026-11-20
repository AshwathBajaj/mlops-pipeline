"""Request and response models for the scoring API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from payguard import __version__

MerchantCategory = Literal[
    "grocery",
    "fuel",
    "dining",
    "electronics",
    "travel",
    "jewelry",
    "digital_goods",
    "gambling",
]
Channel = Literal["chip", "contactless", "swipe", "online"]
CountryMatch = Literal["domestic", "foreign"]
DeviceTrust = Literal["trusted", "new", "shared"]


class Transaction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 186.4,
                "hour": 2,
                "account_age_days": 18,
                "credit_limit": 2500,
                "utilization": 0.55,
                "txn_count_24h": 7,
                "avg_amount_30d": 24.0,
                "days_since_last_txn": 0.3,
                "distance_from_home_km": 820,
                "merchant_risk_score": 0.62,
                "failed_attempts_24h": 2,
                "merchant_category": "electronics",
                "channel": "online",
                "country_match": "foreign",
                "device_trust": "new",
            }
        }
    )

    amount: float = Field(gt=0, lt=1_000_000)
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(ge=0, le=20_000)
    credit_limit: float = Field(gt=0)
    utilization: float = Field(ge=0, le=1)
    txn_count_24h: int = Field(ge=0, le=100)
    avg_amount_30d: float = Field(ge=0)
    days_since_last_txn: float = Field(ge=0)
    distance_from_home_km: float = Field(ge=0)
    merchant_risk_score: float = Field(ge=0, le=1)
    failed_attempts_24h: int = Field(ge=0, le=50)
    merchant_category: MerchantCategory
    channel: Channel
    country_match: CountryMatch
    device_trust: DeviceTrust


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
