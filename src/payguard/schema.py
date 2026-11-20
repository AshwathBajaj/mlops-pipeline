"""Column contract shared by generation, validation, training, and serving."""

from __future__ import annotations

ID_COLUMN = "transaction_id"
TARGET_COLUMN = "fraud"

MERCHANT_CATEGORIES: tuple[str, ...] = (
    "grocery",
    "fuel",
    "dining",
    "electronics",
    "travel",
    "jewelry",
    "digital_goods",
    "gambling",
)
CHANNELS: tuple[str, ...] = ("chip", "contactless", "swipe", "online")
COUNTRY_MATCH: tuple[str, ...] = ("domestic", "foreign")
DEVICE_TRUST: tuple[str, ...] = ("trusted", "new", "shared")

NUMERIC_COLUMNS: tuple[str, ...] = (
    "amount",
    "hour",
    "account_age_days",
    "credit_limit",
    "utilization",
    "txn_count_24h",
    "avg_amount_30d",
    "days_since_last_txn",
    "distance_from_home_km",
    "merchant_risk_score",
    "failed_attempts_24h",
)
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "merchant_category",
    "channel",
    "country_match",
    "device_trust",
)
RAW_FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
ALL_COLUMNS: tuple[str, ...] = (ID_COLUMN, *RAW_FEATURE_COLUMNS, TARGET_COLUMN)
