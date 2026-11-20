"""Pandera checks for the raw transaction table.

Schema failure means the batch is not safe to score or train on. Drift, which
lives in payguard.monitoring, is a separate question: valid rows can still
come from a shifted population.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from payguard.schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CHANNELS,
    COUNTRY_MATCH,
    DEVICE_TRUST,
    MERCHANT_CATEGORIES,
    TARGET_COLUMN,
)

_CATEGORIES = {
    "merchant_category": MERCHANT_CATEGORIES,
    "channel": CHANNELS,
    "country_match": COUNTRY_MATCH,
    "device_trust": DEVICE_TRUST,
}


def transaction_schema() -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "transaction_id": pa.Column(str, nullable=False, unique=True),
        "amount": pa.Column(float, checks=[pa.Check.gt(0), pa.Check.lt(1_000_000)], nullable=False),
        "hour": pa.Column(int, checks=pa.Check.in_range(0, 23), nullable=False),
        "account_age_days": pa.Column(int, checks=pa.Check.in_range(0, 20000), nullable=False),
        "credit_limit": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "utilization": pa.Column(float, checks=pa.Check.in_range(0.0, 1.0), nullable=False),
        "txn_count_24h": pa.Column(int, checks=pa.Check.in_range(0, 100), nullable=False),
        "avg_amount_30d": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "days_since_last_txn": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "distance_from_home_km": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "merchant_risk_score": pa.Column(float, checks=pa.Check.in_range(0.0, 1.0), nullable=False),
        "failed_attempts_24h": pa.Column(int, checks=pa.Check.in_range(0, 50), nullable=False),
        TARGET_COLUMN: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
    }
    for name in CATEGORICAL_COLUMNS:
        columns[name] = pa.Column(
            str, checks=pa.Check.isin(list(_CATEGORIES[name])), nullable=False
        )
    return pa.DataFrameSchema(columns, coerce=True, strict=False)


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return transaction_schema().validate(frame)
