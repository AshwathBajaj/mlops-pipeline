"""Pandera checks for the credit-card transaction table.

A schema failure means the row is not a legal transaction. Drift, in
payguard.monitoring, is a separate question: valid rows can still come from
a shifted population.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from payguard.schema import ALL_COLUMNS, PCA_COLUMNS, TARGET_COLUMN


def transaction_schema() -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "transaction_id": pa.Column(str, nullable=False, unique=True),
        "Time": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "Amount": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        TARGET_COLUMN: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
    }
    for name in PCA_COLUMNS:
        columns[name] = pa.Column(float, nullable=False)
    return pa.DataFrameSchema(columns, coerce=True, strict=False)


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return transaction_schema().validate(frame)
