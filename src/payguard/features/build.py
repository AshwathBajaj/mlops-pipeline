"""Leakage-safe feature pipeline for the ULB credit-card columns.

TransactionFeatureBuilder only adds row-wise transforms of Time and Amount.
Scaling is fit inside each CV fold, in the ColumnTransformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from payguard.schema import NUMERIC_COLUMNS, RAW_FEATURE_COLUMNS

ENGINEERED_COLUMNS: tuple[str, ...] = ("log_amount", "hour", "is_night")
MODEL_NUMERIC_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + ENGINEERED_COLUMNS


class TransactionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add log amount and the hour of day implied by Time."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> TransactionFeatureBuilder:
        del y
        missing = [column for column in RAW_FEATURE_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        missing = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        frame["log_amount"] = np.log1p(frame["Amount"].astype(float))
        frame["hour"] = (frame["Time"].astype(float) % 86400.0) / 3600.0
        hour = frame["hour"]
        frame["is_night"] = ((hour <= 5.0) | (hour >= 23.0)).astype(float)
        return frame

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        del input_features
        return np.asarray([*RAW_FEATURE_COLUMNS, *ENGINEERED_COLUMNS], dtype=object)


def make_preprocessor() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    encoder = ColumnTransformer(
        transformers=[("num", numeric, list(MODEL_NUMERIC_COLUMNS))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("build", TransactionFeatureBuilder()),
            ("encode", encoder),
        ]
    )
