"""Leakage-safe feature pipeline.

TransactionFeatureBuilder only adds row-wise transforms. Anything that would
need a training-set statistic (scaling, imputation, one-hot categories) sits
in the ColumnTransformer and is fit inside each CV fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from payguard.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, RAW_FEATURE_COLUMNS

ENGINEERED_COLUMNS: tuple[str, ...] = (
    "log_amount",
    "amount_to_avg_ratio",
    "is_night",
    "is_high_velocity",
    "is_foreign_online",
    "is_new_device_night",
)
MODEL_NUMERIC_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + ENGINEERED_COLUMNS


class TransactionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add the interactions and ratios the linear baseline is allowed to see."""

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
        amount = frame["amount"].astype(float)
        frame["log_amount"] = np.log1p(amount)
        frame["amount_to_avg_ratio"] = amount / (frame["avg_amount_30d"].astype(float) + 1.0)
        hour = frame["hour"].astype(int)
        night = ((hour <= 5) | (hour >= 23)).astype(float)
        frame["is_night"] = night
        frame["is_high_velocity"] = (frame["txn_count_24h"].astype(float) >= 6).astype(float)
        foreign = (frame["country_match"].astype(str) == "foreign").astype(float)
        online = (frame["channel"].astype(str) == "online").astype(float)
        new_device = (frame["device_trust"].astype(str) == "new").astype(float)
        frame["is_foreign_online"] = foreign * online
        frame["is_new_device_night"] = new_device * night
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
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    encoder = ColumnTransformer(
        transformers=[
            ("num", numeric, list(MODEL_NUMERIC_COLUMNS)),
            ("cat", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("build", TransactionFeatureBuilder()),
            ("encode", encoder),
        ]
    )
