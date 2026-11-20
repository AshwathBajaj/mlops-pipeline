from __future__ import annotations

import numpy as np
import pytest

from payguard.data.generate import generate_transactions
from payguard.features.build import ENGINEERED_COLUMNS, TransactionFeatureBuilder, make_preprocessor
from payguard.schema import RAW_FEATURE_COLUMNS


def _one_row(**overrides: object):
    frame = generate_transactions(1, seed=1, target_fraud_rate=0.02)
    for key, value in overrides.items():
        frame.loc[0, key] = value
    return frame


def test_feature_builder_adds_log_amount_and_hour() -> None:
    frame = _one_row(Amount=100.0, Time=2 * 3600)
    built = (
        TransactionFeatureBuilder()
        .fit(frame[list(RAW_FEATURE_COLUMNS)])
        .transform(frame[list(RAW_FEATURE_COLUMNS)])
    )
    assert set(ENGINEERED_COLUMNS).issubset(built.columns)
    assert built.loc[0, "log_amount"] == pytest.approx(np.log1p(100.0))
    assert built.loc[0, "hour"] == pytest.approx(2.0)
    assert built.loc[0, "is_night"] == 1.0


def test_daytime_is_not_night() -> None:
    frame = _one_row(Time=15 * 3600)
    built = TransactionFeatureBuilder().transform(frame[list(RAW_FEATURE_COLUMNS)])
    assert built.loc[0, "hour"] == pytest.approx(15.0)
    assert built.loc[0, "is_night"] == 0.0


def test_preprocessor_is_finite() -> None:
    frame = generate_transactions(500, seed=8, target_fraud_rate=0.05)
    features = frame[list(RAW_FEATURE_COLUMNS)]
    preprocessor = make_preprocessor()
    preprocessor.fit(features.iloc[:400])
    encoded = np.asarray(preprocessor.transform(features.iloc[400:]), dtype=float)
    assert encoded.ndim == 2
    assert encoded.shape[0] == 100
    assert np.isfinite(encoded).all()
    names = list(preprocessor.get_feature_names_out())
    assert len(names) == encoded.shape[1]
    assert "log_amount" in names
    assert "V14" in names


def test_builder_rejects_missing_columns() -> None:
    frame = generate_transactions(20, seed=1, target_fraud_rate=0.05)
    with pytest.raises(ValueError, match="Amount"):
        TransactionFeatureBuilder().fit(frame.drop(columns=["Amount"]))
