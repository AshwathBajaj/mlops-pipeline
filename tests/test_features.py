from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from payguard.data.generate import generate_transactions
from payguard.features.build import ENGINEERED_COLUMNS, TransactionFeatureBuilder, make_preprocessor
from payguard.schema import RAW_FEATURE_COLUMNS


def _one_row(**overrides: object) -> pd.DataFrame:
    frame = generate_transactions(1, seed=1, target_fraud_rate=0.02)
    for key, value in overrides.items():
        frame.loc[0, key] = value
    return frame


def test_feature_builder_adds_known_columns() -> None:
    frame = _one_row(amount=100.0, avg_amount_30d=10.0, hour=2, txn_count_24h=8)
    frame.loc[0, "country_match"] = "foreign"
    frame.loc[0, "channel"] = "online"
    frame.loc[0, "device_trust"] = "new"
    built = (
        TransactionFeatureBuilder()
        .fit(frame[list(RAW_FEATURE_COLUMNS)])
        .transform(frame[list(RAW_FEATURE_COLUMNS)])
    )
    assert set(ENGINEERED_COLUMNS).issubset(built.columns)
    assert built.loc[0, "amount_to_avg_ratio"] == pytest.approx(100.0 / 11.0)
    assert built.loc[0, "log_amount"] == pytest.approx(np.log1p(100.0))
    assert built.loc[0, "is_night"] == 1
    assert built.loc[0, "is_high_velocity"] == 1
    assert built.loc[0, "is_foreign_online"] == 1
    assert built.loc[0, "is_new_device_night"] == 1


def test_daytime_domestic_chip_has_zero_flags() -> None:
    frame = _one_row(hour=15, txn_count_24h=1)
    frame.loc[0, "country_match"] = "domestic"
    frame.loc[0, "channel"] = "chip"
    frame.loc[0, "device_trust"] = "trusted"
    built = TransactionFeatureBuilder().transform(frame[list(RAW_FEATURE_COLUMNS)])
    assert built.loc[0, "is_night"] == 0
    assert built.loc[0, "is_high_velocity"] == 0
    assert built.loc[0, "is_foreign_online"] == 0
    assert built.loc[0, "is_new_device_night"] == 0


def test_preprocessor_is_finite_and_accepts_unknown_categories() -> None:
    frame = generate_transactions(500, seed=8, target_fraud_rate=0.05)
    features = frame[list(RAW_FEATURE_COLUMNS)]
    preprocessor = make_preprocessor()
    preprocessor.fit(features.iloc[:400])
    held_out = features.iloc[400:].copy()
    held_out.loc[held_out.index[0], "merchant_category"] = "not_a_real_category"
    encoded = np.asarray(preprocessor.transform(held_out), dtype=float)
    assert encoded.ndim == 2
    assert encoded.shape[0] == len(held_out)
    assert np.isfinite(encoded).all()
    names = list(preprocessor.get_feature_names_out())
    assert len(names) == encoded.shape[1]
    assert "log_amount" in names


def test_builder_rejects_missing_columns() -> None:
    frame = generate_transactions(20, seed=1, target_fraud_rate=0.05)
    with pytest.raises(ValueError, match="amount"):
        TransactionFeatureBuilder().fit(frame.drop(columns=["amount"]))
