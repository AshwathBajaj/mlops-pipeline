from __future__ import annotations

import urllib.error

import pandas as pd
import pandera.errors
import pytest

from payguard.data.generate import fraud_effects, generate_transactions
from payguard.data.load import load_transactions
from payguard.data.validate import validate_transactions
from payguard.schema import ALL_COLUMNS, TARGET_COLUMN


def test_generator_matches_contract_and_rate() -> None:
    frame = generate_transactions(n_rows=2000, seed=4, target_fraud_rate=0.02)
    validated = validate_transactions(frame)
    assert list(validated.columns) == list(ALL_COLUMNS)
    rate = float(validated[TARGET_COLUMN].mean())
    assert 0.005 <= rate <= 0.08
    assert validated["transaction_id"].is_unique
    assert set(validated["merchant_category"].unique()) >= {"grocery", "gambling"}


def test_generator_is_deterministic() -> None:
    left = generate_transactions(300, seed=9, target_fraud_rate=0.03)
    right = generate_transactions(300, seed=9, target_fraud_rate=0.03)
    pd.testing.assert_frame_equal(left, right)


def test_effects_increase_for_a_riskier_row() -> None:
    calm = generate_transactions(1, seed=1, target_fraud_rate=0.02)
    risky = calm.copy()
    risky.loc[0, "channel"] = "online"
    risky.loc[0, "country_match"] = "foreign"
    risky.loc[0, "device_trust"] = "new"
    risky.loc[0, "hour"] = 2
    risky.loc[0, "amount"] = float(calm.loc[0, "amount"]) * 8
    assert float(fraud_effects(risky)[0]) > float(fraud_effects(calm)[0])


def test_schema_rejects_bad_amount() -> None:
    frame = generate_transactions(80, seed=2, target_fraud_rate=0.05)
    frame.loc[0, "amount"] = -5
    with pytest.raises(pandera.errors.SchemaError):
        validate_transactions(frame)


def test_schema_rejects_unknown_category() -> None:
    frame = generate_transactions(80, seed=2, target_fraud_rate=0.05)
    frame.loc[0, "channel"] = "wire"
    with pytest.raises(pandera.errors.SchemaError):
        validate_transactions(frame)


def test_missing_column_is_a_clear_error() -> None:
    frame = generate_transactions(80, seed=2, target_fraud_rate=0.05).drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        validate_transactions(frame)


def test_csv_loader_reads_a_local_file(tmp_path) -> None:
    frame = generate_transactions(100, seed=3, target_fraud_rate=0.04)
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)
    loaded = load_transactions({"seed": 3, "data": {"source": "csv", "path": str(path)}})
    assert len(loaded) == 100
    assert "fraud" in loaded.columns


def test_url_failure_falls_back_to_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("payguard.data.load.download_csv", fail)
    loaded = load_transactions(
        {
            "seed": 5,
            "data": {
                "source": "url",
                "url": "https://example.invalid/transactions.csv",
                "download_path": "data/raw/downloaded.csv",
                "n_rows": 90,
                "target_fraud_rate": 0.04,
                "timeout_seconds": 1,
            },
        }
    )
    validated = validate_transactions(loaded)
    assert len(validated) == 90
