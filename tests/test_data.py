from __future__ import annotations

import urllib.error

import pandas as pd
import pandera.errors
import pytest

from payguard.data.generate import fraud_effects, generate_transactions
from payguard.data.load import load_transactions, normalize_transactions
from payguard.data.validate import validate_transactions
from payguard.schema import ALL_COLUMNS, PCA_COLUMNS, TARGET_COLUMN


def test_generator_matches_contract_and_rate() -> None:
    frame = generate_transactions(n_rows=2000, seed=4, target_fraud_rate=0.02)
    validated = validate_transactions(frame)
    assert list(validated.columns) == list(ALL_COLUMNS)
    rate = float(validated[TARGET_COLUMN].mean())
    assert 0.005 <= rate <= 0.08
    assert validated["transaction_id"].is_unique
    assert set(PCA_COLUMNS).issubset(validated.columns)


def test_generator_is_deterministic() -> None:
    left = generate_transactions(300, seed=9, target_fraud_rate=0.03)
    right = generate_transactions(300, seed=9, target_fraud_rate=0.03)
    pd.testing.assert_frame_equal(left, right)


def test_effects_increase_for_a_riskier_row() -> None:
    calm = generate_transactions(1, seed=1, target_fraud_rate=0.02)
    risky = calm.copy()
    risky.loc[0, "V14"] = float(calm.loc[0, "V14"]) - 4.0
    risky.loc[0, "V4"] = float(calm.loc[0, "V4"]) + 4.0
    risky.loc[0, "Amount"] = float(calm.loc[0, "Amount"]) * 8
    risky.loc[0, "Time"] = 2.0 * 3600.0
    assert float(fraud_effects(risky)[0]) > float(fraud_effects(calm)[0])


def test_schema_rejects_bad_amount() -> None:
    frame = generate_transactions(80, seed=2, target_fraud_rate=0.05)
    frame.loc[0, "Amount"] = -5
    with pytest.raises(pandera.errors.SchemaError):
        validate_transactions(frame)


def test_class_alias_is_renamed_to_fraud() -> None:
    frame = generate_transactions(30, seed=2, target_fraud_rate=0.05)
    raw = frame.drop(columns=["transaction_id"]).rename(columns={"fraud": "Class"})
    normalized = normalize_transactions(raw)
    assert "fraud" in normalized.columns
    assert "Class" not in normalized.columns
    assert normalized["transaction_id"].is_unique


def test_missing_column_is_a_clear_error() -> None:
    frame = generate_transactions(80, seed=2, target_fraud_rate=0.05).drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Amount"):
        validate_transactions(frame)


def test_csv_loader_reads_a_local_file(tmp_path) -> None:
    frame = generate_transactions(100, seed=3, target_fraud_rate=0.04)
    path = tmp_path / "transactions.csv"
    frame.to_csv(path, index=False)
    loaded = load_transactions({"seed": 3, "data": {"source": "csv", "path": str(path)}})
    assert len(loaded) == 100
    assert "fraud" in loaded.columns
    assert "V1" in loaded.columns


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


def test_openml_failure_falls_back_to_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("payguard.data.load.fetch_openml_frame", fail)
    monkeypatch.setattr("payguard.data.load.fetch_openml_via_sklearn", fail)
    loaded = load_transactions(
        {
            "seed": 5,
            "data": {
                "source": "openml",
                "openml_id": 1597,
                "n_rows": 70,
                "target_fraud_rate": 0.04,
            },
        }
    )
    assert len(validate_transactions(loaded)) == 70
