from __future__ import annotations

from typing import get_args

from fastapi.testclient import TestClient

from payguard.schema import (
    CATEGORICAL_COLUMNS,
    CHANNELS,
    COUNTRY_MATCH,
    DEVICE_TRUST,
    MERCHANT_CATEGORIES,
)
from payguard.serving.app import create_app
from payguard.serving.schemas import Channel, CountryMatch, DeviceTrust, MerchantCategory


def test_schema_literals_match_the_training_contract() -> None:
    assert set(get_args(MerchantCategory)) == set(MERCHANT_CATEGORIES)
    assert set(get_args(Channel)) == set(CHANNELS)
    assert set(get_args(CountryMatch)) == set(COUNTRY_MATCH)
    assert set(get_args(DeviceTrust)) == set(DEVICE_TRUST)
    assert set(CATEGORICAL_COLUMNS) == {
        "merchant_category",
        "channel",
        "country_match",
        "device_trust",
    }


def test_health_is_degraded_without_a_model(tmp_path, suspicious_transaction: dict) -> None:
    app = create_app(tmp_path / "missing.joblib")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["model_loaded"] is False
        rejected = client.post("/predict", json=suspicious_transaction)
        assert rejected.status_code == 503


def test_predict_batch_and_explain(
    trained: dict,
    benign_transaction: dict,
    suspicious_transaction: dict,
) -> None:
    app = create_app(trained["model_path"])
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["model_name"] == trained["summary"]["selected_model"]

        single = client.post("/predict", json=suspicious_transaction)
        assert single.status_code == 200
        payload = single.json()
        assert 0.0 <= payload["fraud_probability"] <= 1.0
        assert payload["fraud_flag"] is (payload["fraud_probability"] >= payload["threshold"])

        batch = client.post(
            "/predict/batch",
            json={"transactions": [benign_transaction, suspicious_transaction]},
        )
        assert batch.status_code == 200
        predictions = batch.json()["predictions"]
        assert batch.json()["count"] == 2
        assert predictions[1]["fraud_probability"] > predictions[0]["fraud_probability"]

        explanation = client.post("/explain", json=suspicious_transaction)
        assert explanation.status_code == 200
        explained = explanation.json()
        assert explained["contributions"]
        assert explained["space"] == "model_margin"
        assert abs(explained["fraud_probability"] - payload["fraud_probability"]) < 1e-6

        info = client.get("/model")
        assert info.status_code == 200
        assert info.json()["threshold"] == payload["threshold"]


def test_invalid_payloads_are_rejected(trained: dict, suspicious_transaction: dict) -> None:
    app = create_app(trained["model_path"])
    with TestClient(app) as client:
        negative = dict(suspicious_transaction)
        negative["amount"] = -1
        assert client.post("/predict", json=negative).status_code == 422

        unknown = dict(suspicious_transaction)
        unknown["channel"] = "wire"
        assert client.post("/predict", json=unknown).status_code == 422

        empty = client.post("/predict/batch", json={"transactions": []})
        assert empty.status_code == 422

        missing = dict(suspicious_transaction)
        del missing["hour"]
        assert client.post("/predict", json=missing).status_code == 422
