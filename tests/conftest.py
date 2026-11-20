"""Shared fixtures. One small training run serves the model, API, and MLflow tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from payguard.data.generate import generate_transactions
from payguard.data.validate import validate_transactions
from payguard.scoring import encode_frame
from payguard.training.train import run_training


def tiny_config(tmp: Path) -> dict:
    return {
        "seed": 11,
        "data": {
            "source": "synthetic",
            "n_rows": 1800,
            "target_fraud_rate": 0.06,
            "output_path": str(tmp / "transactions.csv"),
        },
        "split": {"test_size": 0.25},
        "training": {
            "cv_folds": 2,
            "precision_target": 0.6,
            "shap_sample_size": 80,
            "models": {
                "logistic_regression": {"C": 1.0, "max_iter": 200},
                "hist_gradient_boosting": {
                    "max_iter": 25,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "min_samples_leaf": 15,
                },
                "xgboost": {
                    "n_estimators": 25,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "min_child_weight": 1,
                },
            },
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp / 'mlflow.db'}",
            "artifact_location": str(tmp / "mlartifacts"),
            "experiment_name": "payguard-test",
            "registered_model_name": "payguard-test-detector",
        },
        "monitoring": {
            "psi_bins": 8,
            "feature_psi_threshold": 0.2,
            "prediction_psi_threshold": 0.1,
            "min_features_drifted": 2,
            "drift_rows": 400,
        },
        "output": {
            "model_path": str(tmp / "model.joblib"),
            "metrics_path": str(tmp / "metrics.json"),
            "shap_path": str(tmp / "shap.json"),
            "drift_json_path": str(tmp / "drift.json"),
            "drift_md_path": str(tmp / "drift.md"),
            "plots_dir": str(tmp / "plots"),
        },
    }


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("trained")
    cfg = tiny_config(tmp)
    frame = validate_transactions(
        generate_transactions(
            n_rows=int(cfg["data"]["n_rows"]),
            seed=int(cfg["seed"]),
            target_fraud_rate=float(cfg["data"]["target_fraud_rate"]),
        )
    )
    summary = run_training(cfg, frame)
    fitted = summary["artifact"]
    encoded = encode_frame(fitted["preprocessor"], summary["reference_frame"])
    background = encoded[: min(40, len(encoded))]
    bundle = {
        "preprocessor": fitted["preprocessor"],
        "model": fitted["model"],
        "calibrator": fitted["calibrator"],
        "threshold": fitted["threshold"],
        "feature_names": fitted["feature_names"],
        "model_family": fitted["model_family"],
        "background": np.asarray(background, dtype=float),
        "model_name": summary["selected_model"],
        "model_version": f"{summary['selected_model']}-test",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "seed": int(cfg["seed"]),
        "precision_target": summary["precision_target"],
        "metrics": summary["models"][summary["selected_model"]],
        "mlflow": {},
        "shap_top_features": [],
        "input_columns": list(frame.columns),
    }
    model_path = Path(cfg["output"]["model_path"])
    joblib.dump(bundle, model_path)
    return {
        "cfg": cfg,
        "frame": frame,
        "summary": summary,
        "bundle": bundle,
        "model_path": model_path,
        "reference_frame": summary["reference_frame"],
    }


@pytest.fixture
def benign_transaction() -> dict:
    return {
        "amount": 18.0,
        "hour": 14,
        "account_age_days": 2400,
        "credit_limit": 10000,
        "utilization": 0.12,
        "txn_count_24h": 1,
        "avg_amount_30d": 22.0,
        "days_since_last_txn": 3.0,
        "distance_from_home_km": 4.0,
        "merchant_risk_score": 0.04,
        "failed_attempts_24h": 0,
        "merchant_category": "grocery",
        "channel": "chip",
        "country_match": "domestic",
        "device_trust": "trusted",
    }


@pytest.fixture
def suspicious_transaction() -> dict:
    return {
        "amount": 420.0,
        "hour": 2,
        "account_age_days": 12,
        "credit_limit": 1500,
        "utilization": 0.93,
        "txn_count_24h": 8,
        "avg_amount_30d": 20.0,
        "days_since_last_txn": 0.2,
        "distance_from_home_km": 900.0,
        "merchant_risk_score": 0.72,
        "failed_attempts_24h": 3,
        "merchant_category": "gambling",
        "channel": "online",
        "country_match": "foreign",
        "device_trust": "new",
    }
