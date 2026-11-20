"""FastAPI scoring service.

The process loads one joblib bundle at startup. MLflow is the system of record
for runs; this service does not call MLflow on the request path.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from payguard import __version__
from payguard.explain.shap_report import build_explainer, shap_matrix
from payguard.schema import RAW_FEATURE_COLUMNS
from payguard.scoring import encode_frame, predict_calibrated
from payguard.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    FeatureContribution,
    HealthResponse,
    PredictionResponse,
    Transaction,
)

EXPLANATION_NOTE = (
    "Contributions are SHAP values in the model margin (log-odds), not "
    "percentage-point effects on the calibrated probability."
)


def _frame(transactions: list[Transaction]) -> pd.DataFrame:
    rows = [transaction.model_dump() for transaction in transactions]
    return pd.DataFrame(rows).loc[:, list(RAW_FEATURE_COLUMNS)]


def _predictions(artifact: dict[str, Any], frame: pd.DataFrame) -> list[PredictionResponse]:
    _, calibrated = predict_calibrated(
        artifact["preprocessor"],
        artifact["model"],
        artifact["calibrator"],
        frame,
    )
    threshold = float(artifact["threshold"])
    return [
        PredictionResponse(
            fraud_probability=float(probability),
            fraud_flag=bool(probability >= threshold),
            threshold=threshold,
            model_name=str(artifact["model_name"]),
            model_version=str(artifact["model_version"]),
        )
        for probability in calibrated
    ]


def _explainer(artifact: dict[str, Any]) -> object:
    cached = artifact.get("_explainer")
    if cached is None:
        cached = build_explainer(
            artifact["model"],
            str(artifact["model_family"]),
            np.asarray(artifact["background"], dtype=float),
        )
        artifact["_explainer"] = cached
    return cached


def create_app(model_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        path = Path(application.state.model_path)
        if path.exists():
            application.state.artifact = joblib.load(path)
            application.state.load_error = None
        else:
            application.state.artifact = None
            application.state.load_error = f"model not found at {path}"
        yield

    application = FastAPI(
        title="PayGuard",
        version=__version__,
        summary="Payment-card fraud scores, batch scores, and SHAP explanations.",
        lifespan=lifespan,
    )
    resolved = (
        model_path
        if model_path is not None
        else os.environ.get("PAYGUARD_MODEL_PATH", "models/model.joblib")
    )
    application.state.model_path = str(resolved)
    application.state.artifact = None
    application.state.load_error = None

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "payguard", "health": "/health", "docs": "/docs"}

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        artifact = application.state.artifact
        if artifact is None:
            return HealthResponse(
                status="degraded",
                model_loaded=False,
                detail=application.state.load_error,
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=str(artifact["model_name"]),
            model_version=str(artifact["model_version"]),
        )

    @application.get("/model")
    def model_info() -> dict[str, Any]:
        artifact = _require_artifact(application)
        return {
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
            "model_family": artifact["model_family"],
            "threshold": artifact["threshold"],
            "precision_target": artifact.get("precision_target"),
            "trained_at": artifact.get("trained_at"),
            "metrics": artifact.get("metrics", {}),
            "mlflow": artifact.get("mlflow", {}),
            "shap_top_features": artifact.get("shap_top_features", []),
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(transaction: Transaction) -> PredictionResponse:
        artifact = _require_artifact(application)
        return _predictions(artifact, _frame([transaction]))[0]

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        artifact = _require_artifact(application)
        predictions = _predictions(artifact, _frame(request.transactions))
        return BatchPredictionResponse(count=len(predictions), predictions=predictions)

    @application.post("/explain", response_model=ExplanationResponse)
    def explain(transaction: Transaction) -> ExplanationResponse:
        artifact = _require_artifact(application)
        frame = _frame([transaction])
        prediction = _predictions(artifact, frame)[0]
        encoded = encode_frame(artifact["preprocessor"], frame)
        values, base_value = shap_matrix(_explainer(artifact), encoded)
        vector = np.asarray(values, dtype=float).reshape(-1)
        names = list(artifact["feature_names"])
        order = np.argsort(np.abs(vector))[::-1][:8]
        contributions = [
            FeatureContribution(
                feature=names[int(index)],
                shap_value=float(vector[int(index)]),
                encoded_value=float(encoded[0, int(index)]),
            )
            for index in order
        ]
        return ExplanationResponse(
            fraud_probability=prediction.fraud_probability,
            fraud_flag=prediction.fraud_flag,
            threshold=prediction.threshold,
            base_value=float(base_value),
            space="model_margin",
            contributions=contributions,
            note=EXPLANATION_NOTE,
        )

    return application


def _require_artifact(application: FastAPI) -> dict[str, Any]:
    artifact = application.state.artifact
    if artifact is None:
        raise HTTPException(
            status_code=503, detail=application.state.load_error or "model not loaded"
        )
    return artifact


app = create_app()
