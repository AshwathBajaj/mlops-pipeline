"""MLflow PythonModel wrapper around the serving bundle."""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import pandas as pd
from mlflow.pyfunc import PythonModel

from payguard.scoring import predict_calibrated


class PayGuardPyfunc(PythonModel):
    """Score raw transactions. The registry entry points at this wrapper."""

    def load_context(self, context: object) -> None:
        artifacts = context.artifacts  # type: ignore[attr-defined]
        self.bundle = joblib.load(artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        del context, params
        frame = pd.DataFrame(model_input)
        bundle = self.bundle
        _, calibrated = predict_calibrated(
            bundle["preprocessor"],
            bundle["model"],
            bundle["calibrator"],
            frame,
        )
        threshold = float(bundle["threshold"])
        return pd.DataFrame(
            {
                "fraud_probability": calibrated,
                "fraud_flag": (calibrated >= threshold).astype(int),
            }
        )
