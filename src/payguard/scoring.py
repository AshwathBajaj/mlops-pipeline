"""Score a frame with a fitted preprocessor, model, and Platt calibrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from payguard.schema import RAW_FEATURE_COLUMNS


def scores_to_logits(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Sigmoid calibration fit on out-of-fold scores.

    The mapping is monotone, so it does not change PR-AUC. It does change Brier
    score and the threshold we ship.
    """
    calibrator = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500)
    calibrator.fit(scores_to_logits(scores), y.astype(int))
    return calibrator


def apply_platt(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(scores_to_logits(scores))[:, 1]


def encode_frame(preprocessor: object, frame: pd.DataFrame) -> np.ndarray:
    encoded = preprocessor.transform(frame.loc[:, list(RAW_FEATURE_COLUMNS)])  # type: ignore[attr-defined]
    return np.asarray(encoded, dtype=float)


def raw_probabilities(model: object, encoded: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(encoded)[:, 1]  # type: ignore[attr-defined]
    return np.asarray(probabilities, dtype=float)


def predict_calibrated(
    preprocessor: object,
    model: object,
    calibrator: LogisticRegression,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = encode_frame(preprocessor, frame)
    raw = raw_probabilities(model, encoded)
    calibrated = apply_platt(calibrator, raw)
    return raw, calibrated
