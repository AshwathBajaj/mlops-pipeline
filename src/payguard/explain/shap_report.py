"""Global and single-row explanations.

Tree SHAP values are in the model's margin (log-odds) space. They rank which
encoded features pushed this score up or down. They are not percentage-point
effects on the calibrated probability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shap

from payguard.scoring import encode_frame


def _positive_shap(values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        return np.asarray(array[:, :, -1], dtype=float)
    if isinstance(values, list):
        return np.asarray(values[-1], dtype=float)
    return np.asarray(array, dtype=float)


def _expected_value(explainer: object) -> float:
    expected = getattr(explainer, "expected_value", 0.0)
    array = np.atleast_1d(np.asarray(expected, dtype=float))
    return float(array[-1])


def build_explainer(model: object, family: str, background: np.ndarray) -> object:
    if family == "tree":
        return shap.TreeExplainer(model)
    sample = np.asarray(background, dtype=float)
    if len(sample) > 100:
        sample = sample[:100]
    return shap.LinearExplainer(model, sample)


def shap_matrix(explainer: object, encoded: np.ndarray) -> tuple[np.ndarray, float]:
    values = explainer.shap_values(encoded)  # type: ignore[attr-defined]
    return _positive_shap(values), _expected_value(explainer)


def global_shap(
    model: object,
    preprocessor: object,
    frame: object,
    feature_names: list[str],
    family: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    encoded = encode_frame(preprocessor, frame)
    rng = np.random.default_rng(seed)
    count = min(sample_size, len(encoded))
    chosen = rng.choice(len(encoded), size=count, replace=False)
    sample = encoded[np.sort(chosen)]
    background = sample[: min(200, len(sample))]
    explainer = build_explainer(model, family, background)
    values, expected = shap_matrix(explainer, sample)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = [
        {
            "feature": feature_names[int(index)],
            "mean_abs_shap": float(mean_abs[int(index)]),
        }
        for index in order[:15]
    ]
    return {
        "sample_size": int(count),
        "base_value": expected,
        "space": "model_margin",
        "top_features": top,
    }


def explain_encoded_row(
    model: object,
    encoded_row: np.ndarray,
    feature_names: list[str],
    family: str,
    background: np.ndarray,
    top_k: int = 8,
) -> dict[str, Any]:
    row = np.asarray(encoded_row, dtype=float).reshape(1, -1)
    explainer = build_explainer(model, family, background)
    values, expected = shap_matrix(explainer, row)
    vector = values.reshape(-1)
    order = np.argsort(np.abs(vector))[::-1][:top_k]
    contributions = [
        {
            "feature": feature_names[int(index)],
            "shap_value": float(vector[int(index)]),
            "encoded_value": float(row[0, int(index)]),
        }
        for index in order
    ]
    return {
        "base_value": expected,
        "contributions": contributions,
        "space": "model_margin",
    }
