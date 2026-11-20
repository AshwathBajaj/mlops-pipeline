"""Synthetic rows with the same columns as the ULB credit-card table.

This generator is the offline fallback for unit tests and for a failed
download. It is not the dataset behind the published metrics. Fraud labels
come from a logistic process on V14, V4, V12, V11, amount, and night-time,
plus a little label noise, so the tests have a signal they can check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from payguard.schema import ID_COLUMN, PCA_COLUMNS, TARGET_COLUMN


def fraud_effects(frame: pd.DataFrame) -> np.ndarray:
    """Linear predictor without the intercept. Used by the generator and tests."""
    amount = frame["Amount"].to_numpy(dtype=float)
    z_log = (np.log1p(amount) - 3.2) / 1.3
    hour = (frame["Time"].to_numpy(dtype=float) % 86400.0) / 3600.0
    night = ((hour <= 5.0) | (hour >= 23.0)).astype(float)
    return (
        -1.50 * frame["V14"].to_numpy(dtype=float)
        + 1.15 * frame["V4"].to_numpy(dtype=float)
        - 0.90 * frame["V12"].to_numpy(dtype=float)
        + 0.70 * frame["V11"].to_numpy(dtype=float)
        + 0.85 * z_log
        + 0.55 * night
        + 0.45 * night * (z_log > 1.0).astype(float)
    )


def _intercept_for_rate(effects: np.ndarray, target_rate: float) -> float:
    low, high = -12.0, 4.0
    for _ in range(40):
        mid = (low + high) / 2.0
        mean_prob = float(np.mean(1.0 / (1.0 + np.exp(-(mid + effects)))))
        if mean_prob > target_rate:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def generate_transactions(
    n_rows: int,
    seed: int,
    target_fraud_rate: float = 0.02,
    label_noise: float = 0.002,
) -> pd.DataFrame:
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    if not 0.001 <= target_fraud_rate <= 0.2:
        raise ValueError("target_fraud_rate must be between 0.001 and 0.2")

    rng = np.random.default_rng(seed)
    components = rng.normal(0.0, 1.0, size=(n_rows, len(PCA_COLUMNS)))
    time = rng.uniform(0.0, 172800.0, size=n_rows)
    amount = np.exp(rng.normal(3.1, 1.1, size=n_rows))
    spike = rng.random(n_rows) < 0.03
    amount = np.where(spike, amount * rng.uniform(4.0, 12.0, size=n_rows), amount)
    amount = np.clip(amount, 0.0, 25000.0)

    frame = pd.DataFrame(components, columns=list(PCA_COLUMNS))
    frame.insert(0, "Amount", amount.astype(float))
    frame.insert(0, "Time", time.astype(float))
    effects = fraud_effects(frame)
    intercept = _intercept_for_rate(effects, target_fraud_rate)
    probabilities = 1.0 / (1.0 + np.exp(-(intercept + effects)))
    labels = rng.binomial(1, probabilities).astype(int)
    if label_noise > 0:
        flip = rng.random(n_rows) < label_noise
        labels = np.where(flip, 1 - labels, labels)

    frame.insert(0, ID_COLUMN, [f"syn_{seed}_{index:08d}" for index in range(n_rows)])
    frame[TARGET_COLUMN] = labels.astype(int)
    return frame
