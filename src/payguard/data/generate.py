"""Deterministic synthetic card transactions with a known fraud process.

The generator is the offline source of truth for CI and for the metrics in the
README. Labels come from a logistic process with a few interactions (foreign
card-not-present, new device at night, amount spikes on online purchases) plus
a small amount of label noise. An intercept search holds the mean fraud
probability near the requested rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from payguard.schema import (
    CHANNELS,
    COUNTRY_MATCH,
    DEVICE_TRUST,
    ID_COLUMN,
    MERCHANT_CATEGORIES,
    TARGET_COLUMN,
)

MERCHANT_PROBABILITIES = np.array([0.28, 0.12, 0.18, 0.12, 0.10, 0.04, 0.13, 0.03])
CHANNEL_PROBABILITIES = np.array([0.40, 0.28, 0.10, 0.22])
COUNTRY_PROBABILITIES = np.array([0.92, 0.08])
DEVICE_PROBABILITIES = np.array([0.88, 0.08, 0.04])


def _sample_labels(
    rng: np.random.Generator, choices: tuple[str, ...], probs: np.ndarray, n: int
) -> np.ndarray:
    index = rng.choice(len(choices), size=n, p=probs)
    return np.asarray(choices, dtype=object)[index]


def fraud_effects(frame: pd.DataFrame) -> np.ndarray:
    """Linear predictor without the intercept. Used by the generator and tests."""
    amount = frame["amount"].to_numpy(dtype=float)
    log_amount = np.log1p(amount)
    z_log = (log_amount - 3.6) / 1.1
    hour = frame["hour"].to_numpy(dtype=int)
    night = ((hour <= 5) | (hour >= 23)).astype(float)
    foreign = (frame["country_match"].to_numpy() == "foreign").astype(float)
    online = (frame["channel"].to_numpy() == "online").astype(float)
    new_device = (frame["device_trust"].to_numpy() == "new").astype(float)
    shared = (frame["device_trust"].to_numpy() == "shared").astype(float)
    high_velocity = (frame["txn_count_24h"].to_numpy(dtype=float) >= 6).astype(float)
    ratio = amount / (frame["avg_amount_30d"].to_numpy(dtype=float) + 1.0)
    high_ratio = (ratio > 4.0).astype(float)
    far = (frame["distance_from_home_km"].to_numpy(dtype=float) > 400).astype(float)
    risky = frame["merchant_risk_score"].to_numpy(dtype=float)
    category = frame["merchant_category"].to_numpy()
    gambling = (category == "gambling").astype(float)
    jewelry = (category == "jewelry").astype(float)
    young = (frame["account_age_days"].to_numpy(dtype=float) < 30).astype(float)
    failed = (frame["failed_attempts_24h"].to_numpy(dtype=float) >= 2).astype(float)
    high_util = (frame["utilization"].to_numpy(dtype=float) > 0.9).astype(float)
    return (
        1.10 * z_log
        + 1.20 * foreign
        + 0.90 * online
        + 1.30 * new_device
        + 0.65 * shared
        + 0.80 * night
        + 0.85 * high_velocity
        + 1.00 * high_ratio
        + 0.75 * far
        + 1.15 * risky
        + 0.70 * gambling
        + 0.50 * jewelry
        + 0.55 * young
        + 0.80 * failed
        + 0.40 * high_util
        + 1.10 * foreign * online
        + 0.90 * new_device * night
        + 0.65 * high_ratio * online
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
    log_typical = rng.normal(3.55, 0.65, size=n_rows)
    amount = np.exp(log_typical)
    spike = rng.random(n_rows) < 0.045
    amount = np.where(spike, amount * rng.uniform(4.0, 12.0, size=n_rows), amount)
    amount = np.clip(amount, 0.5, 20000.0)
    avg_amount = np.exp(log_typical + rng.normal(0.0, 0.22, size=n_rows))
    avg_amount = np.clip(avg_amount, 1.0, 8000.0)

    hour = np.clip(np.rint(rng.normal(14.0, 3.2, size=n_rows)), 0, 23).astype(int)
    night_draw = rng.random(n_rows) < 0.11
    hour = np.where(night_draw, rng.integers(0, 6, size=n_rows), hour).astype(int)

    account_age = np.clip(rng.lognormal(mean=6.1, sigma=0.85, size=n_rows), 1, 9000).astype(int)
    credit_limit = rng.choice(
        np.array([1000.0, 2500.0, 5000.0, 10000.0, 20000.0]),
        size=n_rows,
        p=np.array([0.12, 0.22, 0.30, 0.24, 0.12]),
    )
    utilization = np.clip(rng.beta(2.0, 5.0, size=n_rows), 0.0, 0.99)
    txn_count = np.clip(rng.poisson(1.15, size=n_rows), 0, 25).astype(int)
    days_since = np.clip(rng.exponential(4.5, size=n_rows), 0.0, 180.0)
    distance = rng.exponential(35.0, size=n_rows)
    far_tail = rng.random(n_rows) < 0.05
    distance = np.where(far_tail, distance + rng.uniform(400.0, 1800.0, size=n_rows), distance)
    merchant_risk = np.clip(rng.beta(1.4, 8.0, size=n_rows), 0.0, 1.0)
    failed = np.clip(rng.poisson(0.18, size=n_rows), 0, 12).astype(int)

    frame = pd.DataFrame(
        {
            "amount": amount.astype(float),
            "hour": hour.astype(int),
            "account_age_days": account_age.astype(int),
            "credit_limit": credit_limit.astype(float),
            "utilization": utilization.astype(float),
            "txn_count_24h": txn_count.astype(int),
            "avg_amount_30d": avg_amount.astype(float),
            "days_since_last_txn": days_since.astype(float),
            "distance_from_home_km": distance.astype(float),
            "merchant_risk_score": merchant_risk.astype(float),
            "failed_attempts_24h": failed.astype(int),
            "merchant_category": _sample_labels(
                rng, MERCHANT_CATEGORIES, MERCHANT_PROBABILITIES, n_rows
            ),
            "channel": _sample_labels(rng, CHANNELS, CHANNEL_PROBABILITIES, n_rows),
            "country_match": _sample_labels(rng, COUNTRY_MATCH, COUNTRY_PROBABILITIES, n_rows),
            "device_trust": _sample_labels(rng, DEVICE_TRUST, DEVICE_PROBABILITIES, n_rows),
        }
    )

    effects = fraud_effects(frame)
    intercept = _intercept_for_rate(effects, target_fraud_rate)
    probabilities = 1.0 / (1.0 + np.exp(-(intercept + effects)))
    labels = rng.binomial(1, probabilities).astype(int)
    if label_noise > 0:
        flip = rng.random(n_rows) < label_noise
        labels = np.where(flip, 1 - labels, labels)

    frame.insert(0, ID_COLUMN, [f"txn_{seed}_{i:08d}" for i in range(n_rows)])
    frame[TARGET_COLUMN] = labels.astype(int)
    return frame
