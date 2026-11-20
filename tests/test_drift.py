from __future__ import annotations

import numpy as np
import pandas as pd

from payguard.data.generate import generate_transactions
from payguard.monitoring.drift import (
    categorical_psi,
    population_stability_index,
    retrain_decision,
    simulate_population_shift,
)


def test_psi_is_near_zero_for_identical_samples() -> None:
    sample = np.linspace(0.0, 10.0, 400)
    assert population_stability_index(sample, sample, n_bins=10) < 1e-8


def test_psi_rises_when_the_distribution_shifts() -> None:
    reference = np.random.default_rng(0).normal(0, 1, size=2000)
    shifted = reference + 2.5
    assert population_stability_index(reference, shifted, n_bins=10) > 0.2


def test_constant_feature_has_zero_psi() -> None:
    constant = np.ones(100)
    assert population_stability_index(constant, constant) == 0.0


def test_categorical_psi_detects_a_mix_change() -> None:
    reference = pd.Series(["domestic"] * 90 + ["foreign"] * 10)
    current = pd.Series(["domestic"] * 40 + ["foreign"] * 60)
    assert categorical_psi(reference, current) > 0.2
    assert categorical_psi(reference, reference) < 1e-8


def test_retrain_trigger_requires_the_configured_evidence() -> None:
    quiet = [{"feature": "amount", "psi": 0.01}, {"feature": "hour", "psi": 0.02}]
    decision = retrain_decision(
        quiet,
        prediction_psi=0.01,
        feature_psi_threshold=0.2,
        prediction_psi_threshold=0.1,
        min_features_drifted=2,
    )
    assert decision["retrain"] is False

    noisy = [{"feature": "amount", "psi": 0.4}, {"feature": "hour", "psi": 0.3}]
    decision = retrain_decision(
        noisy,
        prediction_psi=0.02,
        feature_psi_threshold=0.2,
        prediction_psi_threshold=0.1,
        min_features_drifted=2,
    )
    assert decision["retrain"] is True
    assert "amount" in decision["flagged_features"]


def test_prediction_psi_alone_can_trigger() -> None:
    decision = retrain_decision(
        [{"feature": "amount", "psi": 0.01}],
        prediction_psi=0.25,
        feature_psi_threshold=0.2,
        prediction_psi_threshold=0.1,
        min_features_drifted=2,
    )
    assert decision["retrain"] is True
    assert decision["reasons"]


def test_simulated_shift_moves_amount_and_components() -> None:
    frame = generate_transactions(500, seed=12, target_fraud_rate=0.04)
    shifted = simulate_population_shift(frame, seed=3)
    assert shifted["Amount"].mean() > frame["Amount"].mean() * 1.5
    assert shifted["V14"].mean() < frame["V14"].mean() - 3
    assert shifted["Time"].mean() > frame["Time"].mean()
