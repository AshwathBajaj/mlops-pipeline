from __future__ import annotations

import numpy as np

from payguard.training.metrics import (
    expected_calibration_error,
    operating_point,
    recall_at_precision,
)


def test_recall_at_precision_picks_the_highest_recall() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7])
    chosen = recall_at_precision(labels, scores, precision_target=0.9)
    assert chosen["achieved"] is True
    assert chosen["threshold"] == 0.7
    assert chosen["recall"] == 1.0
    assert chosen["precision"] == 1.0


def test_recall_at_precision_reports_when_the_target_is_impossible() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.4, 0.3])
    chosen = recall_at_precision(labels, scores, precision_target=0.95)
    assert chosen["achieved"] is False
    assert chosen["precision"] <= 0.5 + 1e-9


def test_operating_point_counts() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    point = operating_point(labels, scores, threshold=0.5)
    assert point["tp"] == 1
    assert point["fp"] == 1
    assert point["fn"] == 1
    assert point["tn"] == 1
    assert point["precision"] == 0.5
    assert point["recall"] == 0.5
    assert point["flag_rate"] == 0.5


def test_ece_is_zero_for_perfectly_bucketed_probabilities() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, scores, n_bins=2) == 0.0
