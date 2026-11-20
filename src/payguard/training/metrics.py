"""Ranking, operating-point, and calibration metrics for a rare positive class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def ensure_both_classes(y: np.ndarray) -> None:
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        raise ValueError(f"training labels must contain both classes, found {sorted(classes)}")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (y_prob >= edges[index]) & (y_prob <= edges[index + 1])
        else:
            mask = (y_prob >= edges[index]) & (y_prob < edges[index + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / len(y_true)) * abs(
            float(y_true[mask].mean()) - float(y_prob[mask].mean())
        )
    return float(error)


def recall_at_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    precision_target: float,
) -> dict[str, float | bool]:
    """Highest recall whose precision is at least the target.

    The threshold is chosen only on the scores passed in. Callers pass
    out-of-fold scores, then freeze the threshold for the test set.
    If the target is unreachable, the highest-precision point is returned and
    ``achieved`` is false.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return {"recall": 0.0, "precision": 0.0, "threshold": 1.0, "achieved": False}

    precision = precision[:-1]
    recall = recall[:-1]
    eligible = np.flatnonzero(precision >= precision_target)
    if len(eligible) == 0:
        index = int(np.argmax(precision))
        return {
            "recall": float(recall[index]),
            "precision": float(precision[index]),
            "threshold": float(thresholds[index]),
            "achieved": False,
        }

    best_recall = float(recall[eligible].max())
    tied = eligible[np.isclose(recall[eligible], best_recall)]
    index = int(tied[np.argmax(thresholds[tied])])
    return {
        "recall": float(recall[index]),
        "precision": float(precision[index]),
        "threshold": float(thresholds[index]),
        "achieved": True,
    }


def operating_point(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(y_prob) >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    tn = int(np.sum(~flagged & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "flag_rate": float(flagged.mean()) if len(flagged) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
