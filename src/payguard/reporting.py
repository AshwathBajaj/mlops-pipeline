"""Plots and JSON-safe report payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return round(number, 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def save_pr_curves(curves: dict[str, dict[str, Any]], prevalence: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for name, payload in curves.items():
        precision, recall, _ = precision_recall_curve(payload["y_test"], payload["test_raw"])
        average_precision = average_precision_score(payload["y_test"], payload["test_raw"])
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP {average_precision:.3f})")
    ax.axhline(
        prevalence, color="0.45", linestyle="--", linewidth=1, label=f"prevalence {prevalence:.3f}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Held-out precision-recall")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_calibration(summary: dict[str, Any], path: Path) -> None:
    name = summary["selected_model"]
    payload = summary["curves"][name]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label, scores in (("raw", payload["test_raw"]), ("platt", payload["test_calibrated"])):
        observed, predicted = calibration_curve(
            payload["y_test"],
            scores,
            n_bins=8,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(f"Reliability of {name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_shap_bars(shap_summary: dict[str, Any], path: Path) -> None:
    rows = list(reversed(shap_summary["top_features"][:10]))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(
        [row["feature"] for row in rows], [row["mean_abs_shap"] for row in rows], color="#1f4e79"
    )
    ax.set_xlabel("Mean |SHAP| (model margin)")
    ax.set_title("Global explanations")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
