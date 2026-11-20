"""PSI and Kolmogorov-Smirnov drift checks, plus a retrain trigger.

PSI bins are the quantiles of the reference sample. A value under 0.10 is the
usual "stable" band, 0.10 to 0.20 is a watch, and above 0.20 is a shift worth
investigating. Those cutoffs are heuristics, not laws; the config file owns them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from payguard.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    breaks = np.unique(np.quantile(reference, quantiles))
    if len(breaks) < 3:
        return 0.0
    reference_counts = np.histogram(reference, bins=breaks)[0].astype(float)
    current_counts = np.histogram(current, bins=breaks)[0].astype(float)
    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / max(current_counts.sum(), 1.0)
    reference_share = np.clip(reference_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)
    reference_share = reference_share / reference_share.sum()
    current_share = current_share / current_share.sum()
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    levels = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    if not levels:
        return 0.0
    reference_share = reference.astype(str).value_counts(normalize=True)
    current_share = current.astype(str).value_counts(normalize=True)
    expected = np.array([float(reference_share.get(level, 0.0)) for level in levels])
    actual = np.array([float(current_share.get(level, 0.0)) for level in levels])
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    if (
        np.unique(reference).size == 1
        and np.unique(current).size == 1
        and reference[0] == current[0]
    ):
        return 0.0
    return float(ks_2samp(reference, current, method="asymp").statistic)


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in NUMERIC_COLUMNS:
        rows.append(
            {
                "feature": column,
                "kind": "numeric",
                "psi": population_stability_index(
                    reference[column].to_numpy(),
                    current[column].to_numpy(),
                    n_bins=n_bins,
                ),
                "ks": ks_statistic(reference[column].to_numpy(), current[column].to_numpy()),
            }
        )
    for column in CATEGORICAL_COLUMNS:
        rows.append(
            {
                "feature": column,
                "kind": "categorical",
                "psi": categorical_psi(reference[column], current[column]),
                "ks": None,
            }
        )
    return rows


def retrain_decision(
    feature_rows: list[dict[str, Any]],
    prediction_psi: float,
    *,
    feature_psi_threshold: float,
    prediction_psi_threshold: float,
    min_features_drifted: int,
) -> dict[str, Any]:
    flagged = [row["feature"] for row in feature_rows if float(row["psi"]) >= feature_psi_threshold]
    reasons: list[str] = []
    if len(flagged) >= min_features_drifted:
        reasons.append(
            f"{len(flagged)} features have PSI >= {feature_psi_threshold}: {', '.join(flagged)}"
        )
    if prediction_psi >= prediction_psi_threshold:
        reasons.append(f"prediction PSI {prediction_psi:.4f} >= {prediction_psi_threshold}")
    return {
        "retrain": bool(reasons),
        "reasons": reasons,
        "flagged_features": flagged,
        "prediction_psi": float(prediction_psi),
    }


def simulate_population_shift(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A deliberate shift used to prove the trigger fires.

    Production would pass a live scoring window here. This function only exists
    so the offline demo has a positive control.
    """
    shifted = frame.copy()
    rng = np.random.default_rng(seed)
    shifted["amount"] = shifted["amount"].astype(float) * rng.uniform(1.7, 2.6, size=len(shifted))
    shifted["merchant_risk_score"] = np.clip(
        shifted["merchant_risk_score"].astype(float) + 0.28, 0.0, 1.0
    )
    shifted["distance_from_home_km"] = shifted["distance_from_home_km"].astype(float) + 380.0
    shifted["txn_count_24h"] = np.clip(shifted["txn_count_24h"].astype(int) + 4, 0, 100)
    channel_flip = rng.random(len(shifted)) < 0.40
    shifted.loc[channel_flip, "channel"] = "online"
    country_flip = rng.random(len(shifted)) < 0.35
    shifted.loc[country_flip, "country_match"] = "foreign"
    device_flip = rng.random(len(shifted)) < 0.25
    shifted.loc[device_flip, "device_trust"] = "new"
    shifted["hour"] = (shifted["hour"].astype(int) + 8) % 24
    return shifted


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    monitoring_cfg: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    n_bins = int(monitoring_cfg.get("psi_bins", 10))
    features = feature_drift(reference, current, n_bins=n_bins)
    prediction_psi = population_stability_index(reference_scores, current_scores, n_bins=n_bins)
    prediction_ks = ks_statistic(reference_scores, current_scores)
    decision = retrain_decision(
        features,
        prediction_psi,
        feature_psi_threshold=float(monitoring_cfg["feature_psi_threshold"]),
        prediction_psi_threshold=float(monitoring_cfg["prediction_psi_threshold"]),
        min_features_drifted=int(monitoring_cfg["min_features_drifted"]),
    )
    return {
        "label": label,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "features": features,
        "prediction_psi": prediction_psi,
        "prediction_ks": prediction_ks,
        "decision": decision,
    }


def render_drift_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Drift report",
        "",
        "PSI uses reference quantiles. KS is reported for numeric features and for the score.",
        "The retrain flag is the rule in `configs/train.yaml`, not a human review.",
        "",
    ]
    for report in reports:
        decision = report["decision"]
        lines.append(f"## {report['label']}")
        lines.append("")
        lines.append(f"- Reference rows: {report['n_reference']}")
        lines.append(f"- Current rows: {report['n_current']}")
        lines.append(f"- Prediction PSI: {report['prediction_psi']:.4f}")
        lines.append(f"- Prediction KS: {report['prediction_ks']:.4f}")
        lines.append(f"- Retrain: {str(decision['retrain']).lower()}")
        if decision["reasons"]:
            for reason in decision["reasons"]:
                lines.append(f"- Reason: {reason}")
        else:
            lines.append("- Reason: none")
        lines.append("")
        lines.append("| Feature | Kind | PSI | KS |")
        lines.append("| --- | --- | ---: | ---: |")
        for row in report["features"]:
            ks_value = "" if row["ks"] is None else f"{row['ks']:.4f}"
            lines.append(f"| {row['feature']} | {row['kind']} | {row['psi']:.4f} | {ks_value} |")
        lines.append("")
    return "\n".join(lines)
