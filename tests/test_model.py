from __future__ import annotations

import pandas as pd

from payguard.scoring import predict_calibrated
from payguard.tracking import log_training_run


def _row(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([payload])


def test_selected_model_beats_the_prevalence_baseline(trained: dict) -> None:
    summary = trained["summary"]
    selected = summary["selected_model"]
    metrics = summary["models"][selected]
    prevalence = summary["dataset"]["prevalence_baseline_pr_auc"]
    assert metrics["cv_pr_auc_oof"] > prevalence
    assert metrics["test_pr_auc"] > prevalence
    assert 0.0 <= metrics["test_brier_calibrated"] <= 1.0
    assert 0.0 <= metrics["oof_threshold"] <= 1.0
    for name, result in summary["models"].items():
        assert result["test_pr_auc"] > prevalence, name


def test_suspicious_row_scores_above_a_calm_row(
    trained: dict,
    benign_transaction: dict,
    suspicious_transaction: dict,
) -> None:
    bundle = trained["bundle"]
    _, calm = predict_calibrated(
        bundle["preprocessor"],
        bundle["model"],
        bundle["calibrator"],
        _row(benign_transaction),
    )
    _, risky = predict_calibrated(
        bundle["preprocessor"],
        bundle["model"],
        bundle["calibrator"],
        _row(suspicious_transaction),
    )
    assert float(risky[0]) > float(calm[0])


def test_mlflow_logs_and_registers_the_selected_model(trained: dict) -> None:
    info = log_training_run(
        trained["cfg"],
        trained["summary"],
        trained["model_path"],
        [],
    )
    assert info["selected_run_id"]
    assert info["registry_version"]
    assert set(info["run_ids"]) == set(trained["summary"]["models"])
