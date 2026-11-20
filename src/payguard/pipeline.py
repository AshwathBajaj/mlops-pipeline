"""Command-line pipeline: data, train, explain, drift, track."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from payguard.config import load_config
from payguard.data.generate import generate_transactions
from payguard.data.load import load_transactions
from payguard.data.validate import validate_transactions
from payguard.explain.shap_report import explain_encoded_row, global_shap
from payguard.monitoring.drift import (
    build_drift_report,
    render_drift_markdown,
    simulate_population_shift,
)
from payguard.reporting import save_calibration, save_pr_curves, save_shap_bars, to_jsonable
from payguard.schema import RAW_FEATURE_COLUMNS
from payguard.scoring import encode_frame, predict_calibrated
from payguard.tracking import log_training_run
from payguard.training.train import run_training

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _plain_row(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            payload[str(key)] = int(value)
        elif isinstance(value, (np.floating, float)):
            payload[str(key)] = round(float(value), 6)
        else:
            payload[str(key)] = value if isinstance(value, str) else str(value)
    return payload


def prepare_data(cfg: dict[str, Any]) -> pd.DataFrame:
    frame = load_transactions(cfg)
    frame = validate_transactions(frame)
    output = Path(cfg["data"]["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    logger.info(
        "validated %s rows, fraud rate %.4f, wrote %s",
        len(frame),
        float(frame["fraud"].mean()),
        output,
    )
    return frame


def _score(artifact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    _, calibrated = predict_calibrated(
        artifact["preprocessor"],
        artifact["model"],
        artifact["calibrator"],
        frame,
    )
    return calibrated


def run_drift(
    cfg: dict[str, Any], artifact: dict[str, Any], reference: pd.DataFrame
) -> dict[str, Any]:
    monitoring = cfg["monitoring"]
    rows = int(monitoring["drift_rows"])
    seed = int(cfg["seed"])
    stable = validate_transactions(
        generate_transactions(
            n_rows=rows,
            seed=seed + 101,
            target_fraud_rate=float(cfg["data"]["target_fraud_rate"]),
        )
    )
    shifted = validate_transactions(simulate_population_shift(stable, seed + 7))
    reference_sample = reference.sample(n=min(len(reference), max(rows, 1000)), random_state=seed)
    reference_scores = _score(artifact, reference_sample)
    reports = [
        build_drift_report(
            reference_sample,
            stable,
            reference_scores,
            _score(artifact, stable),
            monitoring,
            label="stable_draw",
        ),
        build_drift_report(
            reference_sample,
            shifted,
            reference_scores,
            _score(artifact, shifted),
            monitoring,
            label="shifted_draw",
        ),
    ]
    payload = {"comparisons": reports}
    json_path = Path(cfg["output"]["drift_json_path"])
    md_path = Path(cfg["output"]["drift_md_path"])
    _write_json(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_drift_markdown(reports), encoding="utf-8")
    for report in reports:
        decision = report["decision"]
        logger.info(
            "drift %s prediction PSI %.4f retrain %s",
            report["label"],
            report["prediction_psi"],
            decision["retrain"],
        )
    return payload


def _print_table(summary: dict[str, Any]) -> None:
    header = (
        f"{'model':28} {'cv_pr_auc':10} {'test_pr':10} "
        f"{'test_roc':10} {'recall':10} {'brier_cal':10}"
    )
    print(header)
    for name, metrics in summary["models"].items():
        marker = "*" if name == summary["selected_model"] else " "
        print(
            f"{marker}{name:27} "
            f"{metrics['cv_pr_auc_oof']:.4f}    "
            f"{metrics['test_pr_auc']:.4f}    "
            f"{metrics['test_roc_auc']:.4f}    "
            f"{metrics['test_recall']:.4f}    "
            f"{metrics['test_brier_calibrated']:.4f}"
        )


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    frame = prepare_data(cfg)
    training = run_training(cfg, frame)
    selected_name = training["selected_model"]
    fitted = training["artifact"]
    background_encoded = encode_frame(fitted["preprocessor"], training["reference_frame"])
    rng = np.random.default_rng(int(cfg["seed"]))
    background_count = min(100, len(background_encoded))
    background_index = rng.choice(len(background_encoded), size=background_count, replace=False)
    background = background_encoded[background_index]

    shap_summary = global_shap(
        fitted["model"],
        fitted["preprocessor"],
        training["reference_frame"],
        list(fitted["feature_names"]),
        str(fitted["model_family"]),
        sample_size=int(cfg["training"]["shap_sample_size"]),
        seed=int(cfg["seed"]),
    )
    sample_row = training["test_frame"].iloc[0]
    sample_frame = training["test_frame"].iloc[:1]
    sample_encoded = encode_frame(fitted["preprocessor"], sample_frame)
    sample_explanation = explain_encoded_row(
        fitted["model"],
        sample_encoded,
        list(fitted["feature_names"]),
        str(fitted["model_family"]),
        background,
    )
    sample_probability = float(_score(fitted, sample_frame)[0])

    plots_dir = Path(cfg["output"]["plots_dir"])
    save_pr_curves(
        training["curves"],
        float(training["dataset"]["test_fraud_rate"]),
        plots_dir / "pr_curves.png",
    )
    save_calibration(training, plots_dir / "calibration.png")
    save_shap_bars(shap_summary, plots_dir / "shap_summary.png")

    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    public_metrics = {
        "selected_model": selected_name,
        "selection_metric": training["selection_metric"],
        "precision_target": training["precision_target"],
        "dataset": training["dataset"],
        "models": training["models"],
    }
    bundle = {
        "preprocessor": fitted["preprocessor"],
        "model": fitted["model"],
        "calibrator": fitted["calibrator"],
        "threshold": fitted["threshold"],
        "feature_names": fitted["feature_names"],
        "model_family": fitted["model_family"],
        "background": background,
        "model_name": selected_name,
        "model_version": f"{selected_name}-{version}",
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": int(cfg["seed"]),
        "precision_target": training["precision_target"],
        "metrics": public_metrics,
        "shap_top_features": shap_summary["top_features"],
        "input_columns": list(RAW_FEATURE_COLUMNS),
    }
    model_path = Path(cfg["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    drift = run_drift(cfg, bundle, training["reference_frame"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    shap_path = Path(cfg["output"]["shap_path"])
    sample_request = _plain_row(sample_row)
    report = {
        "generated_at": bundle["trained_at"],
        "model_version": bundle["model_version"],
        "seed": int(cfg["seed"]),
        "data_source": cfg["data"]["source"],
        "selected_model": selected_name,
        "selection_metric": training["selection_metric"],
        "precision_target": training["precision_target"],
        "dataset": training["dataset"],
        "models": training["models"],
        "shap": shap_summary,
        "drift": drift,
        "sample_request": sample_request,
        "sample_response": {
            "fraud_probability": round(sample_probability, 6),
            "fraud_flag": bool(sample_probability >= float(fitted["threshold"])),
            "threshold": float(fitted["threshold"]),
            "model_name": selected_name,
            "model_version": bundle["model_version"],
        },
        "sample_explanation": sample_explanation,
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "pr_curves": str(plots_dir / "pr_curves.png"),
            "calibration": str(plots_dir / "calibration.png"),
            "shap": str(plots_dir / "shap_summary.png"),
        },
    }
    _write_json(shap_path, shap_summary)
    _write_json(Path("reports/sample_request.json"), sample_request)
    _write_json(Path("reports/sample_response.json"), report["sample_response"])
    _write_json(Path("reports/sample_explanation.json"), sample_explanation)
    _write_json(metrics_path, report)

    mlflow_info = log_training_run(
        cfg,
        training,
        model_path,
        [
            metrics_path,
            shap_path,
            Path(cfg["output"]["drift_json_path"]),
            plots_dir / "pr_curves.png",
            plots_dir / "calibration.png",
            plots_dir / "shap_summary.png",
        ],
        input_example=sample_request,
    )
    report["mlflow"] = mlflow_info
    bundle["mlflow"] = mlflow_info
    bundle["metrics"] = public_metrics
    joblib.dump(bundle, model_path)
    _write_json(metrics_path, report)
    if mlflow_info.get("selected_run_id"):
        MlflowClient().log_artifact(mlflow_info["selected_run_id"], str(metrics_path))
    _print_table(training)
    logger.info("wrote %s", metrics_path)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the PayGuard fraud-detection pipeline")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--stage", default="all", choices=["all", "data", "train", "drift", "explain"]
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.stage == "data":
        prepare_data(cfg)
        return

    if args.stage in {"all", "train"}:
        run_pipeline(cfg)
        return

    model_path = Path(cfg["output"]["model_path"])
    if not model_path.exists():
        raise SystemExit(f"missing {model_path}; run with --stage train first")
    bundle = joblib.load(model_path)
    frame = prepare_data(cfg)
    features = frame.loc[:, list(RAW_FEATURE_COLUMNS)]
    if args.stage == "drift":
        run_drift(cfg, bundle, features)
        return
    shap_summary = global_shap(
        bundle["model"],
        bundle["preprocessor"],
        features,
        list(bundle["feature_names"]),
        str(bundle["model_family"]),
        sample_size=int(cfg["training"]["shap_sample_size"]),
        seed=int(cfg["seed"]),
    )
    _write_json(Path(cfg["output"]["shap_path"]), shap_summary)
    save_shap_bars(shap_summary, Path(cfg["output"]["plots_dir"]) / "shap_summary.png")


if __name__ == "__main__":
    main()
