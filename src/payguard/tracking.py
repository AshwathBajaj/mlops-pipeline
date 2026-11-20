"""MLflow experiment tracking and model registry."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from payguard.serving.pyfunc import PayGuardPyfunc

logger = logging.getLogger(__name__)


def configure_mlflow(cfg: dict[str, Any]) -> None:
    mlflow_cfg = cfg["mlflow"]
    tracking_uri = str(mlflow_cfg["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    experiment_name = str(mlflow_cfg["experiment_name"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    artifact_root = Path(mlflow_cfg["artifact_location"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
    elif experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            numeric[key] = float(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric[key] = float(value)
    return numeric


def log_training_run(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    bundle_path: Path,
    extra_artifacts: list[Path],
    input_example: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configure_mlflow(cfg)
    selected = summary["selected_model"]
    registered_name = str(cfg["mlflow"]["registered_model_name"])
    run_ids: dict[str, str] = {}
    selected_run_id: str | None = None
    selected_model_uri: str | None = None
    registry_version: str | None = None

    for name, metrics in summary["models"].items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(
                {
                    "model_name": name,
                    "selected": "true" if name == selected else "false",
                    "pipeline": "payguard",
                }
            )
            mlflow.log_params(
                {
                    "seed": int(cfg["seed"]),
                    "n_rows": int(summary["dataset"]["n_rows"]),
                    "cv_folds": int(cfg["training"]["cv_folds"]),
                    "precision_target": float(cfg["training"]["precision_target"]),
                    "data_source": str(cfg["data"]["source"]),
                    **{
                        f"model.{key}": value
                        for key, value in cfg["training"]["models"][name].items()
                    },
                }
            )
            mlflow.log_metrics(_numeric_metrics(metrics))
            mlflow.log_metrics(
                {
                    "train_rows": float(summary["dataset"]["n_train"]),
                    "test_rows": float(summary["dataset"]["n_test"]),
                    "fraud_rate": float(summary["dataset"]["fraud_rate"]),
                }
            )
            if name == selected:
                for artifact in extra_artifacts:
                    if artifact.exists() and artifact.is_file():
                        mlflow.log_artifact(str(artifact))
                example = pd.DataFrame([input_example]) if input_example is not None else None
                model_info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=PayGuardPyfunc(),
                    artifacts={"bundle": str(bundle_path)},
                    input_example=example,
                    pip_requirements=[
                        "pandas",
                        "numpy",
                        "scikit-learn",
                        "xgboost",
                        "joblib",
                    ],
                )
                selected_run_id = run.info.run_id
                selected_model_uri = model_info.model_uri
            run_ids[name] = run.info.run_id

    if selected_run_id is not None and selected_model_uri is not None:
        try:
            version = mlflow.register_model(selected_model_uri, registered_name)
            registry_version = str(version.version)
            logger.info("registered %s version %s", registered_name, registry_version)
        except Exception as exc:
            logger.warning("model registry call failed: %s", exc)

    return {
        "selected_run_id": selected_run_id,
        "model_uri": selected_model_uri,
        "registry_model": registered_name,
        "registry_version": registry_version,
        "run_ids": run_ids,
    }
