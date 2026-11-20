"""Compare models with out-of-fold scores and fit the one we would ship."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from payguard.features.build import make_preprocessor
from payguard.schema import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from payguard.scoring import apply_platt, encode_frame, fit_platt, raw_probabilities
from payguard.training.metrics import (
    ensure_both_classes,
    operating_point,
    ranking_metrics,
    recall_at_precision,
)
from payguard.training.models import build_model

logger = logging.getLogger(__name__)


def evaluate_candidate(
    name: str,
    model_cfg: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    seed: int,
    n_folds: int,
    precision_target: float,
) -> dict[str, Any]:
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_raw = np.zeros(len(y_train), dtype=float)
    fold_pr_auc: list[float] = []
    fold_roc_auc: list[float] = []

    for fold_id, (train_index, valid_index) in enumerate(folds.split(X_train, y_train)):
        preprocessor = make_preprocessor()
        y_fold = y_train[train_index]
        preprocessor.fit(X_train.iloc[train_index])
        model = build_model(name, model_cfg, y_fold, seed + fold_id)
        encoded_train = encode_frame(preprocessor, X_train.iloc[train_index])
        model.fit(encoded_train, y_fold)
        valid_scores = raw_probabilities(
            model, encode_frame(preprocessor, X_train.iloc[valid_index])
        )
        oof_raw[valid_index] = valid_scores
        valid_labels = y_train[valid_index]
        fold_rank = ranking_metrics(valid_labels, valid_scores)
        fold_pr_auc.append(fold_rank["pr_auc"])
        fold_roc_auc.append(fold_rank["roc_auc"])
        logger.info(
            "%s fold %s PR-AUC %.4f ROC-AUC %.4f",
            name,
            fold_id,
            fold_rank["pr_auc"],
            fold_rank["roc_auc"],
        )

    calibrator = fit_platt(oof_raw, y_train)
    oof_calibrated = apply_platt(calibrator, oof_raw)
    threshold_info = recall_at_precision(y_train, oof_calibrated, precision_target)

    preprocessor = make_preprocessor()
    preprocessor.fit(X_train)
    model = build_model(name, model_cfg, y_train, seed)
    model.fit(encode_frame(preprocessor, X_train), y_train)
    test_raw = raw_probabilities(model, encode_frame(preprocessor, X_test))
    test_calibrated = apply_platt(calibrator, test_raw)
    raw_rank = ranking_metrics(y_test, test_raw)
    calibrated_rank = ranking_metrics(y_test, test_calibrated)
    oof_rank = ranking_metrics(y_train, oof_raw)
    oof_cal_rank = ranking_metrics(y_train, oof_calibrated)
    test_point = operating_point(y_test, test_calibrated, float(threshold_info["threshold"]))
    oof_point = operating_point(y_train, oof_calibrated, float(threshold_info["threshold"]))
    feature_names = [str(feature) for feature in preprocessor.get_feature_names_out()]

    return {
        "name": name,
        "cv_pr_auc_mean": float(np.nanmean(fold_pr_auc)),
        "cv_pr_auc_std": float(np.nanstd(fold_pr_auc)),
        "cv_roc_auc_mean": float(np.nanmean(fold_roc_auc)),
        "cv_pr_auc_oof": oof_rank["pr_auc"],
        "cv_roc_auc_oof": oof_rank["roc_auc"],
        "cv_brier_raw": oof_rank["brier"],
        "cv_brier_calibrated": oof_cal_rank["brier"],
        "cv_ece_raw": oof_rank["ece"],
        "cv_ece_calibrated": oof_cal_rank["ece"],
        "precision_target": precision_target,
        "precision_target_achieved": bool(threshold_info["achieved"]),
        "oof_threshold": float(threshold_info["threshold"]),
        "oof_precision": float(oof_point["precision"]),
        "oof_recall": float(oof_point["recall"]),
        "oof_flag_rate": float(oof_point["flag_rate"]),
        "oof_tp": int(oof_point["tp"]),
        "oof_fp": int(oof_point["fp"]),
        "test_pr_auc": raw_rank["pr_auc"],
        "test_roc_auc": raw_rank["roc_auc"],
        "test_brier_raw": raw_rank["brier"],
        "test_brier_calibrated": calibrated_rank["brier"],
        "test_ece_raw": raw_rank["ece"],
        "test_ece_calibrated": calibrated_rank["ece"],
        "test_precision": test_point["precision"],
        "test_recall": test_point["recall"],
        "test_flag_rate": test_point["flag_rate"],
        "test_tp": test_point["tp"],
        "test_fp": test_point["fp"],
        "test_fn": test_point["fn"],
        "test_tn": test_point["tn"],
        "artifact": {
            "preprocessor": preprocessor,
            "model": model,
            "calibrator": calibrator,
            "threshold": float(threshold_info["threshold"]),
            "feature_names": feature_names,
            "model_family": "linear" if name == "logistic_regression" else "tree",
        },
        "curves": {
            "y_test": y_test,
            "test_raw": test_raw,
            "test_calibrated": test_calibrated,
        },
    }


def run_training(cfg: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    seed = int(cfg["seed"])
    training_cfg = cfg["training"]
    features = frame.loc[:, list(RAW_FEATURE_COLUMNS)].copy()
    labels = frame[TARGET_COLUMN].to_numpy(dtype=int)
    ensure_both_classes(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=float(cfg["split"]["test_size"]),
        stratify=labels,
        random_state=seed,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    candidates: dict[str, Any] = {}
    for name, model_cfg in training_cfg["models"].items():
        logger.info("training %s", name)
        candidates[name] = evaluate_candidate(
            name,
            model_cfg,
            X_train,
            y_train,
            X_test,
            y_test,
            seed=seed,
            n_folds=int(training_cfg["cv_folds"]),
            precision_target=float(training_cfg["precision_target"]),
        )

    selected = max(candidates, key=lambda name: candidates[name]["cv_pr_auc_oof"])
    logger.info("selected %s by pooled out-of-fold PR-AUC", selected)

    public_models = {}
    curves = {}
    for name, result in candidates.items():
        public = {key: value for key, value in result.items() if key not in {"artifact", "curves"}}
        public_models[name] = public
        curves[name] = result["curves"]

    return {
        "selected_model": selected,
        "selection_metric": "cv_pr_auc_oof",
        "precision_target": float(training_cfg["precision_target"]),
        "dataset": {
            "n_rows": int(len(frame)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_fraud_rate": float(y_train.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "fraud_rate": float(labels.mean()),
            "prevalence_baseline_pr_auc": float(y_test.mean()),
        },
        "models": public_models,
        "artifact": candidates[selected]["artifact"],
        "reference_frame": X_train,
        "reference_labels": y_train,
        "test_frame": X_test,
        "curves": curves,
        "y_test": y_test,
    }
