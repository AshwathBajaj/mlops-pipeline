"""Model factories. Class imbalance is handled with weights, not resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def positive_weight(y: np.ndarray) -> float:
    positives = max(int(np.sum(y)), 1)
    negatives = int(len(y) - np.sum(y))
    return float(negatives / positives)


def build_model(name: str, model_cfg: dict[str, Any], y_train: np.ndarray, seed: int) -> object:
    if name == "logistic_regression":
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 400)),
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(model_cfg.get("max_iter", 100)),
            learning_rate=float(model_cfg.get("learning_rate", 0.08)),
            max_depth=int(model_cfg.get("max_depth", 6)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 30)),
            class_weight="balanced",
            random_state=seed,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 150)),
            max_depth=int(model_cfg.get("max_depth", 4)),
            learning_rate=float(model_cfg.get("learning_rate", 0.08)),
            subsample=float(model_cfg.get("subsample", 0.9)),
            colsample_bytree=float(model_cfg.get("colsample_bytree", 0.9)),
            min_child_weight=float(model_cfg.get("min_child_weight", 1)),
            scale_pos_weight=positive_weight(y_train),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
            tree_method="hist",
            verbosity=0,
        )
    raise ValueError(f"unknown model: {name}")
