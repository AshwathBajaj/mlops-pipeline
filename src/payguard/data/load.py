"""Load the ULB credit-card table, a CSV, or the synthetic fallback."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from payguard.data.generate import generate_transactions
from payguard.schema import (
    ALL_COLUMNS,
    DATASET_CITATION,
    DATASET_LICENCE,
    DATASET_ORIGIN,
    ID_COLUMN,
    OPENML_DATA_ID,
    OPENML_NAME,
    OPENML_PARQUET_URL,
    OPENML_URL,
    PCA_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the OpenML column names and return the internal contract."""
    table = frame.copy()
    if table.index.name == "Time" and "Time" not in table.columns:
        table = table.reset_index()
    table = table.rename(columns={"Class": TARGET_COLUMN, "class": TARGET_COLUMN})
    if TARGET_COLUMN not in table.columns:
        raise ValueError("missing fraud label (expected 'fraud' or 'Class')")
    required = ["Time", "Amount", *PCA_COLUMNS]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    table[TARGET_COLUMN] = pd.to_numeric(table[TARGET_COLUMN], errors="raise").astype(int)
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
    if ID_COLUMN not in table.columns:
        table.insert(0, ID_COLUMN, [f"txn_{index:06d}" for index in range(len(table))])
    else:
        table[ID_COLUMN] = table[ID_COLUMN].astype(str)
    return table.loc[:, list(ALL_COLUMNS)].reset_index(drop=True)


def dataset_provenance(source: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Facts a report can quote. Counts come from the frame that was loaded."""
    fraud_count = int(frame[TARGET_COLUMN].sum())
    card = {
        "source": source,
        "n_rows": int(len(frame)),
        "n_features": len(PCA_COLUMNS) + 2,
        "fraud_count": fraud_count,
        "legitimate_count": int(len(frame) - fraud_count),
        "feature_names": ["Time", "Amount", *PCA_COLUMNS],
    }
    if source == "openml":
        card.update(
            {
                "name": OPENML_NAME,
                "openml_id": OPENML_DATA_ID,
                "openml_url": OPENML_URL,
                "licence": DATASET_LICENCE,
                "origin": DATASET_ORIGIN,
                "citation": DATASET_CITATION,
                "distribution": (
                    "OpenML dataset 1597 parquet file "
                    "https://data.openml.org/datasets/0000/1597/dataset_1597.pq. "
                    "sklearn fetch_openml is attempted if that download fails."
                ),
            }
        )
    elif source == "csv":
        card.update(
            {
                "name": OPENML_NAME,
                "openml_id": OPENML_DATA_ID,
                "licence": DATASET_LICENCE,
                "origin": DATASET_ORIGIN,
                "citation": DATASET_CITATION,
                "note": "Loaded from a local CSV with the OpenML creditcard columns.",
            }
        )
    else:
        card.update(
            {
                "name": "synthetic-creditcard-schema",
                "licence": "project-generated",
                "origin": "payguard.data.generate. Not the ULB table.",
                "note": "Same columns as OpenML 1597, drawn from the fallback generator.",
            }
        )
    return card


def _synthetic_from_config(data_cfg: dict[str, Any]) -> pd.DataFrame:
    return generate_transactions(
        n_rows=int(data_cfg.get("n_rows", 2000)),
        seed=int(data_cfg.get("seed", 42)),
        target_fraud_rate=float(data_cfg.get("target_fraud_rate", 0.02)),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return normalize_transactions(pd.read_csv(path))


def fetch_openml_frame(data_id: int, cache_path: Path | None = None) -> pd.DataFrame:
    """Load OpenML creditcard (id 1597).

    sklearn's ``fetch_openml`` currently rejects the ARFF mirror because the
    file's md5 does not match the dataset description. The parquet file listed
    on that same OpenML record downloads cleanly, so that is the path we use.
    ``fetch_openml`` is the second attempt.
    """
    destination = cache_path or Path("data/raw/creditcard.pq")
    if not (destination.exists() and destination.stat().st_size > 1_000_000):
        download_csv(OPENML_PARQUET_URL, destination, timeout_seconds=180)
    frame = pd.read_parquet(destination)
    normalized = normalize_transactions(frame)
    if len(normalized) < 1000:
        raise ValueError(f"OpenML parquet at {destination} has only {len(normalized)} rows")
    return normalized


def fetch_openml_via_sklearn(data_id: int) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    return normalize_transactions(bunch.frame)


def download_csv(url: str, destination: Path, timeout_seconds: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "payguard/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        destination.write_bytes(response.read())


def _fallback(data_cfg: dict[str, Any], reason: Exception) -> pd.DataFrame:
    fallback = data_cfg.get("fallback_csv")
    if fallback and Path(fallback).exists():
        logger.warning("download failed (%s); using fallback CSV %s", reason, fallback)
        return _read_csv(Path(fallback))
    logger.warning("download failed (%s); using the synthetic generator", reason)
    return _synthetic_from_config(data_cfg)


def load_transactions(cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = dict(cfg["data"])
    data_cfg.setdefault("seed", cfg.get("seed", 42))
    source = data_cfg["source"]

    if source == "synthetic":
        frame = _synthetic_from_config(data_cfg)
        logger.info("generated %s synthetic transactions", len(frame))
        return frame

    if source == "csv":
        path = Path(data_cfg["path"])
        frame = _read_csv(path)
        logger.info("loaded %s rows from %s", len(frame), path)
        return frame

    if source == "openml":
        data_id = int(data_cfg.get("openml_id", OPENML_DATA_ID))
        cache = Path(data_cfg.get("cache_path", "data/raw/creditcard.pq"))
        try:
            frame = fetch_openml_frame(data_id, cache_path=cache)
            logger.info("loaded OpenML %s (%s rows) from the parquet mirror", data_id, len(frame))
            return frame
        except (OSError, ValueError, urllib.error.URLError, TimeoutError) as exc:
            logger.warning("parquet download failed (%s); trying sklearn fetch_openml", exc)
            try:
                frame = fetch_openml_via_sklearn(data_id)
                logger.info("loaded OpenML %s via sklearn (%s rows)", data_id, len(frame))
                return frame
            except (OSError, ValueError, urllib.error.URLError, TimeoutError) as second:
                return _fallback(data_cfg, second)

    if source == "url":
        destination = Path(data_cfg.get("download_path", "data/raw/downloaded.csv"))
        try:
            download_csv(
                data_cfg["url"],
                destination,
                timeout_seconds=int(data_cfg.get("timeout_seconds", 30)),
            )
            frame = _read_csv(destination)
            logger.info("downloaded %s rows from %s", len(frame), data_cfg["url"])
            return frame
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return _fallback(data_cfg, exc)

    raise ValueError(f"unknown data source: {source}")
