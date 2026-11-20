"""Load a transaction table from the generator, a CSV, or a URL."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from payguard.data.generate import generate_transactions

logger = logging.getLogger(__name__)


def _synthetic_from_config(data_cfg: dict[str, Any]) -> pd.DataFrame:
    return generate_transactions(
        n_rows=int(data_cfg["n_rows"]),
        seed=int(data_cfg.get("seed", 42)),
        target_fraud_rate=float(data_cfg.get("target_fraud_rate", 0.02)),
    )


def download_csv(url: str, destination: Path, timeout_seconds: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "payguard/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        destination.write_bytes(response.read())


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
        frame = pd.read_csv(path)
        logger.info("loaded %s rows from %s", len(frame), path)
        return frame

    if source == "url":
        destination = Path(data_cfg.get("download_path", "data/raw/downloaded.csv"))
        try:
            download_csv(
                data_cfg["url"],
                destination,
                timeout_seconds=int(data_cfg.get("timeout_seconds", 30)),
            )
            frame = pd.read_csv(destination)
            logger.info("downloaded %s rows from %s", len(frame), data_cfg["url"])
            return frame
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            logger.warning("download failed (%s); falling back to the synthetic generator", exc)
            return _synthetic_from_config(data_cfg)

    raise ValueError(f"unknown data source: {source}")
