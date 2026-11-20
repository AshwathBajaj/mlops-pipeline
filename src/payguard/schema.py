"""Column contract shared by loading, validation, training, and serving.

The contract matches the ULB credit-card fraud table redistributed by OpenML
as dataset 1597: Time, Amount, the PCA components V1-V28, and a fraud label.
"""

from __future__ import annotations

ID_COLUMN = "transaction_id"
TARGET_COLUMN = "fraud"

PCA_COLUMNS: tuple[str, ...] = tuple(f"V{index}" for index in range(1, 29))
NUMERIC_COLUMNS: tuple[str, ...] = ("Time", "Amount", *PCA_COLUMNS)
CATEGORICAL_COLUMNS: tuple[str, ...] = ()
RAW_FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS
ALL_COLUMNS: tuple[str, ...] = (ID_COLUMN, *RAW_FEATURE_COLUMNS, TARGET_COLUMN)

OPENML_DATA_ID = 1597
OPENML_NAME = "creditcard"
OPENML_URL = "https://www.openml.org/d/1597"
OPENML_PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
DATASET_LICENCE = "Public"
DATASET_ORIGIN = (
    "ULB Machine Learning Group and Worldline. European cardholders, September 2013. "
    "Two days of transactions. V1-V28 are principal components; Time and Amount are raw."
)
DATASET_CITATION = (
    "Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. "
    "Calibrating Probability with Undersampling for Unbalanced Classification. "
    "In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015."
)
