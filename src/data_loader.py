"""
Data loader and schema validation utilities for Loan Default Prediction.
"""
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, TARGET_COL, ID_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads raw Home Credit loan application data.
    """
    path = filepath or str(RAW_DATA_PATH)
    logger.info(f"Loading raw loan data from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Data loaded successfully. Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    return df


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates essential schema constraints on the dataset.
    """
    has_target = TARGET_COL in df.columns
    has_id = ID_COL in df.columns
    target_dist = df[TARGET_COL].value_counts(normalize=True).to_dict() if has_target else {}
    null_target_count = int(df[TARGET_COL].isnull().sum()) if has_target else 0

    validation_report = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "has_id_column": has_id,
        "has_target_column": has_target,
        "target_null_count": null_target_count,
        "target_distribution": target_dist,
        "is_valid": has_target and null_target_count == 0,
    }
    logger.info(f"Schema Validation Result: {validation_report}")
    return validation_report


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level dataset summary metrics.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    missing_series = df.isnull().mean()
    heavy_missing_cols = missing_series[missing_series > 0.50].index.tolist()

    return {
        "num_records": len(df),
        "num_columns": len(df.columns),
        "num_numeric_columns": len(num_cols),
        "num_categorical_columns": len(cat_cols),
        "columns_with_over_50pct_missing": len(heavy_missing_cols),
        "overall_missing_fraction": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
    }
