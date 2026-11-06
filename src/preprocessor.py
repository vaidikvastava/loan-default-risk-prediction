"""
Data preprocessor transformer for credit risk datasets.
Handles DAYS_EMPLOYED placeholder anomalies, day-to-year conversions,
building feature reduction, median/categorical imputation, and outlier capping.
"""
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import (
    DAYS_EMPLOYED_ANOM_VAL,
    DAYS_COLS_MAP,
    FINANCIAL_CAP_COLS,
    CAP_LOWER_QUANTILE,
    CAP_UPPER_QUANTILE,
    MISSING_INDICATOR_THRESHOLD,
)

logger = logging.getLogger(__name__)


class LoanDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Fitted preprocessor preserving train distribution parameters
    to prevent data leakage into validation or test sets.
    """

    def __init__(self):
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.building_cols_: List[str] = []
        self.missing_flag_cols_: List[str] = []
        self.numeric_medians_: Dict[str, float] = {}
        self.cap_limits_: Dict[str, Tuple[float, float]] = {}
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        logger.info("Fitting LoanDataPreprocessor on training distribution...")

        # Flag DAYS_EMPLOYED anomaly before computing statistics
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOM_VAL, np.nan)

        # Convert DAYS_* to years
        for col, new_col in DAYS_COLS_MAP.items():
            if col in df.columns:
                df[new_col] = -df[col] / 365.25

        # Identify building columns
        num_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
        self.building_cols_ = [
            c for c in df.columns
            if any(term in c for term in ["_AVG", "_MODE", "_MEDI"]) and c in num_cols_raw
        ]
        df = df.drop(columns=self.building_cols_, errors="ignore")

        # Classify column types after dropping building features
        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude TARGET and ID if present
        for excl in ["TARGET", "SK_ID_CURR"]:
            if excl in self.numeric_cols_:
                self.numeric_cols_.remove(excl)

        self.categorical_cols_ = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        # Learn missingness indicators and medians for numeric columns
        for col in self.numeric_cols_:
            missing_rate = df[col].isnull().mean()
            if missing_rate > MISSING_INDICATOR_THRESHOLD:
                self.missing_flag_cols_.append(col)
            med = float(df[col].median())
            self.numeric_medians_[col] = 0.0 if np.isnan(med) else med

        # Learn outlier capping percentiles for financial variables
        for col in FINANCIAL_CAP_COLS:
            if col in df.columns:
                lower = float(df[col].quantile(CAP_LOWER_QUANTILE))
                upper = float(df[col].quantile(CAP_UPPER_QUANTILE))
                self.cap_limits_[col] = (lower, upper)

        self.is_fitted_ = True
        logger.info(
            f"Preprocessor fitted: {len(self.building_cols_)} building cols to drop, "
            f"{len(self.missing_flag_cols_)} missing indicators learned, "
            f"{len(self.numeric_medians_)} numeric medians computed."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("LoanDataPreprocessor must be fitted before calling transform().")

        df = X.copy()

        # 1. Flag DAYS_EMPLOYED anomaly (365243 placeholder)
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOM_VAL).astype(int)
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOM_VAL, np.nan)
        else:
            df["DAYS_EMPLOYED_ANOM"] = 0

        # 2. Convert DAYS_* columns to positive interpretable years
        for col, new_col in DAYS_COLS_MAP.items():
            if col in df.columns:
                df[new_col] = -df[col] / 365.25

        # 3. Collapse building columns into a single indicator flag
        present_building_cols = [c for c in self.building_cols_ if c in df.columns]
        if present_building_cols:
            df["HAS_BUILDING_INFO"] = df[present_building_cols].notnull().any(axis=1).astype(int)
            df = df.drop(columns=present_building_cols, errors="ignore")
        else:
            df["HAS_BUILDING_INFO"] = 0

        # 4. Generate missingness indicator flags
        for col in self.missing_flag_cols_:
            flag_col = f"{col}_WAS_MISSING"
            if col in df.columns:
                df[flag_col] = df[col].isnull().astype(int)
            else:
                df[flag_col] = 1

        # 5. Impute numeric missing values with training medians
        for col, med in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        # 6. Impute categorical missing values with 'Unknown'
        for col in self.categorical_cols_:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        # 7. Apply financial outlier capping
        for col, (lower, upper) in self.cap_limits_.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

        # Fill any remaining stray NaNs gracefully
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0.0)

        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        for c in cat_cols:
            df[c] = df[c].fillna("Unknown")

        return df.copy()
