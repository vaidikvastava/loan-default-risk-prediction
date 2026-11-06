"""
Financial feature engineering module for loan default prediction.
Creates domain-specific affordability and burden ratios, combines external ratings,
and executes deterministic categorical encoding with strict column alignment.
"""
import re
import logging
from collections import Counter
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import EXT_SOURCE_COLS

logger = logging.getLogger(__name__)


def sanitize_column_names(columns: List[str]) -> List[str]:
    """
    Sanitizes column names to be alphanumeric and underscores only,
    preventing issues with tree algorithms (e.g. LightGBM JSON errors).
    """
    clean_cols = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_") for col in columns]
    counts = Counter()
    deduped = []
    for c in clean_cols:
        counts[c] += 1
        deduped.append(c if counts[c] == 1 else f"{c}_{counts[c]}")
    return deduped


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Engineers financial ratios, external source metrics, and encoded categoricals.
    Guarantees consistent columns between train, test, and inference.
    """

    def __init__(self, cardinality_cutoff: int = 10):
        self.cardinality_cutoff = cardinality_cutoff
        self.low_card_cols_: List[str] = []
        self.high_card_cols_: List[str] = []
        self.freq_maps_: Dict[str, Dict[str, float]] = {}
        self.ratio_medians_: Dict[str, float] = {}
        self.final_feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _compute_ratios_and_ext(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Financial Affordability & Leverage Ratios
        credit = df.get("AMT_CREDIT", pd.Series(0, index=df.index))
        income = df.get("AMT_INCOME_TOTAL", pd.Series(1, index=df.index)).replace(0, 1.0)
        annuity = df.get("AMT_ANNUITY", pd.Series(1, index=df.index)).replace(0, 1.0)
        goods = df.get("AMT_GOODS_PRICE", pd.Series(credit, index=df.index)).replace(0, 1.0)
        fam_members = df.get("CNT_FAM_MEMBERS", pd.Series(1, index=df.index)).replace(0, 1.0)

        age_years = df.get("AGE_YEARS", pd.Series(30, index=df.index)).replace(0, 30.0)
        emp_years = df.get("EMPLOYED_YEARS", pd.Series(0, index=df.index)).clip(lower=0)

        df["DEBT_TO_INCOME"] = credit / income
        df["ANNUITY_TO_INCOME"] = annuity / income
        df["CREDIT_TO_GOODS_RATIO"] = credit / goods
        df["CREDIT_ANNUITY_RATIO"] = credit / annuity
        df["INCOME_PER_FAMILY_MEMBER"] = income / fam_members
        df["EMPLOYED_TO_AGE_RATIO"] = emp_years / age_years

        ratio_cols = [
            "DEBT_TO_INCOME",
            "ANNUITY_TO_INCOME",
            "CREDIT_TO_GOODS_RATIO",
            "CREDIT_ANNUITY_RATIO",
            "INCOME_PER_FAMILY_MEMBER",
            "EMPLOYED_TO_AGE_RATIO",
        ]
        df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)

        # Impute or store ratio medians
        for col in ratio_cols:
            if not self.is_fitted_:
                med = float(df[col].median())
                self.ratio_medians_[col] = 0.0 if np.isnan(med) else med
            df[col] = df[col].fillna(self.ratio_medians_.get(col, 0.0))

        # External Credit Ratings Syntheses (EXT_SOURCE 1, 2, 3)
        available_ext = [col for col in EXT_SOURCE_COLS if col in df.columns]
        if available_ext:
            df["EXT_SOURCE_MEAN"] = df[available_ext].mean(axis=1)
            df["EXT_SOURCE_MIN"] = df[available_ext].min(axis=1)
            df["EXT_SOURCE_MAX"] = df[available_ext].max(axis=1)
            df["EXT_SOURCE_STD"] = df[available_ext].std(axis=1).fillna(0.0)
        else:
            df["EXT_SOURCE_MEAN"] = 0.5
            df["EXT_SOURCE_MIN"] = 0.5
            df["EXT_SOURCE_MAX"] = 0.5
            df["EXT_SOURCE_STD"] = 0.0

        return df

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting FinancialFeatureEngineer...")
        df = X.copy()
        target_series = None
        id_series = None
        if "TARGET" in df.columns:
            target_series = df["TARGET"]
            df = df.drop(columns=["TARGET"])
        if "SK_ID_CURR" in df.columns:
            id_series = df["SK_ID_CURR"]
            df = df.drop(columns=["SK_ID_CURR"])

        df = self._compute_ratios_and_ext(df)

        # Identify categorical columns
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        cardinality = df[cat_cols].nunique()

        self.low_card_cols_ = cardinality[cardinality <= self.cardinality_cutoff].index.tolist()
        self.high_card_cols_ = cardinality[cardinality > self.cardinality_cutoff].index.tolist()

        # Learn frequency encoding maps
        for col in self.high_card_cols_:
            freq = df[col].value_counts(normalize=True).to_dict()
            self.freq_maps_[col] = freq

        # Fit dummy encoding
        df_encoded = pd.get_dummies(df, columns=self.low_card_cols_, drop_first=True)

        for col in self.high_card_cols_:
            freq_map = self.freq_maps_[col]
            df_encoded[f"{col}_FREQ"] = df[col].map(freq_map).fillna(0.0)
            df_encoded = df_encoded.drop(columns=[col], errors="ignore")

        # Sanitize column names
        sanitized_cols = sanitize_column_names(df_encoded.columns.tolist())
        df_encoded.columns = sanitized_cols

        # Save feature list
        self.final_feature_names_ = df_encoded.columns.tolist()
        self.is_fitted_ = True
        logger.info(f"FinancialFeatureEngineer fitted with {len(self.final_feature_names_)} final features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("FinancialFeatureEngineer must be fitted before calling transform().")

        df = X.copy()
        target_series = None
        id_series = None
        if "TARGET" in df.columns:
            target_series = df["TARGET"]
            df = df.drop(columns=["TARGET"])
        if "SK_ID_CURR" in df.columns:
            id_series = df["SK_ID_CURR"]
            df = df.drop(columns=["SK_ID_CURR"])

        df = self._compute_ratios_and_ext(df)

        # Low cardinality one-hot
        present_low = [c for c in self.low_card_cols_ if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=present_low, drop_first=True)

        # High cardinality frequency map
        for col in self.high_card_cols_:
            if col in df.columns:
                freq_map = self.freq_maps_.get(col, {})
                df_encoded[f"{col}_FREQ"] = df[col].map(freq_map).fillna(0.0)
                df_encoded = df_encoded.drop(columns=[col], errors="ignore")
            else:
                df_encoded[f"{col}_FREQ"] = 0.0

        # Sanitize column names
        clean_cols = sanitize_column_names(df_encoded.columns.tolist())
        df_encoded.columns = clean_cols

        # Reindex to exactly match fitted features
        df_final = df_encoded.reindex(columns=self.final_feature_names_, fill_value=0.0)
        df_final = df_final.fillna(0.0)

        # Re-attach target or id if they were present
        if target_series is not None:
            df_final["TARGET"] = target_series.values
        if id_series is not None:
            df_final["SK_ID_CURR"] = id_series.values

        return df_final.copy()
