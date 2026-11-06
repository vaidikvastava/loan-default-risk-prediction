"""
Configuration parameters for Loan Default Risk Prediction ML System.
"""
import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "application_train.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_preprocessed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_preprocessed.csv"
FEATURES_LIST_PATH = PROCESSED_DATA_DIR / "feature_names.json"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "metrics_summary.json"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data Schema & Cleaning Parameters
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
DAYS_EMPLOYED_ANOM_VAL = 365243

DAYS_COLS_MAP = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "EMPLOYED_YEARS",
    "DAYS_REGISTRATION": "REGISTRATION_YEARS",
    "DAYS_ID_PUBLISH": "ID_PUBLISH_YEARS",
}

FINANCIAL_CAP_COLS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]
CAP_LOWER_QUANTILE = 0.01
CAP_UPPER_QUANTILE = 0.99
MISSING_INDICATOR_THRESHOLD = 0.10

# External source rating features
EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# Experiment & Training Parameters
TEST_SPLIT_SIZE = 0.20
RANDOM_SEED = 42

# Underwriting Cost Matrix: In credit risk, a False Negative (unpredicted default)
# is substantially costlier than a False Positive (lost interest from safe client)
COST_FALSE_NEGATIVE = 5.0  # Estimated 5x cost of credit loss vs missed deal
COST_FALSE_POSITIVE = 1.0  # Underwriting review cost or lost customer acquisition
