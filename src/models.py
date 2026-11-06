"""
Model definitions, registry, and hyperparameter grids for credit risk modeling.
Supports Logistic Regression, Random Forest, HistGradientBoosting,
with optional dynamic acceleration for LightGBM and XGBoost if available.
"""
import logging
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

# Check for optional gradient boosting libraries
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_candidate_models(scale_pos_weight: float = 12.0) -> Dict[str, Any]:
    """
    Returns baseline candidate models configured to handle extreme class imbalance.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                solver="lbfgs",
                random_state=RANDOM_SEED
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=150,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=RANDOM_SEED
        ),
    }

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            class_weight="balanced",
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        )

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            tree_method="hist",
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

    return models


def get_tuning_grid(model_name: str) -> Dict[str, Any]:
    """
    Returns hyperparameter search space for tuning.
    """
    if "Hist Gradient" in model_name:
        return {
            "max_iter": [100, 150, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.06, 0.10],
            "min_samples_leaf": [15, 25, 40],
            "l2_regularization": [0.0, 1.0, 5.0],
        }
    elif "Random Forest" in model_name:
        return {
            "n_estimators": [100, 150, 200],
            "max_depth": [6, 8, 10, None],
            "min_samples_split": [10, 20, 30],
            "min_samples_leaf": [10, 20, 30],
            "max_features": ["sqrt", "log2", 0.3],
        }
    elif "Logistic Regression" in model_name:
        return {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        }
    elif "LightGBM" in model_name:
        return {
            "n_estimators": [100, 150, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.07, 0.12],
            "num_leaves": [20, 31, 50],
        }
    elif "XGBoost" in model_name:
        return {
            "n_estimators": [100, 150, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.08, 0.15],
            "subsample": [0.7, 0.85, 1.0],
        }
    return {}
