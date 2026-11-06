"""
Explainability and interpretability module for credit risk modeling.
Computes global feature importances, permutation importances,
and applicant-level risk contribution factors.
"""
import logging
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from src.config import FIGURES_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def extract_global_feature_importances(
    model: Any,
    feature_names: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extracts feature importances using tree MDI, linear coefficients,
    or permutation importance.
    """
    importances = None

    # Pipeline (e.g. Logistic Regression)
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])

    # Tree ensemble models
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # Fallback to permutation importance if model lacks intrinsic importances
    if importances is None and X_val is not None and y_val is not None:
        logger.info("Computing permutation importances...")
        sub_n = min(1000, len(X_val))
        sub_X = X_val.iloc[:sub_n]
        sub_y = y_val.iloc[:sub_n]
        perm = permutation_importance(
            model, sub_X, sub_y, n_repeats=3, random_state=RANDOM_SEED, scoring="roc_auc"
        )
        importances = perm.importances_mean

    if importances is None:
        importances = np.ones(len(feature_names))

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df_imp


def plot_top_features(df_importances: pd.DataFrame, model_name: str, top_n: int = 20, save_path: str = None):
    """
    Saves horizontal bar chart of top predictive features.
    """
    top_df = df_importances.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, 7), dpi=300)
    bars = plt.barh(top_df["Feature"], top_df["Importance"], color="#1f77b4", edgecolor="none", height=0.65)
    plt.xlabel("Importance Metric (Information Gain / Permutation Score)", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_n} Predictive Drivers — {model_name}", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "feature_importance.png")
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved feature importance figure to {out}")


def explain_single_applicant(
    applicant_row: pd.Series,
    feature_importances: pd.DataFrame,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Generates human-readable risk reason codes for an individual credit application.
    """
    top_driver_names = feature_importances["Feature"].head(25).tolist()
    explanations = []

    for feat in top_driver_names:
        if feat in applicant_row:
            val = applicant_row[feat]
            # Domain contextual explanations
            if feat == "EXT_SOURCE_MEAN" or feat == "EXT_SOURCE_2" or feat == "EXT_SOURCE_3":
                if val < 0.35:
                    explanations.append({
                        "feature": feat,
                        "value": round(float(val), 3),
                        "impact": "HIGH RISK",
                        "description": "Credit bureau score significantly below creditworthy threshold."
                    })
                elif val > 0.65:
                    explanations.append({
                        "feature": feat,
                        "value": round(float(val), 3),
                        "impact": "LOW RISK",
                        "description": "Strong external credit bureau history."
                    })
            elif feat == "DEBT_TO_INCOME":
                if val > 4.0:
                    explanations.append({
                        "feature": feat,
                        "value": round(float(val), 2),
                        "impact": "HIGH RISK",
                        "description": f"Loan amount is {val:.1f}x annual income, representing severe leverage."
                    })
                elif val < 2.0:
                    explanations.append({
                        "feature": feat,
                        "value": round(float(val), 2),
                        "impact": "LOW RISK",
                        "description": "Healthy debt-to-income ratio."
                    })
            elif feat == "AGE_YEARS":
                if val < 25.0:
                    explanations.append({
                        "feature": feat,
                        "value": round(float(val), 1),
                        "impact": "ELEVATED RISK",
                        "description": "Young credit profile with limited borrowing tenure."
                    })
            elif feat == "EMPLOYED_YEARS":
                if val < 1.5:
                    explanations.append({
                        "feature": feat,
                        "value": round(float(val), 1),
                        "impact": "ELEVATED RISK",
                        "description": "Short employment tenure (<1.5 years)."
                    })
                elif val > 5.0:
                    explanations.append({
                        "feature": feat,
                        "value": round(float(val), 1),
                        "impact": "LOW RISK",
                        "description": "Stable multi-year employment tenure."
                    })

    # Return top N insights
    return explanations[:top_n]
