"""
Production Inference and Underwriting Engine for Loan Default Risk.
Provides real-time scoring, risk tier assignment, approval recommendations,
and applicant-level risk explanations.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH
from src.explainability import explain_single_applicant

logger = logging.getLogger(__name__)


class LoanRiskPredictor:
    """
    Thread-safe production predictor for credit risk scoring.
    """

    def __init__(self, model_path: str = None, preprocessor_path: str = None, metrics_path: str = None):
        self.model_path = model_path or str(BEST_MODEL_PATH)
        self.preprocessor_path = preprocessor_path or str(PREPROCESSOR_PATH)
        self.metrics_path = metrics_path or str(METRICS_PATH)

        self._load_artifacts()

    def _load_artifacts(self):
        logger.info("Loading inference artifacts...")
        self.model = joblib.load(self.model_path)
        pipe_data = joblib.load(self.preprocessor_path)
        self.preprocessor = pipe_data["preprocessor"]
        self.feat_engineer = pipe_data["feat_engineer"]
        self.feature_names = pipe_data["feature_names"]

        if Path(self.metrics_path).exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                self.optimal_threshold = metrics.get("cost_optimization", {}).get("optimal_cost_threshold", 0.35)
                self.top_features_df = pd.DataFrame(metrics.get("top_features", []))
        else:
            self.optimal_threshold = 0.35
            self.top_features_df = pd.DataFrame()

    def predict_single(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scores a single loan applicant dictionary.
        """
        df_raw = pd.DataFrame([applicant])
        return self.predict_dataframe(df_raw)[0]

    def predict_dataframe(self, df_raw: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scores a batch DataFrame of loan applicants.
        """
        # Transform through fitted pipeline
        df_clean = self.preprocessor.transform(df_raw)
        df_feat = self.feat_engineer.transform(df_clean)

        # Align with model features
        X = df_feat[self.feature_names]
        probabilities = self.model.predict_proba(X)[:, 1]

        results = []
        for idx, prob in enumerate(probabilities):
            prob = float(prob)

            # Assign Risk Tier
            if prob < 0.20:
                tier = "LOW RISK"
                tier_color = "#28a745"
            elif prob < 0.40:
                tier = "MODERATE RISK"
                tier_color = "#ffc107"
            else:
                tier = "HIGH RISK"
                tier_color = "#dc3545"

            # Underwriting recommendation based on cost-benefit threshold
            if prob < (self.optimal_threshold * 0.75):
                recommendation = "APPROVE"
                rationale = "Applicant demonstrates high creditworthiness and low probability of default."
            elif prob <= (self.optimal_threshold * 1.25):
                recommendation = "MANUAL REVIEW"
                rationale = "Borderline credit profile. Recommend human underwriter audit of collateral and income verification."
            else:
                recommendation = "DECLINE"
                rationale = "Predicted default risk exceeds permissible portfolio loss tolerance."

            # Applicant explanations
            explanations = []
            if not self.top_features_df.empty:
                applicant_row = df_feat.iloc[idx]
                explanations = explain_single_applicant(applicant_row, self.top_features_df, top_n=4)

            results.append({
                "default_probability": round(prob, 4),
                "default_risk_percentage": f"{prob * 100:.1f}%",
                "risk_tier": tier,
                "risk_color": tier_color,
                "recommendation": recommendation,
                "recommendation_rationale": rationale,
                "decision_cutoff": self.optimal_threshold,
                "key_risk_factors": explanations,
            })

        return results


def get_sample_applicant() -> Dict[str, Any]:
    """
    Returns an authentic test applicant profile.
    """
    return {
        "AMT_INCOME_TOTAL": 180000.0,
        "AMT_CREDIT": 450000.0,
        "AMT_ANNUITY": 23500.0,
        "AMT_GOODS_PRICE": 400000.0,
        "DAYS_BIRTH": -14500,  # ~39.7 years old
        "DAYS_EMPLOYED": -2200,  # ~6.0 years employed
        "DAYS_REGISTRATION": -5000,
        "DAYS_ID_PUBLISH": -2500,
        "EXT_SOURCE_1": 0.42,
        "EXT_SOURCE_2": 0.58,
        "EXT_SOURCE_3": 0.61,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_RATING_CLIENT": 2,
    }


def main():
    parser = argparse.ArgumentParser(description="Loan Default Risk Scoring Engine")
    parser.add_argument("--sample", action="store_true", help="Score sample applicant")
    parser.add_argument("--input", type=str, help="Path to input JSON applicant file")
    parser.add_argument("--batch", type=str, help="Path to batch CSV file")
    parser.add_argument("--output", type=str, help="Path to output CSV predictions")
    args = parser.parse_args()

    predictor = LoanRiskPredictor()

    if args.sample:
        sample = get_sample_applicant()
        res = predictor.predict_single(sample)
        print("\n--- LOAN RISK UNDERWRITING ASSESSMENT ---")
        print(f"Default Probability : {res['default_risk_percentage']}")
        print(f"Risk Tier           : {res['risk_tier']}")
        print(f"Recommendation      : {res['recommendation']}")
        print(f"Rationale           : {res['recommendation_rationale']}")
        print("Key Driving Factors :")
        for factor in res["key_risk_factors"]:
            print(f"  • {factor['feature']} = {factor['value']} ({factor['impact']}): {factor['description']}")
        print("-----------------------------------------\n")

    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        res = predictor.predict_single(data)
        print(json.dumps(res, indent=2))

    elif args.batch:
        df = pd.read_csv(args.batch)
        preds = predictor.predict_dataframe(df)
        df_out = pd.concat([df.reset_index(drop=True), pd.DataFrame(preds)], axis=1)
        out_path = args.output or "predictions_output.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Scored {len(df)} records and saved to {out_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
