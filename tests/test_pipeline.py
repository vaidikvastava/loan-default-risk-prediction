"""
Unit and integration test suite for Loan Default Prediction ML System.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data, validate_schema
from src.preprocessor import LoanDataPreprocessor
from src.feature_engineering import FinancialFeatureEngineer
from src.predict import LoanRiskPredictor, get_sample_applicant
from src.database import db_manager


class TestLoanRiskPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_df = load_raw_data()
        cls.sample_applicant = get_sample_applicant()
        cls.predictor = LoanRiskPredictor()

    def test_01_schema_validation(self):
        report = validate_schema(self.raw_df)
        self.assertTrue(report["is_valid"])
        self.assertTrue(report["has_target_column"])
        self.assertEqual(report["target_null_count"], 0)
        self.assertGreater(report["total_records"], 5000)

    def test_02_preprocessor_anomaly_handling(self):
        sample_df = pd.DataFrame([{
            "DAYS_EMPLOYED": 365243,  # Anomaly placeholder
            "DAYS_BIRTH": -14610,     # ~40 years
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 300000.0,
        }])
        preprocessor = LoanDataPreprocessor()
        preprocessor.fit(sample_df)
        transformed = preprocessor.transform(sample_df)

        self.assertIn("DAYS_EMPLOYED_ANOM", transformed.columns)
        self.assertEqual(transformed["DAYS_EMPLOYED_ANOM"].iloc[0], 1)
        self.assertAlmostEqual(transformed["AGE_YEARS"].iloc[0], 40.0, places=1)

    def test_03_financial_feature_engineer_ratios(self):
        sample_df = pd.DataFrame([{
            "AMT_INCOME_TOTAL": 100000.0,
            "AMT_CREDIT": 250000.0,
            "AMT_ANNUITY": 20000.0,
            "AMT_GOODS_PRICE": 250000.0,
            "AGE_YEARS": 35.0,
            "EMPLOYED_YEARS": 7.0,
            "CNT_FAM_MEMBERS": 2,
            "EXT_SOURCE_1": 0.40,
            "EXT_SOURCE_2": 0.50,
            "EXT_SOURCE_3": 0.60,
            "NAME_CONTRACT_TYPE": "Cash loans"
        }])
        fe = FinancialFeatureEngineer()
        fe.fit(sample_df)
        out = fe.transform(sample_df)

        self.assertIn("DEBT_TO_INCOME", out.columns)
        self.assertAlmostEqual(out["DEBT_TO_INCOME"].iloc[0], 2.5, places=2)
        self.assertAlmostEqual(out["ANNUITY_TO_INCOME"].iloc[0], 0.20, places=2)
        self.assertAlmostEqual(out["EXT_SOURCE_MEAN"].iloc[0], 0.50, places=2)

    def test_04_predictor_inference_bounds(self):
        result = self.predictor.predict_single(self.sample_applicant)
        prob = result["default_probability"]

        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertIn(result["risk_tier"], ["LOW RISK", "MODERATE RISK", "HIGH RISK"])
        self.assertIn(result["recommendation"], ["APPROVE", "MANUAL REVIEW", "DECLINE"])
        self.assertIsInstance(result["key_risk_factors"], list)

    def test_05_edge_case_zero_income(self):
        zero_inc_applicant = self.sample_applicant.copy()
        zero_inc_applicant["AMT_INCOME_TOTAL"] = 0.0
        zero_inc_applicant["AMT_GOODS_PRICE"] = 0.0

        # Must not raise ZeroDivisionError
        result = self.predictor.predict_single(zero_inc_applicant)
        self.assertIsInstance(result["default_probability"], float)

    def test_06_database_logging(self):
        log_id = db_manager.log_underwriting_decision(
            sk_id_curr=999991,
            default_probability=0.25,
            risk_tier="MODERATE RISK",
            recommendation="APPROVE",
            decision_threshold=0.61,
            notes="Unit test record"
        )
        self.assertIsNotNone(log_id)
        recent = db_manager.get_recent_decisions(limit=10)
        self.assertFalse(recent.empty)


if __name__ == "__main__":
    unittest.main()
