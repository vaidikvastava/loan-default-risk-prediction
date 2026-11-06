"""
End-to-end training pipeline for Loan Default Risk Prediction.
Executes data preprocessing, feature engineering, multi-model benchmarking,
hyperparameter tuning, cost-sensitive threshold search, and artifact serialization.
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
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TRAIN_PROCESSED_PATH,
    TEST_PROCESSED_PATH,
    FEATURES_LIST_PATH,
    MODELS_DIR,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    TEST_SPLIT_SIZE,
    RANDOM_SEED,
    TARGET_COL,
    ID_COL,
)
from src.data_loader import load_raw_data, validate_schema
from src.preprocessor import LoanDataPreprocessor
from src.feature_engineering import FinancialFeatureEngineer
from src.models import get_candidate_models, get_tuning_grid
from src.evaluate import (
    evaluate_predictions,
    find_optimal_cost_threshold,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_cost_loss_curve,
    plot_risk_distribution,
)
from src.explainability import (
    extract_global_feature_importances,
    plot_top_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes the full machine learning training lifecycle.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Validate
    raw_df = load_raw_data(str(RAW_DATA_PATH))
    schema_report = validate_schema(raw_df)

    # 2. Stratified Train / Test Split
    logger.info(f"Splitting dataset into {1 - TEST_SPLIT_SIZE:.0%} train and {TEST_SPLIT_SIZE:.0%} test...")
    train_raw, test_raw = train_test_split(
        raw_df,
        test_size=TEST_SPLIT_SIZE,
        stratify=raw_df[TARGET_COL],
        random_state=RANDOM_SEED,
    )
    logger.info(f"Train Raw: {train_raw.shape}, Test Raw: {test_raw.shape}")

    # 3. Fit Preprocessor on Train Data
    preprocessor = LoanDataPreprocessor()
    preprocessor.fit(train_raw)
    train_clean = preprocessor.transform(train_raw)
    test_clean = preprocessor.transform(test_raw)

    # 4. Fit Feature Engineer on Train Data
    feat_engineer = FinancialFeatureEngineer(cardinality_cutoff=10)
    feat_engineer.fit(train_clean)
    train_proc = feat_engineer.transform(train_clean)
    test_proc = feat_engineer.transform(test_clean)

    # Save processed datasets
    train_proc.to_csv(TRAIN_PROCESSED_PATH, index=False)
    test_proc.to_csv(TEST_PROCESSED_PATH, index=False)
    logger.info(f"Saved processed datasets to {PROCESSED_DATA_DIR}")

    # Save feature names
    feature_cols = [c for c in train_proc.columns if c not in [TARGET_COL, ID_COL]]
    with open(FEATURES_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Save preprocessing pipeline
    pipeline_artifacts = {
        "preprocessor": preprocessor,
        "feat_engineer": feat_engineer,
        "feature_names": feature_cols,
    }
    joblib.dump(pipeline_artifacts, PREPROCESSOR_PATH)
    logger.info(f"Saved preprocessing pipeline to {PREPROCESSOR_PATH}")

    # 5. Prepare Modeling Matrices
    X_train = train_proc[feature_cols]
    y_train = train_proc[TARGET_COL].astype(int)
    X_test = test_proc[feature_cols]
    y_test = test_proc[TARGET_COL].astype(int)

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    # 6. Multi-Model Benchmark
    candidate_models = get_candidate_models(scale_pos_weight=scale_pos_weight)
    fitted_models = {}
    benchmark_results = []
    roc_plot_data = {}
    pr_plot_data = {}

    for name, model in candidate_models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        probs = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test.values, probs, threshold=0.5)

        fpr, tpr, _ = roc_curve(y_test, probs)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
        roc_plot_data[name] = (fpr, tpr, metrics["roc_auc"])
        pr_plot_data[name] = (rec_curve, prec_curve, metrics["pr_auc"])

        benchmark_results.append({
            "model": name,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "brier_score": metrics["brier_score"],
        })
        logger.info(f"--> {name}: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, Recall={metrics['recall']:.4f}")

    benchmark_df = pd.DataFrame(benchmark_results).sort_values("roc_auc", ascending=False)
    best_candidate_name = benchmark_df.iloc[0]["model"]
    logger.info(f"Top benchmarked model: {best_candidate_name}")

    # 7. Hyperparameter Tuning on Top Candidate
    logger.info(f"Tuning hyperparameters for {best_candidate_name}...")
    tuning_grid = get_tuning_grid(best_candidate_name)
    best_raw_model = fitted_models[best_candidate_name]

    if tuning_grid:
        tune_sample_n = min(5000, len(X_train))
        X_tune = X_train.sample(tune_sample_n, random_state=RANDOM_SEED)
        y_tune = y_train.loc[X_tune.index]

        search = RandomizedSearchCV(
            best_raw_model,
            tuning_grid,
            n_iter=6,
            scoring="roc_auc",
            cv=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        search.fit(X_tune, y_tune)
        best_model = search.best_estimator_
        # Refit on full training set
        best_model.fit(X_train, y_train)
        best_params = search.best_params_
        logger.info(f"Best tuned parameters: {best_params}")
    else:
        best_model = best_raw_model
        best_params = {}

    # 8. Final Holdout Evaluation & Cost Threshold Analysis
    final_probs = best_model.predict_proba(X_test)[:, 1]
    final_metrics_default = evaluate_predictions(y_test.values, final_probs, threshold=0.50)
    cost_optimization = find_optimal_cost_threshold(y_test.values, final_probs)
    opt_threshold = cost_optimization["optimal_cost_threshold"]
    final_metrics_optimal = evaluate_predictions(y_test.values, final_probs, threshold=opt_threshold)

    logger.info(
        f"Final Tuned Test ROC-AUC: {final_metrics_optimal['roc_auc']:.4f} | "
        f"Default Cost (0.50): {cost_optimization['default_0_5_cost']} | "
        f"Optimal Cost ({opt_threshold:.2f}): {cost_optimization['min_financial_loss']} "
        f"(Cost reduced by {cost_optimization['cost_reduction_pct']}%)"
    )

    # 9. Generate Charts
    baseline_def_rate = float(y_test.mean())
    plot_roc_curves(roc_plot_data)
    plot_precision_recall_curves(pr_plot_data, baseline_def_rate)
    plot_confusion_matrices(y_test.values, final_probs, opt_threshold, best_candidate_name)
    plot_cost_loss_curve(cost_optimization["thresholds"], cost_optimization["costs"], opt_threshold)
    plot_risk_distribution(y_test.values, final_probs, opt_threshold)

    # 10. Feature Importance & Explainability
    logger.info("Computing feature importances...")
    df_importances = extract_global_feature_importances(
        best_model, feature_cols, X_val=X_test, y_val=y_test, top_n=20
    )
    plot_top_features(df_importances, best_candidate_name, top_n=20)

    # 11. Serialize Best Model and Metrics Summary
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info(f"Saved best model to {BEST_MODEL_PATH}")

    metrics_payload = {
        "best_model_name": best_candidate_name,
        "best_parameters": best_params,
        "benchmark_comparison": benchmark_results,
        "evaluation_default_threshold": final_metrics_default,
        "evaluation_optimal_threshold": final_metrics_optimal,
        "cost_optimization": {
            "optimal_cost_threshold": opt_threshold,
            "min_financial_loss": cost_optimization["min_financial_loss"],
            "default_0_5_cost": cost_optimization["default_0_5_cost"],
            "cost_reduction_pct": cost_optimization["cost_reduction_pct"],
        },
        "top_features": df_importances.head(15).to_dict(orient="records"),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved metrics summary to {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Best Model: {best_candidate_name}")
    print(f"Test ROC-AUC: {final_metrics_optimal['roc_auc']:.4f}")
    print(f"Optimal Cost-Sensitive Threshold: {opt_threshold:.3f}")
    print(f"Recall on Defaulters at Optimal Cutoff: {final_metrics_optimal['recall']:.1%}")
    print(f"Business Cost Reduction: {cost_optimization['cost_reduction_pct']:.1f}%")
    print("=" * 60 + "\n")

    return metrics_payload


if __name__ == "__main__":
    run_training_pipeline()
