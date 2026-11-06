"""
Model evaluation, calibration, and cost-sensitive threshold optimization.
Generates publication-quality figures for ROC curves, PR curves, Confusion Matrix,
and expected business loss curves.
"""
import logging
from typing import Dict, Any, Tuple, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from src.config import (
    FIGURES_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
)

logger = logging.getLogger(__name__)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes key classification metrics at a given decision threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    total_cost = float((COST_FALSE_NEGATIVE * fn) + (COST_FALSE_POSITIVE * fp))

    return {
        "threshold": threshold,
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_financial_loss": total_cost,
    }


def find_optimal_cost_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Sweeps thresholds to identify the business-optimal threshold that minimizes
    total financial loan loss (Cost = 5 * FN + 1 * FP).
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = []
    f1_scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (COST_FALSE_NEGATIVE * fn) + (COST_FALSE_POSITIVE * fp)
        costs.append(cost)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    optimal_threshold = float(thresholds[opt_idx])
    min_cost = float(costs[opt_idx])

    # Also find F1-maximizing threshold
    f1_opt_idx = np.argmax(f1_scores)
    f1_optimal_threshold = float(thresholds[f1_opt_idx])

    logger.info(
        f"Optimal Cost Threshold: {optimal_threshold:.3f} (Min Cost: {min_cost:,.1f}) "
        f"vs Default 0.5 Cost: {costs[np.abs(thresholds - 0.5).argmin()]:,.1f}"
    )

    return {
        "optimal_cost_threshold": round(optimal_threshold, 3),
        "min_financial_loss": round(min_cost, 2),
        "default_0_5_cost": round(float(costs[np.abs(thresholds - 0.5).argmin()]), 2),
        "cost_reduction_pct": round(
            float((costs[np.abs(thresholds - 0.5).argmin()] - min_cost) / costs[np.abs(thresholds - 0.5).argmin()] * 100), 2
        ),
        "optimal_f1_threshold": round(f1_optimal_threshold, 3),
        "thresholds": thresholds.tolist(),
        "costs": costs.tolist(),
    }


def plot_roc_curves(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]], save_path: str = None):
    """
    Plots multi-model ROC curves.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "roc_curves.png")
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved ROC curves to {out}")


def plot_precision_recall_curves(pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]], baseline_rate: float, save_path: str = None):
    """
    Plots multi-model Precision-Recall curves.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name, (recall, precision, pr_auc) in pr_data.items():
        plt.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    plt.axhline(baseline_rate, color="gray", lw=1.5, linestyle="--", label=f"Baseline Defaulter Rate ({baseline_rate:.1%})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Coverage of Actual Defaulters)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Accuracy of Flagged Defaults)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curves under Class Imbalance", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "pr_curves.png")
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved PR curves to {out}")


def plot_confusion_matrices(y_true: np.ndarray, y_prob: np.ndarray, opt_threshold: float, model_name: str, save_path: str = None):
    """
    Plots side-by-side confusion matrices: Standard 0.50 vs Cost-Optimal Threshold.
    """
    cm_standard = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    cm_optimal = confusion_matrix(y_true, (y_prob >= opt_threshold).astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    cms = [("Default Threshold (0.50)", cm_standard), (f"Cost-Optimal Threshold ({opt_threshold:.2f})", cm_optimal)]

    for ax, (title, cm) in zip(axes, cms):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=12, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Non-Defaulter", "Defaulter"], fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Non-Defaulter", "Defaulter"], fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]:,d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold"
                )
        ax.set_ylabel("Actual Ground Truth", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model Underwriting Decision", fontsize=10, fontweight="bold")

    plt.suptitle(f"Confusion Matrix Comparison: {model_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "confusion_matrix.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrices to {out}")


def plot_cost_loss_curve(thresholds: List[float], costs: List[float], opt_threshold: float, save_path: str = None):
    """
    Plots total expected financial loss across decision thresholds.
    """
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(thresholds, costs, lw=2.5, color="#1f77b4", label="Total Financial Loss ($5 FN + $1 FP)")
    min_cost = min(costs)
    plt.scatter([opt_threshold], [min_cost], color="#d62728", s=100, zorder=5, label=f"Optimal Threshold = {opt_threshold:.2f}")
    plt.axvline(0.50, color="gray", linestyle="--", label="Standard 0.50 Baseline")
    plt.xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Portfolio Loss Units", fontsize=11, fontweight="bold")
    plt.title("Credit Risk Cost-Benefit Optimization Curve", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "cost_curve.png")
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved cost curve to {out}")


def plot_risk_distribution(y_true: np.ndarray, y_prob: np.ndarray, opt_threshold: float, save_path: str = None):
    """
    Plots predicted risk probability histogram stratified by actual repayment status.
    """
    plt.figure(figsize=(8, 5), dpi=300)
    prob_repaid = y_prob[y_true == 0]
    prob_default = y_prob[y_true == 1]

    bins = np.linspace(0, 1, 40)
    plt.hist(prob_repaid, bins=bins, alpha=0.6, color="#2ca02c", density=True, label="Actual Non-Defaulters (Repaid)")
    plt.hist(prob_default, bins=bins, alpha=0.6, color="#d62728", density=True, label="Actual Defaulters")
    plt.axvline(opt_threshold, color="black", linestyle="--", lw=2, label=f"Decision Cutoff ({opt_threshold:.2f})")

    plt.xlabel("Predicted Probability of Default", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.title("Model Risk Discrimination Distribution", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "risk_score_distribution.png")
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved risk distribution to {out}")
