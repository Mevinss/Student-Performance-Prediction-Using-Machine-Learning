"""
Evaluation and reporting module.

Generates:
  - Confusion matrix plots
  - ROC curve comparisons
  - Precision-Recall curve comparisons
  - Feature importance bar charts
  - Summary classification reports
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

from .models import ModelResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PALETTE = sns.color_palette("tab10")
FIG_DPI = 120


# ---------------------------------------------------------------------------
# Individual model evaluation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=FIG_DPI)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["At-Risk", "Pass"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def print_classification_report(y_true, y_pred, model_name: str = "") -> None:
    """Print a formatted classification report."""
    header = f"\n{'=' * 60}\nClassification Report — {model_name}\n{'=' * 60}"
    print(header)
    print(
        classification_report(
            y_true, y_pred,
            target_names=["At-Risk (0)", "Pass (1)"],
            zero_division=0,
        )
    )


# ---------------------------------------------------------------------------
# Multi-model comparison plots
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: Dict[str, ModelResult],
    X_test: pd.DataFrame,
    y_test,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for all trained models on the same axes.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=FIG_DPI)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC = 0.50)")

    for i, (name, result) in enumerate(results.items()):
        if result.pipeline is None:
            continue
        y_proba = result.pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.8,
            label=f"{name} (AUC = {auc_score:.4f})",
        )

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_precision_recall_curves(
    results: Dict[str, ModelResult],
    X_test: pd.DataFrame,
    y_test,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Precision-Recall curves for all trained models."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=FIG_DPI)
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=0.8,
               label=f"Baseline (AP = {baseline:.2f})")

    for i, (name, result) in enumerate(results.items()):
        if result.pipeline is None:
            continue
        y_proba = result.pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = auc(recall, precision)
        ax.plot(
            recall, precision,
            color=PALETTE[i % len(PALETTE)],
            linewidth=1.8,
            label=f"{name} (AP = {ap:.4f})",
        )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing key metrics across all models."""
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"]
    available = [m for m in metrics if m in results_df.columns]

    plot_df = results_df.set_index("Model")[available].copy()
    # Convert string CV columns to numeric if they slipped in
    for col in plot_df.columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    x = np.arange(len(plot_df))
    width = 0.15
    for j, metric in enumerate(available):
        ax.bar(
            x + j * width,
            plot_df[metric],
            width,
            label=metric,
            color=PALETTE[j % len(PALETTE)],
        )

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "Model",
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-N most important features."""
    n = min(top_n, len(feature_names))
    sorted_idx = np.argsort(importances)[-n:]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)), dpi=FIG_DPI)
    bars = ax.barh(sorted_names, sorted_vals, color=sns.color_palette("viridis", n))
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name} (Top {n})", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all evaluation plots
# ---------------------------------------------------------------------------

def save_all_evaluation_plots(
    trainer,
    feature_names: List[str],
    output_dir: str = "outputs/figures",
) -> None:
    """
    Generate and save all evaluation plots to *output_dir*.

    Parameters
    ----------
    trainer : ModelTrainer
        A fully trained ModelTrainer instance.
    feature_names : list[str]
        Ordered list of feature names matching the columns of X_train/X_test.
    output_dir : str
        Directory where PNG figures are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- ROC curves -------------------------------------------------------
    plot_roc_curves(
        trainer.results, trainer.X_test, trainer.y_test,
        save_path=os.path.join(output_dir, "roc_curves.png"),
    )

    # --- PR curves --------------------------------------------------------
    plot_precision_recall_curves(
        trainer.results, trainer.X_test, trainer.y_test,
        save_path=os.path.join(output_dir, "pr_curves.png"),
    )

    # --- Model comparison bar chart ---------------------------------------
    plot_model_comparison(
        trainer.get_results_df(),
        save_path=os.path.join(output_dir, "model_comparison.png"),
    )

    # --- Per-model confusion matrices and feature importance ---------------
    for name, result in trainer.results.items():
        safe_name = name.replace(" ", "_").lower()
        if result.pipeline is not None:
            y_pred = result.pipeline.predict(trainer.X_test)
            plot_confusion_matrix(
                trainer.y_test, y_pred,
                title=f"Confusion Matrix — {name}",
                save_path=os.path.join(output_dir, f"cm_{safe_name}.png"),
            )
        if result.feature_importances is not None:
            plot_feature_importance(
                feature_names,
                result.feature_importances,
                model_name=name,
                save_path=os.path.join(output_dir, f"fi_{safe_name}.png"),
            )

    plt.close("all")
    print(f"All evaluation plots saved to '{output_dir}'.")
