"""
Exploratory Data Analysis (EDA) visualization module.

Provides reusable plotting functions for demographic distributions,
outcome distributions, VLE engagement patterns, and correlation analysis.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG_DPI = 120
RESULT_PALETTE = {
    "Pass": "#2ecc71",
    "Distinction": "#3498db",
    "Fail": "#e74c3c",
    "Withdrawn": "#95a5a6",
}


# ---------------------------------------------------------------------------
# Outcome distribution
# ---------------------------------------------------------------------------

def plot_outcome_distribution(
    student_info: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of final_result counts."""
    counts = student_info["final_result"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=FIG_DPI)

    # Count
    colors = [RESULT_PALETTE.get(r, "#999") for r in counts.index]
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title("Final Result Distribution (Counts)", fontsize=12)
    axes[0].set_ylabel("Number of Students")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontsize=9)

    # Pie
    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
    )
    axes[1].set_title("Final Result Distribution (%)", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Demographic analysis
# ---------------------------------------------------------------------------

def plot_demographic_analysis(
    student_info: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grid of bar charts showing outcome breakdown by demographic variables."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=FIG_DPI)
    axes = axes.ravel()

    demographic_cols = [
        ("gender", "Gender"),
        ("highest_education", "Highest Education"),
        ("age_band", "Age Band"),
        ("imd_band", "IMD Band"),
        ("disability", "Disability"),
        ("num_of_prev_attempts", "Previous Attempts"),
    ]

    for ax, (col, label) in zip(axes, demographic_cols):
        if col == "num_of_prev_attempts":
            ct = pd.crosstab(student_info[col], student_info["final_result"], normalize="index") * 100
            ct.plot(kind="bar", ax=ax, colormap="tab10", legend=False)
            ax.set_xlabel("Previous Attempts")
        else:
            ct = pd.crosstab(student_info[col], student_info["final_result"], normalize="index") * 100
            ct.plot(kind="bar", ax=ax, colormap="tab10", legend=False)
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        ax.set_title(f"Outcome by {label}", fontsize=11)
        ax.set_ylabel("Percentage (%)")
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Final Result",
               loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# VLE engagement analysis
# ---------------------------------------------------------------------------

def plot_vle_engagement(
    master_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Box plots comparing VLE engagement metrics by outcome."""
    vle_cols = ["total_clicks", "n_unique_sites", "n_active_days", "avg_clicks_per_day"]
    available = [c for c in vle_cols if c in master_df.columns]
    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No VLE data available", ha="center", va="center")
        return fig

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), dpi=FIG_DPI)
    if n == 1:
        axes = [axes]

    labels_map = {0: "At-Risk", 1: "Pass"}
    plot_df = master_df.copy()
    plot_df["Result"] = plot_df["outcome"].map(labels_map)

    for ax, col in zip(axes, available):
        sns.boxplot(
            data=plot_df, x="Result", y=col, hue="Result",
            palette={"At-Risk": "#e74c3c", "Pass": "#2ecc71"},
            ax=ax, showfliers=False, legend=False,
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("VLE Engagement by Student Outcome", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Assessment score analysis
# ---------------------------------------------------------------------------

def plot_assessment_scores(
    master_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Distribution of mean assessment scores by outcome."""
    if "mean_score" not in master_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No assessment data available", ha="center", va="center")
        return fig

    labels_map = {0: "At-Risk", 1: "Pass"}
    plot_df = master_df.copy()
    plot_df["Result"] = plot_df["outcome"].map(labels_map)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=FIG_DPI)

    # Histogram
    for result, color in [("Pass", "#2ecc71"), ("At-Risk", "#e74c3c")]:
        subset = plot_df[plot_df["Result"] == result]["mean_score"].dropna()
        axes[0].hist(subset, bins=25, alpha=0.6, color=color, label=result, density=True)
    axes[0].set_title("Distribution of Mean Assessment Score", fontsize=12)
    axes[0].set_xlabel("Mean Score")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    sns.boxplot(
        data=plot_df, x="Result", y="mean_score", hue="Result",
        palette={"At-Risk": "#e74c3c", "Pass": "#2ecc71"},
        ax=axes[1], legend=False,
    )
    axes[1].set_title("Mean Assessment Score by Outcome", fontsize=12)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean Score")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    feature_df: pd.DataFrame,
    y: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Correlation heatmap of features and outcome label."""
    data = feature_df.copy()
    data["outcome"] = y.values
    corr = data.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=FIG_DPI)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Save all EDA plots
# ---------------------------------------------------------------------------

def save_all_eda_plots(
    student_info: pd.DataFrame,
    master_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    y: pd.Series,
    output_dir: str = "outputs/figures",
) -> None:
    """Generate and save all EDA plots to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    plot_outcome_distribution(
        student_info,
        save_path=os.path.join(output_dir, "eda_outcome_distribution.png"),
    )
    plot_demographic_analysis(
        student_info,
        save_path=os.path.join(output_dir, "eda_demographic_analysis.png"),
    )
    plot_vle_engagement(
        master_df,
        save_path=os.path.join(output_dir, "eda_vle_engagement.png"),
    )
    plot_assessment_scores(
        master_df,
        save_path=os.path.join(output_dir, "eda_assessment_scores.png"),
    )
    plot_correlation_heatmap(
        feature_df, y,
        save_path=os.path.join(output_dir, "eda_correlation_heatmap.png"),
    )

    plt.close("all")
    print(f"All EDA plots saved to '{output_dir}'.")
