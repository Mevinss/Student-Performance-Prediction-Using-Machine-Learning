"""
Student Performance Prediction Using Machine Learning
Open University Learning Analytics Dataset (OULAD)

Usage
-----
    python main.py                          # run with synthetic data
    python main.py --data-dir data/         # run with real OULAD CSV files
    python main.py --output-dir results/    # custom output directory
    python main.py --no-plots               # skip figure generation
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student Performance Prediction (OULAD)"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing OULAD CSV files (default: data/)",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for results and figures (default: outputs/)",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=5000,
        help="Number of synthetic student records when real data is absent (default: 5000)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STUDENT PERFORMANCE PREDICTION — OULAD")
    print("=" * 60)

    from src.data_loader import OULADLoader
    loader = OULADLoader(
        data_dir=args.data_dir,
        n_synthetic=args.n_synthetic,
        random_state=args.random_state,
    )
    loader.load()
    student_info = loader.get_table("studentInfo")
    master_df = loader.build_master_dataset()

    print(f"\nMaster dataset: {len(master_df):,} rows × {master_df.shape[1]} columns")
    print(f"Class balance:\n{master_df['outcome'].value_counts(normalize=True).round(3)}")

    # ------------------------------------------------------------------
    # 2. Exploratory Data Analysis
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Running EDA …")
    if not args.no_plots:
        from src.visualization import save_all_eda_plots
        from src.feature_engineering import engineer_features
        X_raw, y = engineer_features(master_df)
        save_all_eda_plots(student_info, master_df, X_raw, y, output_dir=figures_dir)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Engineering features …")
    from src.feature_engineering import engineer_features, get_feature_names
    X, y = engineer_features(master_df)
    feature_names = get_feature_names()
    print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # ------------------------------------------------------------------
    # 4. Model training
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Training models …")
    from src.models import ModelTrainer
    trainer = ModelTrainer(random_state=args.random_state)
    trainer.split(X, y)
    trainer.train_all()

    # ------------------------------------------------------------------
    # 5. Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    results_df = trainer.get_results_df()
    print(results_df.to_string(index=False))

    # Save results CSV
    csv_path = os.path.join(args.output_dir, "model_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")

    best_name, best_result = trainer.best_model()
    print(f"\nBest model: {best_name}  (ROC-AUC = {best_result.roc_auc:.4f})")

    # Detailed classification report for the best model
    from src.evaluation import print_classification_report
    if best_result.pipeline is not None:
        y_pred_best = best_result.pipeline.predict(trainer.X_test)
        print_classification_report(trainer.y_test, y_pred_best, model_name=best_name)

    # ------------------------------------------------------------------
    # 6. Save evaluation plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n" + "-" * 40)
        print("Saving evaluation plots …")
        from src.evaluation import save_all_evaluation_plots
        save_all_evaluation_plots(trainer, feature_names, output_dir=figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
