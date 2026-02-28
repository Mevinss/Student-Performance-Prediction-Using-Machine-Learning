"""
Unit tests for the Student Performance Prediction pipeline.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Data Loader Tests
# ---------------------------------------------------------------------------

class TestOULADLoader:
    def setup_method(self):
        from src.data_loader import OULADLoader
        self.loader = OULADLoader(use_synthetic=True, n_synthetic=500, random_state=0)
        self.loader.load()

    def test_tables_present(self):
        expected = [
            "studentInfo", "courses", "assessments",
            "studentAssessment", "vle", "studentVle", "studentRegistration",
        ]
        for name in expected:
            assert name in self.loader._tables, f"Missing table: {name}"

    def test_student_info_shape(self):
        df = self.loader.get_table("studentInfo")
        assert len(df) == 500
        assert "final_result" in df.columns

    def test_final_result_values(self):
        df = self.loader.get_table("studentInfo")
        valid = {"Pass", "Distinction", "Fail", "Withdrawn"}
        assert set(df["final_result"].unique()).issubset(valid)

    def test_master_dataset_columns(self):
        master = self.loader.build_master_dataset()
        for col in ["outcome", "final_result", "gender", "highest_education"]:
            assert col in master.columns, f"Missing column: {col}"

    def test_master_outcome_binary(self):
        master = self.loader.build_master_dataset()
        assert set(master["outcome"].unique()).issubset({0, 1})

    def test_master_no_duplicate_key(self):
        master = self.loader.build_master_dataset()
        dupes = master.duplicated(subset=["id_student", "code_module", "code_presentation"])
        assert not dupes.any(), "Duplicate rows found in master dataset"

    def test_vle_aggregation(self):
        master = self.loader.build_master_dataset()
        # total_clicks should be non-negative where present
        assert (master["total_clicks"].dropna() >= 0).all()


class TestSyntheticDataGeneration:
    def test_reproducibility(self):
        from src.data_loader import generate_synthetic_oulad
        d1 = generate_synthetic_oulad(n_students=100, random_state=7)
        d2 = generate_synthetic_oulad(n_students=100, random_state=7)
        pd.testing.assert_frame_equal(d1["studentInfo"], d2["studentInfo"])

    def test_different_seeds_differ(self):
        from src.data_loader import generate_synthetic_oulad
        d1 = generate_synthetic_oulad(n_students=100, random_state=1)
        d2 = generate_synthetic_oulad(n_students=100, random_state=2)
        assert not d1["studentInfo"]["final_result"].equals(d2["studentInfo"]["final_result"])


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def setup_method(self):
        from src.data_loader import OULADLoader
        loader = OULADLoader(use_synthetic=True, n_synthetic=300, random_state=0)
        loader.load()
        self.master = loader.build_master_dataset()

    def test_engineer_features_shape(self):
        from src.feature_engineering import engineer_features, get_feature_names
        X, y = engineer_features(self.master)
        feature_names = get_feature_names()
        assert X.shape[0] == len(self.master)
        assert X.shape[1] == len(feature_names)

    def test_no_missing_in_X(self):
        from src.feature_engineering import engineer_features
        X, _ = engineer_features(self.master)
        assert X.isnull().sum().sum() == 0, "Feature matrix contains NaN values"

    def test_outcome_binary(self):
        from src.feature_engineering import engineer_features
        _, y = engineer_features(self.master)
        assert set(y.unique()).issubset({0, 1})

    def test_ordinal_encoding_bounds(self):
        from src.feature_engineering import engineer_features, EDUCATION_ORDER
        X, _ = engineer_features(self.master)
        assert X["highest_education_enc"].between(-1, len(EDUCATION_ORDER) - 1).all()

    def test_feature_engineer_fit_transform(self):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        fe.fit(self.master)
        X = fe.transform(self.master)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(self.master)


# ---------------------------------------------------------------------------
# Model Training Tests
# ---------------------------------------------------------------------------

class TestModelTrainer:
    def setup_method(self):
        from src.data_loader import OULADLoader
        from src.feature_engineering import engineer_features
        loader = OULADLoader(use_synthetic=True, n_synthetic=400, random_state=0)
        loader.load()
        master = loader.build_master_dataset()
        self.X, self.y = engineer_features(master)

    def test_split_sizes(self):
        from src.models import ModelTrainer
        trainer = ModelTrainer(test_size=0.2, random_state=0)
        trainer.split(self.X, self.y)
        total = len(trainer.X_train) + len(trainer.X_test)
        assert total == len(self.X)
        assert abs(len(trainer.X_test) / total - 0.2) < 0.05

    def test_logistic_regression_only(self):
        """Training a single model completes without error."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from src.feature_engineering import build_preprocessing_pipeline
        from src.models import ModelTrainer

        pipeline = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=200, random_state=0)),
        ])
        trainer = ModelTrainer(
            models={"LR": pipeline},
            test_size=0.2,
            cv_folds=3,
            random_state=0,
        )
        trainer.split(self.X, self.y)
        trainer.train_all()
        assert "LR" in trainer.results

    def test_result_metrics_in_range(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from src.feature_engineering import build_preprocessing_pipeline
        from src.models import ModelTrainer

        pipeline = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=200, random_state=0)),
        ])
        trainer = ModelTrainer(
            models={"LR": pipeline},
            test_size=0.2,
            cv_folds=3,
            random_state=0,
        )
        trainer.split(self.X, self.y)
        trainer.train_all()
        result = trainer.results["LR"]
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.roc_auc <= 1.0
        assert 0.0 <= result.f1 <= 1.0

    def test_best_model_returns(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from src.feature_engineering import build_preprocessing_pipeline
        from src.models import ModelTrainer

        pipeline = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=200, random_state=0)),
        ])
        trainer = ModelTrainer(
            models={"LR": pipeline},
            test_size=0.2,
            cv_folds=3,
            random_state=0,
        )
        trainer.split(self.X, self.y)
        trainer.train_all()
        name, result = trainer.best_model()
        assert name == "LR"
        assert result.roc_auc >= 0.0


# ---------------------------------------------------------------------------
# Evaluation Tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.y_true = pd.Series(rng.integers(0, 2, size=100))
        self.y_pred = pd.Series(rng.integers(0, 2, size=100))

    def test_plot_confusion_matrix_returns_figure(self):
        import matplotlib.pyplot as plt
        from src.evaluation import plot_confusion_matrix
        fig = plot_confusion_matrix(self.y_true, self.y_pred)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_print_classification_report(self, capsys):
        from src.evaluation import print_classification_report
        print_classification_report(self.y_true, self.y_pred, model_name="Test")
        captured = capsys.readouterr()
        assert "Classification Report" in captured.out
        assert "precision" in captured.out
