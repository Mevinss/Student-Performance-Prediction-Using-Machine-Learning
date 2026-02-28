"""
Machine learning models for student performance prediction.

Provides a unified interface for training, predicting, and comparing
multiple classification algorithms on the OULAD feature set.

Supported models
----------------
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (sklearn)
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .feature_engineering import build_preprocessing_pipeline


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_default_models() -> Dict[str, object]:
    """
    Return a dictionary of model name → unfitted estimator.

    All classifiers are wrapped inside a Pipeline that first applies
    median imputation and standard scaling.
    """
    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42
        ),
    }
    if _HAS_XGB:
        base_models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    if _HAS_LGB:
        base_models["LightGBM"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1,
        )

    # Wrap each model in a preprocessing + classifier pipeline
    pipelines = {}
    for name, estimator in base_models.items():
        pipelines[name] = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", estimator),
        ])
    return pipelines


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Stores training and evaluation results for a single model."""

    name: str
    train_time: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0
    feature_importances: Optional[np.ndarray] = field(default=None, repr=False)
    pipeline: Optional[Pipeline] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "Model": self.name,
            "Accuracy": round(self.accuracy, 4),
            "F1 Score": round(self.f1, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "ROC-AUC": round(self.roc_auc, 4),
            "CV Accuracy": f"{self.cv_accuracy_mean:.4f} ± {self.cv_accuracy_std:.4f}",
            "CV F1": f"{self.cv_f1_mean:.4f} ± {self.cv_f1_std:.4f}",
            "Train Time (s)": round(self.train_time, 2),
        }


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Trains and evaluates multiple classifiers on the OULAD feature matrix.

    Parameters
    ----------
    models : dict, optional
        Mapping of model name → Pipeline.  Defaults to `get_default_models()`.
    test_size : float
        Fraction of data reserved for the hold-out test set.
    cv_folds : int
        Number of stratified cross-validation folds.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        models: Optional[Dict[str, Pipeline]] = None,
        test_size: float = 0.20,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.models = models or get_default_models()
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.results: Dict[str, ModelResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, X: pd.DataFrame, y: pd.Series) -> "ModelTrainer":
        """Perform stratified train/test split."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        print(
            f"Train set: {len(self.X_train):,} samples | "
            f"Test set:  {len(self.X_test):,} samples"
        )
        return self

    def train_all(self) -> "ModelTrainer":
        """Train every registered model and store evaluation results."""
        assert self.X_train is not None, "Call .split() before .train_all()"
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        for name, pipeline in self.models.items():
            print(f"  Training {name} …", end=" ", flush=True)
            result = self._train_single(name, pipeline, cv)
            self.results[name] = result
            print(
                f"done  (acc={result.accuracy:.4f}, "
                f"roc-auc={result.roc_auc:.4f}, "
                f"time={result.train_time:.1f}s)"
            )
        return self

    def get_results_df(self) -> pd.DataFrame:
        """Return a DataFrame comparing all trained models."""
        rows = [r.as_dict() for r in self.results.values()]
        return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    def best_model(self) -> Tuple[str, ModelResult]:
        """Return the (name, ModelResult) of the best model by ROC-AUC."""
        best = max(self.results.items(), key=lambda kv: kv[1].roc_auc)
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_single(
        self, name: str, pipeline: Pipeline, cv: StratifiedKFold
    ) -> ModelResult:
        # Cross-validation
        cv_scores = cross_validate(
            pipeline, self.X_train, self.y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1"},
            n_jobs=1,
        )

        # Final fit on full training set
        t0 = time.perf_counter()
        pipeline.fit(self.X_train, self.y_train)
        train_time = time.perf_counter() - t0

        # Predictions on hold-out test set
        y_pred = pipeline.predict(self.X_test)
        y_proba = pipeline.predict_proba(self.X_test)[:, 1]

        # Feature importances (where available)
        importances = _extract_importances(pipeline)

        return ModelResult(
            name=name,
            train_time=train_time,
            accuracy=accuracy_score(self.y_test, y_pred),
            f1=f1_score(self.y_test, y_pred, zero_division=0),
            precision=precision_score(self.y_test, y_pred, zero_division=0),
            recall=recall_score(self.y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(self.y_test, y_proba),
            cv_accuracy_mean=float(cv_scores["test_accuracy"].mean()),
            cv_accuracy_std=float(cv_scores["test_accuracy"].std()),
            cv_f1_mean=float(cv_scores["test_f1"].mean()),
            cv_f1_std=float(cv_scores["test_f1"].std()),
            feature_importances=importances,
            pipeline=pipeline,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_importances(pipeline: Pipeline) -> Optional[np.ndarray]:
    """Extract feature importances from the classifier step if available."""
    clf = pipeline.named_steps.get("classifier")
    if clf is None:
        return None
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_).ravel()
    return None
