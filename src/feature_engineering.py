"""
Feature engineering module for student performance prediction.

Transforms raw OULAD master dataset into model-ready numeric features,
applying ordinal encoding, one-hot encoding, imputation, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Ordinal mappings (domain-specific ordering)
# ---------------------------------------------------------------------------

EDUCATION_ORDER = [
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
]

IMD_BAND_ORDER = [
    "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]

AGE_BAND_ORDER = ["0-35", "35-55", "55<="]

GENDER_MAP = {"M": 0, "F": 1}
DISABILITY_MAP = {"N": 0, "Y": 1}


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

DEMOGRAPHIC_FEATURES = [
    "gender_enc",
    "highest_education_enc",
    "imd_band_enc",
    "age_band_enc",
    "num_of_prev_attempts",
    "studied_credits",
    "disability_enc",
]

VLE_FEATURES = [
    "total_clicks",
    "n_unique_sites",
    "n_active_days",
    "avg_clicks_per_day",
]

ASSESSMENT_FEATURES = [
    "mean_score",
    "min_score",
    "max_score",
    "n_submissions",
    "n_late_submissions",
]

REGISTRATION_FEATURES = [
    "date_registration",
    "early_registration",
]

ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + VLE_FEATURES
    + ASSESSMENT_FEATURES
    + REGISTRATION_FEATURES
)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that converts the OULAD master
    DataFrame into a numeric feature matrix.

    Steps:
    1. Binary encode gender and disability.
    2. Ordinal encode education level, IMD band, age band.
    3. Fill missing VLE / assessment / registration features with 0.
    4. Return a DataFrame with a fixed, documented column order.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()

        # -- Binary encodings ------------------------------------------------
        df["gender_enc"] = df["gender"].map(GENDER_MAP).fillna(0).astype(int)
        df["disability_enc"] = df["disability"].map(DISABILITY_MAP).fillna(0).astype(int)

        # -- Ordinal encodings -----------------------------------------------
        df["highest_education_enc"] = _ordinal_encode(
            df["highest_education"], EDUCATION_ORDER
        )
        df["imd_band_enc"] = _ordinal_encode(df["imd_band"], IMD_BAND_ORDER)
        df["age_band_enc"] = _ordinal_encode(df["age_band"], AGE_BAND_ORDER)

        # -- Fill missing numeric features -----------------------------------
        for col in VLE_FEATURES + ASSESSMENT_FEATURES + REGISTRATION_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        df[VLE_FEATURES + ASSESSMENT_FEATURES + REGISTRATION_FEATURES] = (
            df[VLE_FEATURES + ASSESSMENT_FEATURES + REGISTRATION_FEATURES]
            .fillna(0)
        )

        return df[ALL_FEATURES].copy()


# ---------------------------------------------------------------------------
# Scikit-learn preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline() -> Pipeline:
    """
    Return a scikit-learn Pipeline that imputes then scales all features.

    Intended to be used *after* FeatureEngineer, which produces a DataFrame
    where all columns are already numeric.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ordinal_encode(series: pd.Series, order: list) -> pd.Series:
    """Map a categorical series to integer ranks according to *order*."""
    mapping = {val: idx for idx, val in enumerate(order)}
    return series.map(mapping).fillna(-1).astype(int)


def get_feature_names() -> list:
    """Return the ordered list of feature column names used by the model."""
    return list(ALL_FEATURES)


def engineer_features(master_df: pd.DataFrame) -> tuple:
    """
    Convenience function: apply FeatureEngineer and return (X, y).

    Parameters
    ----------
    master_df : pd.DataFrame
        Output of OULADLoader.build_master_dataset().

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix (all rows from master_df).
    y : pd.Series
        Binary outcome label (1 = Pass/Distinction, 0 = Fail/Withdrawn).
    """
    engineer = FeatureEngineer()
    X = engineer.transform(master_df)
    y = master_df["outcome"].astype(int)
    return X, y
