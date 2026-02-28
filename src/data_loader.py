"""
Data loading and preprocessing module for OULAD dataset.

The Open University Learning Analytics Dataset (OULAD) contains data about
courses, students, and their interactions with the Virtual Learning Environment
(VLE) for seven selected courses (modules).

Dataset files expected in the data/ directory:
    - studentInfo.csv
    - courses.csv
    - assessments.csv
    - studentAssessment.csv
    - vle.csv
    - studentVle.csv
    - studentRegistration.csv
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OULAD schema constants
# ---------------------------------------------------------------------------

FINAL_RESULT_MAP = {
    "Pass": 1,
    "Distinction": 2,
    "Fail": 0,
    "Withdrawn": 0,
}

BINARY_RESULT_MAP = {
    "Pass": 1,
    "Distinction": 1,
    "Fail": 0,
    "Withdrawn": 0,
}

GENDER_MAP = {"M": 0, "F": 1}
DISABILITY_MAP = {"N": 0, "Y": 1}


# ---------------------------------------------------------------------------
# Synthetic data generation (mirrors OULAD schema for demonstration)
# ---------------------------------------------------------------------------

def generate_synthetic_oulad(n_students: int = 5000, random_state: int = 42) -> dict:
    """
    Generate synthetic data that mirrors the OULAD schema.

    Returns a dictionary of DataFrames keyed by table name, matching the
    structure of the real OULAD CSV files.

    Parameters
    ----------
    n_students : int
        Number of synthetic student records to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys: 'studentInfo', 'courses', 'assessments',
        'studentAssessment', 'vle', 'studentVle', 'studentRegistration'.
    """
    rng = np.random.default_rng(random_state)

    modules = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    presentations = ["2013J", "2014J", "2013B", "2014B"]
    regions = [
        "East Anglian Region", "Scotland", "North Western Region",
        "South East Region", "West Midlands Region", "Wales",
        "Yorkshire Region", "London Region", "East Midlands Region",
        "South West Region", "Ireland", "North Region",
    ]
    education_levels = [
        "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
        "HE Qualification", "Post Graduate Qualification",
    ]
    imd_bands = [
        "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
    ]
    age_bands = ["0-35", "35-55", "55<="]
    results = ["Pass", "Distinction", "Fail", "Withdrawn"]
    result_weights = [0.40, 0.15, 0.25, 0.20]

    # ------------------------------------------------------------------
    # studentInfo
    # ------------------------------------------------------------------
    student_ids = np.arange(1, n_students + 1)
    code_modules = rng.choice(modules, size=n_students)
    code_presentations = rng.choice(presentations, size=n_students)

    # Correlated outcome generation
    edu_idx = rng.integers(0, len(education_levels), size=n_students)
    imd_idx = rng.integers(0, len(imd_bands), size=n_students)
    age_idx = rng.integers(0, len(age_bands), size=n_students)
    prev_attempts = rng.integers(0, 4, size=n_students)
    studied_credits = rng.choice([30, 60, 90, 120, 150, 180, 210, 240], size=n_students)

    # Higher education level and fewer previous attempts → better outcome
    outcome_score = (
        edu_idx * 0.3
        - prev_attempts * 0.5
        + rng.normal(0, 1, size=n_students)
    )
    outcome_probs = np.exp(outcome_score) / (1 + np.exp(outcome_score))

    final_results = []
    for prob in outcome_probs:
        if prob > 0.70:
            final_results.append(rng.choice(["Pass", "Distinction"], p=[0.6, 0.4]))
        elif prob > 0.40:
            final_results.append("Pass")
        elif prob > 0.20:
            final_results.append(rng.choice(["Fail", "Withdrawn"], p=[0.55, 0.45]))
        else:
            final_results.append(rng.choice(["Fail", "Withdrawn"], p=[0.4, 0.6]))

    student_info = pd.DataFrame({
        "id_student": student_ids,
        "code_module": code_modules,
        "code_presentation": code_presentations,
        "gender": rng.choice(["M", "F"], size=n_students),
        "region": rng.choice(regions, size=n_students),
        "highest_education": [education_levels[i] for i in edu_idx],
        "imd_band": [imd_bands[i] for i in imd_idx],
        "age_band": [age_bands[i] for i in age_idx],
        "num_of_prev_attempts": prev_attempts,
        "studied_credits": studied_credits,
        "disability": rng.choice(["N", "Y"], size=n_students, p=[0.90, 0.10]),
        "final_result": final_results,
    })

    # ------------------------------------------------------------------
    # courses
    # ------------------------------------------------------------------
    courses_rows = []
    for mod in modules:
        for pres in presentations:
            courses_rows.append({
                "code_module": mod,
                "code_presentation": pres,
                "module_presentation_length": rng.integers(220, 270),
            })
    courses = pd.DataFrame(courses_rows)

    # ------------------------------------------------------------------
    # assessments
    # ------------------------------------------------------------------
    assessment_types = ["TMA", "CMA", "Exam"]
    assessments_rows = []
    assessment_id = 1
    for mod in modules:
        for pres in presentations:
            pres_length = courses.loc[
                (courses["code_module"] == mod)
                & (courses["code_presentation"] == pres),
                "module_presentation_length",
            ].values[0]
            n_tma = rng.integers(2, 5)
            for k in range(n_tma):
                assessments_rows.append({
                    "id_assessment": assessment_id,
                    "code_module": mod,
                    "code_presentation": pres,
                    "assessment_type": "TMA",
                    "date": int(pres_length * (k + 1) / (n_tma + 1)),
                    "weight": round(60 / n_tma, 2),
                })
                assessment_id += 1
            assessments_rows.append({
                "id_assessment": assessment_id,
                "code_module": mod,
                "code_presentation": pres,
                "assessment_type": "Exam",
                "date": pres_length,
                "weight": 40.0,
            })
            assessment_id += 1
    assessments = pd.DataFrame(assessments_rows)

    # ------------------------------------------------------------------
    # studentAssessment
    # ------------------------------------------------------------------
    sa_rows = []
    for _, row in student_info.iterrows():
        sid = row["id_student"]
        mod = row["code_module"]
        pres = row["code_presentation"]
        result = row["final_result"]
        student_assessments = assessments[
            (assessments["code_module"] == mod)
            & (assessments["code_presentation"] == pres)
        ]
        for _, a in student_assessments.iterrows():
            if result == "Withdrawn" and rng.random() < 0.5:
                continue
            if result in ("Pass", "Distinction"):
                base_score = rng.integers(50, 101)
            elif result == "Fail":
                base_score = rng.integers(20, 65)
            else:
                base_score = rng.integers(10, 55)
            sa_rows.append({
                "id_assessment": a["id_assessment"],
                "id_student": sid,
                "date_submitted": a["date"] + rng.integers(-5, 6),
                "is_banked": 0,
                "score": min(100, max(0, base_score)),
            })
    student_assessment = pd.DataFrame(sa_rows)

    # ------------------------------------------------------------------
    # vle
    # ------------------------------------------------------------------
    activity_types = [
        "forumng", "oucontent", "url", "resource", "oucollaborate",
        "dataplus", "quiz", "ouelluminate", "sharedsubpage", "folder",
        "homepage", "subpage", "glossary",
    ]
    vle_rows = []
    site_id = 1
    for mod in modules:
        for pres in presentations:
            for act in activity_types:
                n_sites = rng.integers(2, 8)
                for _ in range(n_sites):
                    vle_rows.append({
                        "id_site": site_id,
                        "code_module": mod,
                        "code_presentation": pres,
                        "activity_type": act,
                        "week_from": rng.integers(1, 25),
                        "week_to": rng.integers(25, 39),
                    })
                    site_id += 1
    vle = pd.DataFrame(vle_rows)

    # ------------------------------------------------------------------
    # studentVle  (aggregated click counts per student per site)
    # ------------------------------------------------------------------
    svle_rows = []
    for _, row in student_info.sample(min(n_students, 2000), random_state=random_state).iterrows():
        sid = row["id_student"]
        mod = row["code_module"]
        pres = row["code_presentation"]
        result = row["final_result"]
        sites = vle[
            (vle["code_module"] == mod) & (vle["code_presentation"] == pres)
        ]["id_site"].values

        if result in ("Pass", "Distinction"):
            n_interactions = rng.integers(50, 200)
        elif result == "Fail":
            n_interactions = rng.integers(10, 100)
        else:
            n_interactions = rng.integers(1, 50)

        for _ in range(n_interactions):
            svle_rows.append({
                "code_module": mod,
                "id_student": sid,
                "id_site": rng.choice(sites),
                "code_presentation": pres,
                "date": rng.integers(1, 270),
                "sum_click": rng.integers(1, 20),
            })
    student_vle = pd.DataFrame(svle_rows)

    # ------------------------------------------------------------------
    # studentRegistration
    # ------------------------------------------------------------------
    sr_rows = []
    for _, row in student_info.iterrows():
        unregistration = None
        if row["final_result"] == "Withdrawn":
            unregistration = int(rng.integers(30, 180))
        sr_rows.append({
            "code_module": row["code_module"],
            "code_presentation": row["code_presentation"],
            "id_student": row["id_student"],
            "date_registration": int(rng.integers(-30, 1)),
            "date_unregistration": unregistration,
        })
    student_registration = pd.DataFrame(sr_rows)

    return {
        "studentInfo": student_info,
        "courses": courses,
        "assessments": assessments,
        "studentAssessment": student_assessment,
        "vle": vle,
        "studentVle": student_vle,
        "studentRegistration": student_registration,
    }


# ---------------------------------------------------------------------------
# OULAD data loader
# ---------------------------------------------------------------------------

class OULADLoader:
    """
    Loads and preprocesses the Open University Learning Analytics Dataset.

    If the CSV files are not found in *data_dir*, synthetic data matching
    the OULAD schema is generated automatically.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the OULAD CSV files.
    use_synthetic : bool
        Force use of synthetic data even if real files are found.
    n_synthetic : int
        Number of synthetic students when real data is not available.
    random_state : int
        Random seed for synthetic data generation and train/test splits.
    """

    REQUIRED_FILES = [
        "studentInfo.csv",
        "courses.csv",
        "assessments.csv",
        "studentAssessment.csv",
        "vle.csv",
        "studentVle.csv",
        "studentRegistration.csv",
    ]

    def __init__(
        self,
        data_dir: str = "data",
        use_synthetic: bool = False,
        n_synthetic: int = 5000,
        random_state: int = 42,
    ):
        self.data_dir = data_dir
        self.use_synthetic = use_synthetic
        self.n_synthetic = n_synthetic
        self.random_state = random_state
        self._tables: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "OULADLoader":
        """Load all tables (from CSV or synthetic generation)."""
        if self.use_synthetic or not self._real_data_available():
            print("Using synthetic OULAD data for demonstration.")
            self._tables = generate_synthetic_oulad(
                n_students=self.n_synthetic, random_state=self.random_state
            )
        else:
            print(f"Loading real OULAD data from '{self.data_dir}'.")
            self._tables = self._load_csv_files()
        return self

    def get_table(self, name: str) -> pd.DataFrame:
        """Return a copy of the named table."""
        return self._tables[name].copy()

    def build_master_dataset(self) -> pd.DataFrame:
        """
        Merge all tables into a single analysis-ready DataFrame.

        The resulting DataFrame has one row per student–module–presentation
        combination and includes:
          - demographic features from studentInfo
          - aggregated VLE engagement metrics
          - aggregated assessment scores
          - registration metadata
          - binary outcome label (1 = Pass/Distinction, 0 = Fail/Withdrawn)
        """
        student_info = self._tables["studentInfo"].copy()
        student_vle = self._tables["studentVle"].copy()
        student_assessment = self._tables["studentAssessment"].copy()
        assessments = self._tables["assessments"].copy()
        student_registration = self._tables["studentRegistration"].copy()

        # ---- VLE features -----------------------------------------------
        vle_features = self._aggregate_vle(student_vle)

        # ---- Assessment features ----------------------------------------
        assessment_features = self._aggregate_assessments(
            student_assessment, assessments
        )

        # ---- Registration features --------------------------------------
        reg_features = self._aggregate_registration(student_registration)

        # ---- Merge -------------------------------------------------------
        master = student_info.copy()
        master = master.merge(vle_features, on=["id_student", "code_module", "code_presentation"], how="left")
        master = master.merge(assessment_features, on=["id_student", "code_module", "code_presentation"], how="left")
        master = master.merge(reg_features, on=["id_student", "code_module", "code_presentation"], how="left")

        # ---- Encode outcome ---------------------------------------------
        master["outcome"] = master["final_result"].map(BINARY_RESULT_MAP)

        return master

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _real_data_available(self) -> bool:
        return all(
            os.path.exists(os.path.join(self.data_dir, f))
            for f in self.REQUIRED_FILES
        )

    def _load_csv_files(self) -> dict:
        tables = {}
        name_map = {
            "studentInfo": "studentInfo.csv",
            "courses": "courses.csv",
            "assessments": "assessments.csv",
            "studentAssessment": "studentAssessment.csv",
            "vle": "vle.csv",
            "studentVle": "studentVle.csv",
            "studentRegistration": "studentRegistration.csv",
        }
        for key, filename in name_map.items():
            path = os.path.join(self.data_dir, filename)
            tables[key] = pd.read_csv(path)
            print(f"  Loaded {filename}: {len(tables[key]):,} rows")
        return tables

    @staticmethod
    def _aggregate_vle(student_vle: pd.DataFrame) -> pd.DataFrame:
        """Compute per-student VLE engagement metrics."""
        if student_vle.empty:
            return pd.DataFrame(
                columns=[
                    "id_student", "code_module", "code_presentation",
                    "total_clicks", "n_unique_sites", "n_active_days",
                    "avg_clicks_per_day",
                ]
            )
        agg = (
            student_vle.groupby(["id_student", "code_module", "code_presentation"])
            .agg(
                total_clicks=("sum_click", "sum"),
                n_unique_sites=("id_site", "nunique"),
                n_active_days=("date", "nunique"),
            )
            .reset_index()
        )
        agg["avg_clicks_per_day"] = (
            agg["total_clicks"] / agg["n_active_days"].replace(0, np.nan)
        ).fillna(0)
        return agg

    @staticmethod
    def _aggregate_assessments(
        student_assessment: pd.DataFrame, assessments: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute per-student assessment summary metrics."""
        if student_assessment.empty:
            return pd.DataFrame(
                columns=[
                    "id_student", "code_module", "code_presentation",
                    "mean_score", "min_score", "max_score",
                    "n_submissions", "n_late_submissions",
                ]
            )
        merged = student_assessment.merge(
            assessments[["id_assessment", "code_module", "code_presentation", "date"]],
            on="id_assessment",
            how="left",
        )
        merged["is_late"] = (merged["date_submitted"] > merged["date"]).astype(int)
        agg = (
            merged.groupby(["id_student", "code_module", "code_presentation"])
            .agg(
                mean_score=("score", "mean"),
                min_score=("score", "min"),
                max_score=("score", "max"),
                n_submissions=("score", "count"),
                n_late_submissions=("is_late", "sum"),
            )
            .reset_index()
        )
        return agg

    @staticmethod
    def _aggregate_registration(student_registration: pd.DataFrame) -> pd.DataFrame:
        """Extract registration-based features."""
        reg = student_registration.copy()
        reg["is_withdrawn"] = reg["date_unregistration"].notna().astype(int)
        reg["early_registration"] = (reg["date_registration"] < 0).astype(int)
        return reg[
            [
                "id_student", "code_module", "code_presentation",
                "date_registration", "is_withdrawn", "early_registration",
            ]
        ]
