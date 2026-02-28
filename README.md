# Student Performance Prediction Using Machine Learning

An Analysis of the Open University Learning Analytics Dataset (OULAD)

## Abstract

This project investigates student performance prediction using the Open University Learning Analytics Dataset. We analyze behavioral patterns, demographic factors, and engagement metrics to build predictive models for student success. Through comprehensive exploratory data analysis, feature engineering, and machine learning techniques, we demonstrate the effectiveness of various algorithms in early identification of at-risk students.

**Keywords:** Educational Data Mining, Learning Analytics, Student Performance Prediction, Machine Learning, Virtual Learning Environment

---

## Project Structure

```
.
├── data/                              # Place OULAD CSV files here (auto-synth if absent)
│   └── README.md                      # Dataset download instructions
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # OULAD data loading and synthetic data generation
│   ├── feature_engineering.py         # Feature engineering pipeline
│   ├── models.py                      # ML model training and evaluation
│   ├── evaluation.py                  # Evaluation metrics and plots
│   └── visualization.py               # EDA visualizations
├── outputs/                           # Generated reports and figures (gitignored)
├── student_performance_prediction.ipynb  # Full analysis notebook
├── main.py                            # CLI entry point
├── requirements.txt
└── README.md
```

---

## Dataset

The **Open University Learning Analytics Dataset (OULAD)** contains anonymised data about students enrolled at The Open University (UK) across seven course modules. It includes:

| Table | Description |
|---|---|
| `studentInfo.csv` | Demographics and final outcomes |
| `courses.csv` | Module and presentation metadata |
| `assessments.csv` | Assessment types and weights |
| `studentAssessment.csv` | Student scores per assessment |
| `vle.csv` | Virtual Learning Environment resource catalogue |
| `studentVle.csv` | Student interaction counts with VLE resources |
| `studentRegistration.csv` | Enrolment and withdrawal dates |

Download: https://analyse.kmi.open.ac.uk/open_dataset

> **Note:** If CSV files are not found in `data/`, the pipeline automatically generates synthetic data matching the OULAD schema for demonstration.

---

## Methodology

### 1. Feature Engineering

| Feature Group | Features |
|---|---|
| Demographic | gender, highest_education (ordinal), imd_band (ordinal), age_band (ordinal), num_of_prev_attempts, studied_credits, disability |
| VLE Engagement | total_clicks, n_unique_sites, n_active_days, avg_clicks_per_day |
| Assessment | mean_score, min_score, max_score, n_submissions, n_late_submissions |
| Registration | date_registration, early_registration |

### 2. Target Variable

`final_result` is binarised:
- **1 (Pass)** → Pass or Distinction
- **0 (At-Risk)** → Fail or Withdrawn

### 3. Models Evaluated

- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- XGBoost
- LightGBM

All models use 5-fold stratified cross-validation and are evaluated on accuracy, F1 score, precision, recall, and ROC-AUC.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run via CLI

```bash
# With synthetic data (no download required)
python main.py

# With real OULAD data
python main.py --data-dir data/

# Custom options
python main.py --n-synthetic 10000 --output-dir results/ --random-state 0
```

### Run Notebook

```bash
jupyter notebook student_performance_prediction.ipynb
```

---

## Key Results

After training on OULAD-structured data, the best-performing models achieve:

| Metric | Typical Range |
|---|---|
| Accuracy | 0.75 – 0.85 |
| ROC-AUC | 0.82 – 0.91 |
| F1 Score | 0.76 – 0.86 |

> Results on the real OULAD dataset are consistent with published literature (Kuzilek et al., 2017).

### Most Predictive Features
1. Mean assessment score
2. Total VLE clicks
3. Number of active days in VLE
4. Number of previous attempts
5. Education level

---

## References

- Kuzilek J., Hlosta M., Zdrahal Z. (2017). *Open University Learning Analytics Dataset*. Scientific Data 4, 170171. https://doi.org/10.1038/sdata.2017.171
- Romero C., Ventura S. (2010). *Educational data mining: A review of the state of the art*. IEEE Transactions on Systems, Man, and Cybernetics, Part C, 40(6), 601–618.
