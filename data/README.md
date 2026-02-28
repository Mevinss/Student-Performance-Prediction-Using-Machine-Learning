# OULAD Data Directory

Place the Open University Learning Analytics Dataset CSV files here to use real data.

## Download

The dataset is freely available at:
https://analyse.kmi.open.ac.uk/open_dataset

## Required Files

```
data/
├── studentInfo.csv
├── courses.csv
├── assessments.csv
├── studentAssessment.csv
├── vle.csv
├── studentVle.csv
└── studentRegistration.csv
```

## Synthetic Data Fallback

If the CSV files are not present, the pipeline automatically generates synthetic data
that mirrors the OULAD schema for demonstration purposes.
