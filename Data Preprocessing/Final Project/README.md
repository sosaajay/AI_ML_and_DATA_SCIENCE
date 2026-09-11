<div align="center">

# 📊 Holistic Data Preparer
### An end-to-end data preprocessing & feature engineering project

*Turning a messy, real-world credit-risk dataset into a clean, fully-numeric,
ML-ready table — one technique at a time.*

![Status](https://img.shields.io/badge/status-completed-brightgreen)
![Python](https://img.shields.io/badge/python-pandas%20%7C%20scikit--learn-blue)


</div>



---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Objectives](#-objectives)
3. [Dataset Sources](#-dataset-sources)
4. [Pipeline at a Glance](#-pipeline-at-a-glance)
5. [Techniques Implemented](#️-techniques-implemented)
6. [Best Methods Selected](#-best-methods-selected)
7. [Project Structure](#-project-structure)
8. [Output Files](#-output-files)
9. [Fixes Applied](#-fixes-applied-vs-the-original-draft)
10. [Key Learnings](#-key-learnings)
11. [Conclusion](#-conclusion)
12. [Author](#-author)

---

## 🚀 Overview

**Holistic Data Preparer** is a complete, beginner-friendly walkthrough of a real
data-preprocessing workflow — built to transform raw, messy data into a
machine-learning-ready format.

This project demonstrates practical preprocessing skills including:

| 🧩 Skill | What it solves |
|---|---|
| Data acquisition | Pulling data from CSV, JSON, SQL, and a live API |
| Data cleaning | Understanding and fixing data quality issues |
| Missing value handling | Filling in the gaps, the right way |
| Outlier treatment | Taming extreme values without losing data |
| Encoding | Turning text/categories into numbers |
| Feature engineering | Building smarter, more informative columns |
| Scaling | Putting every numeric column on a level playing field |
| Data transformations | Reshaping skewed columns for better modeling |

---

## 🎯 Objectives

- Load data from multiple sources
- Analyze data quality
- Handle missing values using multiple techniques
- Detect and treat outliers
- Encode categorical variables
- Perform feature scaling and transformations
- Generate final cleaned and scaled datasets

---

## 📁 Dataset Sources

This project uses data from **three different formats**, on purpose — to practice
sourcing real-world data the way it actually shows up in the wild:

- `customer_credit_risk.csv` — main customer table
- `customer_metadata.json` — supplementary customer metadata
- `loan_history.db` — historical loan payments (SQLite)

### 📂 Dataset Folder Structure

```bash
Dataset/
│
├── customer_credit_risk.csv
├── customer_metadata.json
└── loan_history.db
```

---

## 🧭 Pipeline at a Glance

```text
  Raw Data (CSV + JSON + SQL + API)
          │
          ▼
  Part C · Understand & Profile   →  data_quality_report.html
          │
          ▼
  Part D · Fix Missing Values     (compare 6 methods → pick best)
          │
          ▼
  Part E · Treat Outliers         (compare 4 methods → pick best)
          │
          ▼
  Part F · Encode + Engineer      (Label / Ordinal / One-Hot + new features)
          │
          ▼
  Part G · Scale + Transform      (StandardScaler, log, Yeo-Johnson…)
          │
          ▼
  Part H · Merge EVERYTHING into ONE pipeline
          │
          ├──▶  final_cleaned_dataset.csv   (unscaled, 0 missing values)
          └──▶  scaled_dataset.csv          (StandardScaler applied)
```


---

## 🛠️ Techniques Implemented

### 🔹 Missing Value Handling
- Complete Case Analysis
- Mean Imputation
- Most Frequent Imputation
- Missing Indicator
- Random Sample Imputation
- KNN Imputation
- MICE Imputation

### 🔹 Outlier Detection & Treatment
- Z-Score Method
- IQR Method
- Percentile Method
- Winsorization

### 🔹 Encoding Techniques
- Label Encoding
- Ordinal Encoding
- One-Hot Encoding
- Binarization

### 🔹 Feature Scaling
- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- RobustScaler
- Normalization

### 🔹 Data Transformations
- Log Transformation
- Square Root Transformation
- Reciprocal Transformation
- Box-Cox Transformation
- Yeo-Johnson Transformation

### 🔹 Feature Engineering
Custom features created:
- Debt-to-Income Ratio
- Spending-to-Income Ratio
- Monthly Transaction Metrics

---

## 📊 Best Methods Selected

After comparing multiple preprocessing techniques side by side, these were chosen
for the final pipeline:

| Step | Best Method | Why |
|---|---|---|
| Numerical Imputation | ✅ Mean Imputation | Missing values were moderate; distribution was fairly stable |
| Categorical Imputation | ✅ Most Frequent Imputation | Sensible default for low-cardinality category columns |
| Outlier Handling | ✅ Winsorization | Caps extreme values without deleting any rows |
| Encoding | ✅ One-Hot / Ordinal Encoding | Matches whether a category has a natural order or not |
| Scaling | ✅ StandardScaler | Puts all numeric features on a comparable scale for modeling |

---

## 📂 Project Structure

```bash
Final Project/
│
├── Dataset/
│   ├── customer_credit_risk.csv
│   ├── customer_metadata.json
│   └── loan_history.db
│
├── holistic_data_preparer.ipynb
├── final_cleaned_dataset.csv
├── scaled_dataset.csv
├── data_quality_report.html
├── theory_concepts.pdf
└── README.md
```

---

## 📈 Output Files

| File | Description |
|---|---|
| `final_cleaned_dataset.csv` | Fully cleaned, outlier-treated, encoded and feature-engineered — **35 columns, 0 missing values** |
| `scaled_dataset.csv` | Same as above, with numeric columns standardized (StandardScaler) for modeling |
| `data_quality_report.html` | Automated `ydata-profiling` report of the raw dataset |
| `theory_concepts.pdf` | Theory/definitions reference document (required by submission guidelines) |

---

## 🔧 Fixes Applied (vs. the original draft)

1. **Final export was incomplete** — the original `Part H` cell only imputed 4-5 columns
   and exported the dataset almost unchanged; all the encoding/binning/transformation/
   feature-construction work done in Parts E–G was never merged into the exported files.
   **Fixed:** `Part H` now builds one consolidated pipeline that merges *every* technique
   (imputation → outlier winsorization on all 3 flagged columns → Label/Ordinal/One-Hot
   encoding → date parts → binning/binarization/K-Means clustering → log/Yeo-Johnson
   transforms → engineered ratios) into `final_cleaned_dataset.csv` (35 columns,
   **0 missing values**, all numeric except `customer_id`) and `scaled_dataset.csv`.
2. **Outlier treatment was partial** — only `annual_income` was winsorized.
   **Fixed:** `loan_amount` and `credit_score` are now treated too, matching the brief.
3. **Missing theory PDF** — the submission guidelines require *"a document (PDF)
   explaining theory concepts with definitions"*. **Fixed:** added `theory_concepts.pdf`.
4. **External API call could break the notebook offline** — the live World Bank API
   call had no error handling. **Fixed:** wrapped in `try/except` with a documented
   dummy fallback response, so the notebook runs end-to-end with or without internet access.
5. **Duplicate profiling cell** — the data-quality report was being generated twice
   in a row, wasting run time. **Fixed:** removed the redundant duplicate cell.
6. **Hard to follow for beginners** — the notebook jumped straight into code with
   little context. **Fixed:** added short, plain-language notes before every section
   explaining *what* is being done and *why*, without changing any logic or output.

The notebook was fully re-executed top-to-bottom after these fixes with **zero errors**.

---

## 🧠 Key Learnings

- Handling real-world messy datasets
- Comparing preprocessing methods before picking one
- Feature engineering workflow
- Preparing datasets for machine learning pipelines

---

## 🚀 Conclusion

This project successfully transforms raw customer credit data into a clean,
structured, and machine-learning-ready dataset.

It demonstrates a holistic preprocessing workflow by evaluating multiple techniques
and selecting the most effective methods for the dataset — while staying approachable
enough for someone preparing their **first** preprocessing project.

---

## 🙌 Author

**Ajay sosa**
*Data Science | Machine Learning*
