# Customer Purchase Propensity - Data Cleaning and Feature Engineering Pipeline

A complete Data Preprocessing and Feature Engineering project designed
to prepare customer and transaction data for a future Machine Learning
classification model.

## Project Overview

This project simulates a real-world data preparation task for an
e-commerce company.

The objective is to collect customer and purchase-related information
from multiple raw data sources, combine the data using relevant keys,
analyze data quality, clean missing and abnormal values, transform
variables, encode categorical data, scale numerical features, construct
meaningful features, and finally generate a Machine Learning-ready
dataset.

The final goal is to prepare the data for a future model that can
predict whether a customer will make a purchase or not.

> **Important:** This project focuses on **Data Analysis, Data Cleaning,
> Data Preprocessing, and Feature Engineering**. Training and evaluating
> a Machine Learning model are outside the scope of this project.

------------------------------------------------------------------------

## Project Title

**Customer Purchase Propensity - Data Cleaning and Feature Engineering
Pipeline**

## Exam Information

-   **Exam Type:** Practical
-   **Duration:** 6 Hours
-   **Project Domain:** E-commerce / Customer Analytics
-   **Main Focus:** Data Preprocessing and Feature Engineering
-   **Future ML Problem:** Binary Classification

------------------------------------------------------------------------

## Problem Statement

An e-commerce company has customer information stored across different
data sources such as CSV files, JSON records, SQL tables, and an
external API.

Raw data can contain:

-   Missing values
-   Duplicate or inconsistent records
-   Numerical outliers
-   Different data types
-   Date and time values
-   Mixed-format identifiers
-   Categorical variables
-   Features with different scales
-   Skewed numerical distributions

Before this data can be used for Machine Learning, it must be properly
understood, cleaned, transformed, and prepared.

This project builds a complete preprocessing pipeline to convert raw
customer data into a clean and feature-engineered dataset suitable for a
future purchase prediction model.

------------------------------------------------------------------------

# Learning Objectives

By completing this project, the following concepts are demonstrated:

-   Understanding a complete Data Science preprocessing workflow
-   Importing data from CSV, JSON, SQL, and API sources
-   Combining multiple datasets using meaningful keys
-   Performing Exploratory Data Analysis (EDA)
-   Identifying missing values and selecting appropriate imputation
    methods
-   Detecting and handling numerical outliers
-   Working with date and time variables
-   Handling mixed-format variables
-   Encoding categorical variables
-   Scaling numerical features
-   Creating new interaction and derived features
-   Applying mathematical transformations
-   Performing binning and binarization
-   Creating a final Machine Learning-ready dataset

------------------------------------------------------------------------

# Data Sources

The project uses multiple data sources.

  Source   File / Endpoint                 Purpose
  -------- ------------------------------- -------------------------------
  CSV      `customers.csv`                 Customer demographics and IDs
  JSON     `transactions.json`             Customer transaction records
  SQL      `products.sql`                  Product information table
  API      `https://dummyjson.com/users`   Additional user details

### Main Keys

Datasets are combined using relevant identifiers such as:

-   `customer_id`
-   `product_id`

The exact merge strategy depends on the structure and relationships
present in the supplied datasets.

------------------------------------------------------------------------

# Project Workflow

The complete preprocessing workflow follows these major stages:

``` text
Raw Data Sources
      |
      v
Data Import
      |
      v
Data Understanding
      |
      v
Data Integration / Merging
      |
      v
Exploratory Data Analysis
      |
      v
Missing Value Handling
      |
      v
Outlier Detection & Handling
      |
      v
Date / Time Processing
      |
      v
Categorical Encoding
      |
      v
Feature Scaling
      |
      v
Feature Construction & Transformation
      |
      v
Binning & Binarization
      |
      v
Final Clean Dataset
      |
      v
processed_customer_data.csv
```

------------------------------------------------------------------------

# Step 1 - Project Planning and Problem Framing

## What is Data Analysis?

Data Analysis is the process of inspecting, cleaning, transforming, and
interpreting data to discover useful information, patterns,
relationships, and insights.

It helps understand the quality and structure of data before further
analytical or Machine Learning work.

## Typical Steps in a Data Science Project

A general Data Science workflow includes:

1.  Problem Understanding
2.  Data Collection
3.  Data Understanding
4.  Data Cleaning
5.  Exploratory Data Analysis
6.  Feature Engineering
7.  Feature Selection
8.  Model Building
9.  Model Evaluation
10. Deployment and Monitoring

In this project, the primary focus is on steps related to **data
understanding, cleaning, preprocessing, and feature engineering**.

## Machine Learning Problem

The prepared dataset is intended for a future **binary classification**
problem.

Example target:

``` text
purchased = 1  -> Customer purchased
purchased = 0  -> Customer did not purchase
```

The preprocessing pipeline prepares the input features so they can later
be used by a classification model.

------------------------------------------------------------------------

# Step 2 - Data Import and Understanding

The first stage is to load all available data sources.

## Data Loading

The project works with:

-   CSV data
-   JSON data
-   SQL data using SQLite
-   API data

After loading the datasets, their structure is inspected using common
Pandas operations such as:

``` python
df.info()
df.describe()
df.head()
df.shape
df.columns
df.isnull().sum()
```

## Data Integration

The separate datasets are combined using relevant keys.

Important keys include:

``` text
customer_id
product_id
```

The purpose of merging the datasets is to create a combined view
containing customer, transaction, and product-related information.

## Initial Data Understanding

The following points are checked:

-   Number of rows and columns
-   Column names
-   Data types
-   Missing values
-   Duplicate records
-   Numerical statistics
-   Categorical values
-   Possible inconsistent values
-   Relationships between datasets

------------------------------------------------------------------------

# Step 3 - Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed before extensive preprocessing.

EDA helps identify patterns, distributions, relationships, and potential
data-quality problems.

## 3.1 Univariate Analysis

Univariate analysis studies one variable at a time.

The project includes:

-   Numerical feature distributions
-   Histograms
-   Detection of skewed variables
-   Summary statistics
-   Automated profiling using Pandas Profiling / YData Profiling where
    applicable

Example numerical analysis:

``` python
df.describe()
```

Histograms can be used to understand:

-   Central tendency
-   Spread
-   Skewness
-   Possible extreme values

------------------------------------------------------------------------

## 3.2 Bivariate Analysis

Bivariate analysis studies relationships between two variables.

Examples include:

-   Income vs Purchase
-   Numerical features vs `purchased`
-   Grouped mean comparisons
-   Correlation between numerical variables

The target variable can be represented as:

``` text
purchased = 0
purchased = 1
```

This helps identify whether customer characteristics are associated with
purchasing behavior.

------------------------------------------------------------------------

## 3.3 Multivariate Analysis

Multivariate analysis examines relationships among multiple variables.

Possible techniques include:

-   Pair plots
-   Correlation heatmaps
-   Grouped statistics
-   Multiple-feature comparisons

The goal is to understand how different customer and transaction
features interact with each other.

------------------------------------------------------------------------

# Step 4 - Handling Missing Data

Missing values are identified and handled using multiple techniques.

The appropriate technique depends on the variable type and the reason
for missingness.

## Techniques Used

### 1. Simple Imputer

Simple imputation can be applied to numerical and categorical data.

Common strategies include:

-   Mean
-   Median
-   Most frequent
-   Constant

Example:

``` python
from sklearn.impute import SimpleImputer
```

------------------------------------------------------------------------

### 2. Most Frequent Imputation

Most frequent imputation is useful for categorical variables.

The most commonly occurring category is used to fill missing values.

------------------------------------------------------------------------

### 3. Missing Indicator

A missing-value indicator can be created to preserve information about
whether the original value was missing.

Example concept:

``` text
income_missing = 1
```

indicates that the original income value was missing.

------------------------------------------------------------------------

### 4. Random Sample Imputation

Random sample imputation fills a missing value using a randomly selected
value from the existing distribution of the same variable.

This can help preserve the original distribution better than some basic
imputation methods.

------------------------------------------------------------------------

### 5. KNN Imputer

K-Nearest Neighbors imputation estimates missing values using similar
observations.

``` python
from sklearn.impute import KNNImputer
```

It is useful when multiple features contain meaningful relationships.

------------------------------------------------------------------------

### 6. MICE

MICE stands for **Multiple Imputation by Chained Equations**.

It estimates missing values using relationships among multiple
variables.

It is useful when missing values are associated with other correlated
features.

------------------------------------------------------------------------

### 7. Complete Case Analysis

Complete Case Analysis removes rows containing missing values.

It is useful as a comparison technique, but excessive row deletion can
reduce the amount of available data.

------------------------------------------------------------------------

# Step 5 - Outlier Detection and Handling

Outliers are observations that are unusually far from the general
distribution of a variable.

The project applies multiple outlier detection techniques.

## 5.1 Z-Score Method

The Z-score measures how far a value is from the mean in terms of
standard deviations.

General interpretation:

``` text
Z = (x - mean) / standard deviation
```

Large absolute Z-scores can indicate potential outliers.

------------------------------------------------------------------------

## 5.2 IQR Method

IQR stands for **Interquartile Range**.

``` text
IQR = Q3 - Q1
```

Typical outlier boundaries are:

``` text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these limits can be treated as potential outliers.

------------------------------------------------------------------------

## 5.3 Percentile Method

The percentile method identifies extreme observations using selected
lower and upper percentiles.

For example:

``` text
Lower percentile = 1%
Upper percentile = 99%
```

The exact thresholds should be selected according to the dataset.

------------------------------------------------------------------------

## 5.4 Winsorization

Winsorization caps extreme values at selected percentile boundaries
instead of deleting the observations.

This can reduce the influence of extreme values while retaining the
rows.

------------------------------------------------------------------------

# Step 6 - Handling Mixed and Date/Time Variables

Date and time features are converted into proper datetime format.

Important variables include:

``` text
signup_date
last_purchase_date
```

Example:

``` python
pd.to_datetime(df["signup_date"])
```

## Derived Date Feature

A meaningful feature can be created:

``` text
days_since_last_purchase
```

This represents the number of days since the customer's last purchase.

Date features can provide useful behavioral information for future
purchase prediction.

------------------------------------------------------------------------

## Mixed Variables

Some variables may contain multiple types of information.

For example, customer IDs may contain both letters and numbers:

``` text
CUS001
CUST-102
USERA45
```

These variables should be handled carefully so that identifiers are not
incorrectly treated as ordinary numerical features.

------------------------------------------------------------------------

# Step 7 - Encoding Categorical Data

Machine Learning algorithms generally require numerical representations
of categorical variables.

The project demonstrates multiple encoding methods.

## 7.1 Label Encoding

Label Encoding converts categories into integer values.

Example:

``` text
Male   -> 0
Female -> 1
```

It is most appropriate when categories have a meaningful ordering or
when used carefully for suitable variables.

------------------------------------------------------------------------

## 7.2 One-Hot Encoding

One-Hot Encoding creates a separate binary column for each category.

Example:

``` text
City = Rajkot
```

may become:

``` text
City_Rajkot = 1
City_Ahmedabad = 0
City_Surat = 0
```

It is commonly used for nominal categorical variables.

------------------------------------------------------------------------

## 7.3 Ordinal Encoding

Ordinal Encoding is useful when categories have a natural order.

Example:

``` text
Low       -> 0
Medium    -> 1
High      -> 2
```

Possible applications include:

-   Education levels
-   Satisfaction levels
-   Customer categories with a genuine order

------------------------------------------------------------------------

## 7.4 Numerical Feature Encoding / Binning

Numerical variables can also be converted into meaningful groups.

For example, income can be divided into groups such as:

``` text
Low Income
Medium Income
High Income
```

This process is known as **binning**.

------------------------------------------------------------------------

# Step 8 - Feature Scaling

Feature scaling makes numerical variables comparable when their original
ranges differ significantly.

The project demonstrates multiple scaling techniques.

## 8.1 StandardScaler

StandardScaler transforms features so they generally have:

``` text
Mean ≈ 0
Standard Deviation ≈ 1
```

------------------------------------------------------------------------

## 8.2 MinMaxScaler

MinMaxScaler generally transforms values into a selected range,
commonly:

``` text
0 to 1
```

------------------------------------------------------------------------

## 8.3 MaxAbsScaler

MaxAbsScaler scales values based on their maximum absolute value.

It is especially useful when preserving sparsity is important.

------------------------------------------------------------------------

## 8.4 RobustScaler

RobustScaler uses statistics such as the median and interquartile range.

It is less affected by extreme outliers than standard mean-based
scaling.

------------------------------------------------------------------------

## 8.5 Normalizer

Normalizer scales individual observations based on their vector norm.

It is useful for specific datasets where the relative direction of
feature vectors is important.

------------------------------------------------------------------------

## ColumnTransformer

`ColumnTransformer` can apply different preprocessing operations to
different groups of columns.

Conceptually:

``` text
Numerical Columns
       |
       +--> Imputation
       +--> Scaling

Categorical Columns
       |
       +--> Imputation
       +--> Encoding
```

This creates a structured preprocessing pipeline.

------------------------------------------------------------------------

# Step 9 - Feature Construction and Transformation

Feature engineering creates new variables that may provide more useful
information than the original raw columns.

## 9.1 Interaction / Ratio Features

An example feature from the project requirement is:

``` text
purchase_per_day = total_purchases / days_since_signup
```

This can represent purchase activity relative to the customer's time
with the platform.

Zero or invalid denominators should be handled before calculating ratio
features.

------------------------------------------------------------------------

## 9.2 FunctionTransformer

`FunctionTransformer` can be used to apply custom mathematical
transformations.

Examples include:

-   Log transformation
-   Square-root transformation
-   Reciprocal transformation

These transformations can help manage skewed distributions.

------------------------------------------------------------------------

## 9.3 PowerTransformer

Power transformations can reduce skewness and make numerical
distributions more suitable for statistical and Machine Learning
techniques.

The project demonstrates:

-   Box-Cox transformation
-   Yeo-Johnson transformation

### Box-Cox

Box-Cox generally requires positive input values.

### Yeo-Johnson

Yeo-Johnson is more flexible because it can handle zero and negative
values.

------------------------------------------------------------------------

# Step 10 - Binning and Binarization

## Equal-Width / Quantile Binning

The `income` feature can be divided into groups.

Binning converts a continuous numerical variable into meaningful
categories.

Examples:

``` text
Low
Medium
High
```

Quantile-based binning attempts to create groups based on the
distribution of observations.

------------------------------------------------------------------------

## Purchase Frequency Binarization

Purchase frequency can be converted into a binary indicator.

Example:

``` text
frequent_buyer = 1  if purchase frequency > threshold
frequent_buyer = 0  otherwise
```

The threshold should be clearly defined in the notebook based on the
dataset and project logic.

------------------------------------------------------------------------

# Final Output

After completing all preprocessing and feature engineering steps, the
final dataset is exported as:

``` text
processed_customer_data.csv
```

The final dataset should contain:

-   Cleaned data
-   Handled missing values
-   Handled outliers where appropriate
-   Processed date features
-   Encoded categorical variables
-   Scaled numerical features where required
-   Engineered features
-   Transformed variables
-   Binned / binary features where applicable

The final output is intended to be suitable as input for a future
Machine Learning classification pipeline.

------------------------------------------------------------------------

# Project Deliverables

The project requirements specify the following deliverables:

### 1. Python Notebook

``` text
DataPreprocessing.ipynb
```

The notebook contains the complete preprocessing workflow, analysis,
transformations, and observations.

### 2. Final CSV Output

``` text
processed_customer_data.csv
```

This is the final cleaned and feature-engineered dataset.

### 3. Summary Report

A short report containing:

-   Techniques used
-   Major problems found in the raw data
-   Missing-value handling approach
-   Outlier handling approach
-   Scaling techniques used
-   Observations from EDA
-   Comparison of preprocessing approaches
-   Final conclusions

### 4. GitHub Repository

All project files should be maintained inside a GitHub repository,
including this `README.md`.

------------------------------------------------------------------------

# Suggested Project Structure

``` text
Customer-Purchase-Propensity/
│
├── DataPreprocessing.ipynb
│
├── processed_customer_data.csv
│
├── README.md
│
├── Summary_Report.md
│
├── data/
│   ├── customers.csv
│   ├── transactions.json
│   └── products.sql
│
└── outputs/
    └── processed_customer_data.csv
```

The actual folder structure can be adjusted according to the files
included in the project repository.

------------------------------------------------------------------------

# Technologies and Tools

## Programming Language

-   Python

## Main Libraries

-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   YData Profiling / Pandas Profiling
-   SQLite / SQLite3

## Development Environment

-   Jupyter Notebook
-   VS Code
-   Git
-   GitHub

------------------------------------------------------------------------

# Important Python Concepts Used

The project demonstrates practical use of:

-   DataFrame operations
-   Data type conversion
-   Missing-value detection
-   Statistical summaries
-   GroupBy operations
-   Merging and joining datasets
-   Datetime processing
-   Data visualization
-   Imputation
-   Outlier detection
-   Encoding
-   Scaling
-   Transformation
-   Feature engineering
-   Exporting data to CSV

------------------------------------------------------------------------

# Example Preprocessing Pipeline

A conceptual preprocessing pipeline can be organized as:

``` text
1. Load raw data
       ↓
2. Inspect structure
       ↓
3. Clean column names and data types
       ↓
4. Merge datasets
       ↓
5. Perform EDA
       ↓
6. Identify missing values
       ↓
7. Apply appropriate imputation
       ↓
8. Detect and handle outliers
       ↓
9. Convert date/time variables
       ↓
10. Create date-based features
       ↓
11. Encode categorical variables
       ↓
12. Scale numerical variables
       ↓
13. Create interaction features
       ↓
14. Apply transformations
       ↓
15. Perform binning/binarization
       ↓
16. Validate final dataset
       ↓
17. Export processed_customer_data.csv
```

------------------------------------------------------------------------

# Data Quality Checks

Before exporting the final dataset, the following checks should be
performed:

``` python
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.shape
```

Additional checks should confirm:

-   Expected columns are present
-   Data types are correct
-   Missing values are handled
-   Duplicate records are reviewed
-   Numerical features contain valid values
-   Date features are correctly converted
-   Encoded columns contain appropriate values
-   The final dataset is suitable for downstream ML processing

------------------------------------------------------------------------

# Key Challenges Addressed

This project addresses several common real-world data problems:

### Multiple Data Sources

Customer information comes from CSV, JSON, SQL, and API sources.

### Missing Data

Different imputation strategies are demonstrated instead of blindly
deleting missing rows.

### Outliers

Multiple statistical methods are considered to identify extreme
observations.

### Date Variables

Raw date fields are converted into useful numerical features.

### Categorical Variables

Different encoding strategies are selected according to the type of
categorical feature.

### Different Numerical Scales

Scaling methods are compared and applied according to preprocessing
requirements.

### Skewed Data

Mathematical and power transformations are used to reduce skewness where
appropriate.

### Feature Engineering

Raw variables are converted into more informative features for future
predictive modeling.

------------------------------------------------------------------------

# Expected Outcome

At the end of the project, the raw multi-source customer data should be
converted into a structured, clean, and feature-engineered dataset.

The resulting file:

``` text
processed_customer_data.csv
```

can then be used as the foundation for a future binary classification
model that predicts customer purchase behavior.

------------------------------------------------------------------------

# Limitations

This project intentionally does not focus on:

-   Training a Machine Learning model
-   Hyperparameter tuning
-   Model comparison
-   Cross-validation
-   Model deployment
-   Production monitoring

The primary objective is the **data preprocessing and feature
engineering pipeline**.

------------------------------------------------------------------------

# Conclusion

The **Customer Purchase Propensity - Data Cleaning and Feature
Engineering Pipeline** demonstrates how raw, multi-source e-commerce
data can be transformed into a structured dataset suitable for Machine
Learning.

The project covers the complete preprocessing journey, including:

-   Data import
-   Data understanding
-   Dataset integration
-   Exploratory Data Analysis
-   Missing-value handling
-   Outlier detection
-   Date/time processing
-   Categorical encoding
-   Feature scaling
-   Feature transformation
-   Feature construction
-   Binning and binarization
-   Final dataset export

The final processed dataset provides a strong foundation for the next
stage of the Data Science workflow: building a model that can predict
whether a customer is likely to make a purchase.

------------------------------------------------------------------------

# Author

**Ajay Sosa**

Data Science / AI-ML Learning Project

------------------------------------------------------------------------

# Project Focus

``` text
Data Cleaning
      +
Exploratory Data Analysis
      +
Data Preprocessing
      +
Feature Engineering
      =
Machine Learning Ready Dataset
```
