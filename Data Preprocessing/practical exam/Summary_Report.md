# Summary Report — Customer Purchase Propensity Pipeline

## Theory

**Data Analysis** is the process of inspecting, cleaning, and modeling data to extract useful information and
support decisions. **A typical data science project** follows: problem framing → data acquisition → exploration
& cleaning → feature engineering → modeling → evaluation → deployment. This project covers the first four stages
for an e-commerce customer dataset, framed as a **binary classification problem**: predicting `purchased` (1/0)
from customer demographics, behavioral signals, and product context.

Multiple **missing value imputation strategies** exist because different columns fail "missing at random"
assumptions differently — Simple/Most-Frequent imputation suits independent columns with a dominant value, while
Missing Indicator + Random Sampling and multivariate methods (KNN, MICE) are needed when a column is skewed or
correlated with other features. Likewise, **outlier detection** methods (Z-score, IQR, Percentile) each assume
a different underlying distribution shape, and **Winsorization** offers a middle ground between removing rows
and ignoring extreme values entirely. **Encoding** choice (Label vs One-Hot vs Ordinal) depends on whether a
category has a natural order. **Scaling** matters because algorithms sensitive to feature magnitude (e.g.
distance-based or gradient-based models) can be biased by unscaled columns with very different ranges.

## Observations

**Techniques used:** Data was acquired from 4 sources (CSV, JSON, SQL via SQLite, and a simulated API) and
merged on `customer_id`/`product_id`. Missing values were resolved using Simple Imputer, Most Frequent
Imputation, Missing Indicator + Random Sample, KNN Imputer, and MICE. Outliers were treated using Z-score, IQR,
Percentile, and Winsorization. Categorical features were encoded with Label, One-Hot, and Ordinal Encoding;
numeric features were binned and binarized. Five scalers (Standard, MinMax, MaxAbs, Robust, Normalizer) and a
ColumnTransformer pipeline were demonstrated, along with FunctionTransformer/PowerTransformer transformations
and new interaction features (`purchase_per_day`, `days_since_last_purchase`).

**Biggest issues with the raw data:** Missingness was scattered independently across all 4 sources (age, gender,
income, satisfaction_level), so Complete Case Analysis would have discarded a large share of usable rows.
`income` and `price` were both heavily right-skewed with a handful of extreme injected outliers, which would
have distorted means and variances if left untreated. Dates were stored as plain strings, and customer IDs mixed
letters with numbers — both needed explicit parsing before they were usable.

**Which imputation and scaling worked best:** Missing Indicator + Random Sample was most effective for `income`,
since it preserved its true skewed shape better than a single mean/median fill. MICE gave the most statistically
consistent result for the jointly-correlated group of `income`, `total_purchases`, and `loyalty_points`. For
scaling, RobustScaler was the most reliable overall choice given residual skew in income and price even after
outlier treatment, while StandardScaler remains the right default inside the ColumnTransformer pipeline that
would feed a future linear/logistic model.
