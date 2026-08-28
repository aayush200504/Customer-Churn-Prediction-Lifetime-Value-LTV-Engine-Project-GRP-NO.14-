# Project 1: Customer Churn Prediction & LTV Engine
## Technical Documentation — Week 1 & Week 2

**Team Size:** 7
**Dataset:** Telco Customer Churn Dataset (Kaggle / IBM Watson version)
**Tech Stack:** Python, PostgreSQL, SQLAlchemy, Pandas, Scikit-Learn, XGBoost, SHAP, FastAPI

---

## 1. Week 1 — Data Ingestion & Exploratory Data Analysis (EDA)

### 1.1 Objective
Set up a reproducible data pipeline that loads the raw Telco Customer Churn dataset into a relational database, and perform exploratory analysis to understand churn drivers before modeling.

### 1.2 Database Design

A `customers` table was designed in PostgreSQL to mirror the source CSV schema:

| Column | Type | Notes |
|---|---|---|
| customerID | VARCHAR (PK) | Unique customer identifier |
| gender | VARCHAR | Male / Female |
| SeniorCitizen | BOOLEAN/INT | 0 or 1 |
| Partner, Dependents | VARCHAR | Yes / No |
| tenure | INTEGER | Months with the company |
| PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | VARCHAR | Service subscription flags |
| Contract | VARCHAR | Month-to-month / One year / Two year |
| PaperlessBilling, PaymentMethod | VARCHAR | Billing details |
| MonthlyCharges | NUMERIC | Monthly billed amount |
| TotalCharges | NUMERIC | Cumulative billed amount |
| Churn | VARCHAR | Yes / No (target variable) |

**Rationale:** Keeping column types close to the source data avoids premature information loss; categorical encoding is deferred to the feature engineering stage (Week 2) so the raw layer stays analysis-friendly.

### 1.3 Ingestion Pipeline

- Built using **SQLAlchemy** as the ORM/connection layer and **Pandas** (`to_sql()`) for the bulk load.
- Pipeline steps:
  1. Read raw CSV with `pandas.read_csv()`.
  2. Validate column count and dtypes against the expected schema.
  3. Push to PostgreSQL via `to_sql('customers', engine, if_exists='replace', index=False)`.
  4. Log row count loaded vs. row count in source file, as a basic integrity check.
- Containerized PostgreSQL via Docker (`postgres` image) for environment consistency across all 7 members.

### 1.4 Data Quality Checks

Before EDA, the Data Analyst ran an audit covering:
- Column dtypes vs. expected types
- Null counts per column
- Unique value counts for categorical columns (to confirm expected category sets, e.g., `Contract` has exactly 3 values)
- Duplicate `customerID` check

**Key finding:** `TotalCharges` is stored as a string in the source file and contains blank entries for customers with `tenure = 0` (i.e., brand-new customers who haven't been billed yet). This required a type-coercion + handling step before it could be used numerically.

### 1.5 Exploratory Data Analysis

EDA was split by analysis area across the team to run in parallel rather than sequentially. Areas covered:

1. **Distributions:** Histograms and boxplots for `tenure`, `MonthlyCharges`, and `TotalCharges` — used to check skew and identify outliers.
2. **Churn by Contract type:** Bar chart of churn rate segmented by `Contract` (month-to-month / one year / two year).
3. **Churn by PaymentMethod:** Bar chart of churn rate by payment method.
4. **Churn by InternetService and add-ons:** Churn rate broken down by `InternetService`, `TechSupport`, and `OnlineSecurity` subscription status.
5. **Tenure vs. Churn:** Tenure bucketed into ranges (e.g., 0–6, 7–12, 13–24, 25+ months) with churn % plotted per bucket.
6. **Correlation heatmap:** Pairwise correlation across numeric fields (`tenure`, `MonthlyCharges`, `TotalCharges`) and churn (encoded as 0/1).
7. **TotalCharges data quality investigation:** Confirmed the blank-string rows correspond to `tenure = 0` customers, and checked consistency between `MonthlyCharges × tenure ≈ TotalCharges` for the remaining rows.

**Directional findings** (to be replaced with the team's actual computed values before final submission):

| Metric | Observation |
|---|---|
| Overall churn rate | ~26% of customers in the dataset are churners (imbalanced classification problem) |
| Contract type | Month-to-month customers churn at a markedly higher rate than one-year/two-year customers |
| Tenure | Churn risk is highest in the first 0–6 months and decreases sharply as tenure increases |
| TechSupport / OnlineSecurity | Customers without these add-ons churn more than those with them |
| MonthlyCharges | Higher monthly charges correlate with higher churn likelihood |

> **Note:** Replace the "Directional findings" table with the team's actual notebook output (exact percentages, chart references, and heatmap values) once the merged EDA notebook is finalized.

### 1.6 Data Cleaning & Preprocessing

- `TotalCharges`: coerced to numeric (`pd.to_numeric(errors='coerce')`); blank rows (tenure = 0, ~11 records) filled with 0 or dropped, and the fix was implemented in the ingestion pipeline itself (not just the notebook) so it is reproducible.
- Categorical encoding:
  - One-hot encoding applied to `Contract`, `InternetService`, `PaymentMethod` (implemented as a reusable `preprocessing.py` function under `src/features/`).
  - Binary encoding (Yes/No → 1/0) applied to `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, and similar flag columns.

### 1.7 Baseline Analytics Report

A baseline report was compiled summarizing:
- Overall churn %
- Average tenure and average charges by customer segment (contract type, tenure bucket, internet service)
- Churn-rate-by-segment tables

This report forms the foundation for the Week 4 dashboard.

### 1.8 API Skeleton (parallel workstream)

While the data team worked on ingestion and EDA, the Backend Engineer scaffolded the FastAPI project (`src/api/`) with a `/health` endpoint and Pydantic request/response models aligned to the finalized, cleaned column set — so Week 3's API development does not start from zero.

---

## 2. Week 2 — Feature Engineering & Predictive Modeling

### 2.1 Objective
Engineer predictive features from the cleaned dataset and train/evaluate classification models to predict customer churn.

### 2.2 Feature Engineering

New features derived from the raw/cleaned columns:

| Feature | Description |
|---|---|
| `avg_monthly_usage_vs_base_charge` | Customer's `MonthlyCharges` relative to the average for their contract type |
| `tenure_bucket` | Categorical bucketing of `tenure` (e.g., 0–6, 7–12, 13–24, 25+ months) |
| `total_services_subscribed` | Count of add-on services a customer is subscribed to |
| `charge_per_tenure_ratio` | `TotalCharges / tenure`, as a proxy for spend intensity |

### 2.3 Train/Test Split Strategy

- Data split into train and test sets using a **stratified split on `Churn`**, since the target is imbalanced (~26% positive class).
- Stratification ensures both sets preserve the same churn/non-churn ratio as the full dataset.

### 2.4 Model Training

Three classification models were trained for comparison:

1. **Logistic Regression** — baseline/interpretable model.
2. **Random Forest** — ensemble baseline with feature importance.
3. **XGBoost** — gradient-boosted trees, typically the strongest performer on tabular churn data.

**Class imbalance handling:** `class_weight='balanced'` and/or SMOTE oversampling was applied, since a model optimizing for raw accuracy on a 26%-positive dataset would default toward predicting "no churn" and still look accurate while missing most churners.

### 2.5 Model Evaluation

Models were compared on:
- **Precision, Recall, F1-score** (with particular emphasis on **recall for the `Churn = Yes` class**, since a missed churner is more costly to the business than a false alarm on a loyal customer)
- **ROC-AUC**

An evaluation rubric with target precision/recall/F1 thresholds was drafted in advance (Day 1–2) so model comparison in Week 2 had an objective bar to clear, rather than being judged purely on relative ranking.

**Model selection:** The best-performing model (commonly XGBoost or a tuned Random Forest on this dataset) was selected for downstream use in the LTV engine and API.

### 2.6 Explainability (SHAP)

- SHAP (SHapley Additive exPlanations) was run on the winning model to identify which features most influence churn predictions.
- A **SHAP summary plot** was generated, typically surfacing `Contract` type, `tenure`, `MonthlyCharges`, and `TechSupport` as the top drivers.
- A business-facing explanation was written translating the SHAP output into plain language, e.g.: *"Customers on month-to-month contracts with fewer than 6 months of tenure represent the highest churn risk segment."* This framing is intended for the stakeholder-facing presentation.

---

## 3. Team Contribution Summary (Week 1–2)

| Member | Week 1 Contribution | Week 2 Contribution |
|---|---|---|
| Team Lead | Repo/branch setup, README skeleton, correlation heatmap, EDA notebook merge, final report compilation | Coordination, model selection sign-off, business-facing SHAP summary |
| Data Engineer | PostgreSQL schema, ingestion pipeline, TotalCharges cleaning fix (in pipeline) | Support on feature pipeline integration |
| Data Analyst | Data quality audit, distribution plots, one-hot encoding function | Feature engineering support, train/test split |
| ML Engineer A | Churn-by-InternetService/add-ons analysis, binary encoding | Logistic Regression + Random Forest training |
| ML Engineer B | Tenure-vs-churn analysis, evaluation rubric draft | Model evaluation (precision/recall/F1/ROC-AUC), SHAP analysis |
| ML Engineer C | TotalCharges investigation, LTV approach design doc | XGBoost training support, feature engineering |
| Backend Engineer | FastAPI scaffold, Pydantic models, `/health` & `/schema` endpoints | API alignment with finalized feature set |

---

