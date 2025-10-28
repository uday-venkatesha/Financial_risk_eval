# Data Dictionary

## Overview
This document provides detailed information about all data tables, columns, and their relationships in the Financial Risk Assessment System.

---

## Staging Schema

### 1. staging.customers

Stores customer demographic and employment information.

| Column Name | Data Type | Nullable | Description | Example Values |
|------------|-----------|----------|-------------|----------------|
| customer_id | INTEGER | No | Unique customer identifier (Primary Key) | 1, 2, 3 |
| age | INTEGER | Yes | Customer age in years | 25, 45, 62 |
| gender | VARCHAR(10) | Yes | Customer gender | Male, Female, Other |
| marital_status | VARCHAR(20) | Yes | Marital status | Single, Married, Divorced, Widowed |
| education_level | VARCHAR(50) | Yes | Highest education level | High School, Bachelor, Master, PhD |
| employment_status | VARCHAR(50) | Yes | Current employment status | Employed, Self-Employed, Unemployed |
| annual_income | DECIMAL(12,2) | Yes | Annual income in USD | 45000.00, 85000.00 |
| years_employed | INTEGER | Yes | Years in current employment | 2, 10, 25 |
| home_ownership | VARCHAR(20) | Yes | Home ownership status | Own, Rent, Mortgage |
| num_dependents | INTEGER | Yes | Number of dependents | 0, 1, 3 |
| created_at | TIMESTAMP | No | Record creation timestamp | 2024-01-15 10:30:00 |

**Business Rules:**
- Age must be between 18 and 100
- Annual income must be positive
- Years employed cannot exceed (age - 18)

---

### 2. staging.credit_history

Contains credit-related information for customers.

| Column Name | Data Type | Nullable | Description | Example Values |
|------------|-----------|----------|-------------|----------------|
| credit_id | INTEGER | No | Unique credit record identifier (Primary Key) | 1, 2, 3 |
| customer_id | INTEGER | No | Foreign key to customers table | 1, 2, 3 |
| credit_score | INTEGER | Yes | FICO credit score | 650, 720, 800 |
| num_credit_accounts | INTEGER | Yes | Total number of credit accounts | 3, 7, 12 |
| credit_utilization_ratio | DECIMAL(5,4) | Yes | Credit utilization (0-1) | 0.35, 0.65, 0.90 |
| num_delinquent_accounts | INTEGER | Yes | Number of accounts with late payments | 0, 1, 3 |
| total_debt | DECIMAL(12,2) | Yes | Total outstanding debt in USD | 15000.00, 50000.00 |
| bankruptcy_flag | BOOLEAN | Yes | Has filed for bankruptcy | true, false |
| years_credit_history | INTEGER | Yes | Length of credit history in years | 5, 10, 20 |
| created_at | TIMESTAMP | No | Record creation timestamp | 2024-01-15 10:30:00 |

**Business Rules:**
- Credit score must be between 300 and 850
- Credit utilization ratio must be between 0 and 1
- Total debt must be non-negative
- Years credit history cannot exceed customer age - 18

**Relationships:**
- Many-to-One with staging.customers

---

### 3. staging.transactions

Transaction history for all customers.

| Column Name | Data Type | Nullable | Description | Example Values |
|------------|-----------|----------|-------------|----------------|
| transaction_id | INTEGER | No | Unique transaction identifier (Primary Key) | 1, 2, 3 |
| customer_id | INTEGER | No | Foreign key to customers table | 1, 2, 3 |
| transaction_date | DATE | Yes | Date of transaction | 2024-01-15 |
| transaction_amount | DECIMAL(12,2) | Yes | Transaction amount in USD | 50.00, 250.00, 1000.00 |
| transaction_type | VARCHAR(50) | Yes | Type of transaction | Debit, Credit |
| category | VARCHAR(50) | Yes | Transaction category | Groceries, Utilities, Entertainment, Healthcare, Transportation, Other |
| created_at | TIMESTAMP | No | Record creation timestamp | 2024-01-15 10:30:00 |

**Business Rules:**
- Transaction amount must be positive
- Transaction date cannot be in the future

**Relationships:**
- Many-to-One with staging.customers

---

### 4. staging.loan_applications

Loan application and performance data.

| Column Name | Data Type | Nullable | Description | Example Values |
|------------|-----------|----------|-------------|----------------|
| loan_id | INTEGER | No | Unique loan identifier (Primary Key) | 1, 2, 3 |
| customer_id | INTEGER | No | Foreign key to customers table | 1, 2, 3 |
| application_date | DATE | Yes | Date of loan application | 2024-01-15 |
| loan_amount | DECIMAL(12,2) | Yes | Requested loan amount in USD | 50000.00, 200000.00 |
| loan_purpose | VARCHAR(50) | Yes | Purpose of the loan | Home, Auto, Personal, Education, Business |
| loan_term_months | INTEGER | Yes | Loan term in months | 12, 24, 36, 48, 60, 72 |
| interest_rate | DECIMAL(5,4) | Yes | Annual interest rate | 0.05, 0.10, 0.20 |
| monthly_payment | DECIMAL(12,2) | Yes | Monthly payment amount | 500.00, 1200.00 |
| debt_to_income_ratio | DECIMAL(5,4) | Yes | Debt-to-income ratio | 0.25, 0.40, 0.55 |
| default_flag | BOOLEAN | Yes | Did customer default? (target variable) | true, false |
| days_past_due | INTEGER | Yes | Days payment is overdue | 0, 30, 90 |
| created_at | TIMESTAMP | No | Record creation timestamp | 2024-01-15 10:30:00 |

**Business Rules:**
- Loan amount must be positive
- Interest rate must be between 0 and 0.5 (0-50%)
- Loan term must be one of: 12, 24, 36, 48, 60, 72
- Days past due must be non-negative

**Relationships:**
- Many-to-One with staging.customers

---

## Analytics Schema

### 5. analytics.risk_features

Processed features for model training and prediction.

| Column Name | Data Type | Nullable | Description | Calculation/Source |
|------------|-----------|----------|-------------|-------------------|
| feature_id | INTEGER | No | Unique feature record identifier (Primary Key) | Auto-generated |
| customer_id | INTEGER | Yes | Customer identifier | From staging.customers |
| loan_id | INTEGER | Yes | Loan identifier | From staging.loan_applications |
| age | INTEGER | Yes | Customer age | From staging.customers |
| income_category | VARCHAR(20) | Yes | Income bracket | Derived: Low (<30K), Medium (30-60K), High (60-100K), Very High (>100K) |
| employment_stability_score | DECIMAL(5,2) | Yes | Employment stability metric | (years_employed / age) * 100 |
| credit_score | INTEGER | Yes | FICO credit score | From staging.credit_history |
| credit_utilization | DECIMAL(5,4) | Yes | Credit utilization ratio | From staging.credit_history |
| delinquency_score | DECIMAL(5,2) | Yes | Delinquency risk metric | (num_delinquent_accounts * 20) + ((1 - credit_utilization) * 30) |
| avg_monthly_spending | DECIMAL(12,2) | Yes | Average monthly transaction amount | AVG(transaction_amount) from staging.transactions |
| transaction_volatility | DECIMAL(5,2) | Yes | Standard deviation of transactions | STDDEV(transaction_amount) from staging.transactions |
| payment_consistency_score | DECIMAL(5,2) | Yes | Payment consistency metric | Calculated from transaction patterns |
| debt_to_income | DECIMAL(5,4) | Yes | Debt-to-income ratio | From staging.loan_applications |
| loan_to_income_ratio | DECIMAL(5,4) | Yes | Loan amount to income ratio | loan_amount / annual_income |
| created_at | TIMESTAMP | No | Record creation timestamp | Auto-generated |

**Purpose:** This table serves as the feature store for machine learning models.

---

## Predictions Schema

### 6. predictions.risk_scores

Stores model predictions for loan applications.

| Column Name | Data Type | Nullable | Description | Example Values |
|------------|-----------|----------|-------------|----------------|
| prediction_id | INTEGER | No | Unique prediction identifier (Primary Key) | 1, 2, 3 |
| customer_id | INTEGER | Yes | Customer identifier | 1, 2, 3 |
| loan_id | INTEGER | Yes | Loan identifier | 1, 2, 3 |
| risk_score | DECIMAL(5,4) | Yes | Risk score (0-100) | 25.5, 55.8, 85.2 |
| risk_category | VARCHAR(20) | Yes | Risk classification | Low, Medium, High |
| default_probability | DECIMAL(5,4) | Yes | Probability of default (0-1) | 0.255, 0.558, 0.852 |
| model_version | VARCHAR(50) | Yes | Version of model used | v1.0.0, v1.1.0 |
| prediction_date | TIMESTAMP | No | When prediction was made | 2024-01-15 10:30:00 |

**Risk Categories:**
- Low: risk_score < 30
- Medium: 30 ≤ risk_score < 60
- High: risk_score ≥ 60

---

### 7. predictions.model_metadata

Metadata and performance metrics for deployed models.

| Column Name | Data Type | Nullable | Description | Example Values |
|------------|-----------|----------|-------------|----------------|
| model_id | INTEGER | No | Unique model identifier (Primary Key) | 1, 2, 3 |
| model_name | VARCHAR(100) | Yes | Descriptive model name | Risk Prediction Model |
| model_version | VARCHAR(50) | Yes | Model version | v1.0.0 |
| algorithm | VARCHAR(50) | Yes | ML algorithm used | RandomForestClassifier, XGBoost |
| accuracy | DECIMAL(5,4) | Yes | Model accuracy | 0.873 |
| precision_score | DECIMAL(5,4) | Yes | Precision metric | 0.846 |
| recall | DECIMAL(5,4) | Yes | Recall metric | 0.792 |
| f1_score | DECIMAL(5,4) | Yes | F1 score | 0.818 |
| auc_roc | DECIMAL(5,4) | Yes | AUC-ROC score | 0.91 |
| training_date | TIMESTAMP | Yes | When model was trained | 2024-01-10 10:00:00 |
| features_used | JSONB | Yes | List of features used | JSON array of feature names |

**Purpose:** Tracks model versions and performance over time for monitoring and auditing.

---

## Derived Features Documentation

### Feature Engineering Calculations

#### 1. employment_stability_score
```python
employment_stability_score = (years_employed / age) * 100
```
**Interpretation:** Higher values indicate longer, more stable employment history relative to age.

#### 2. delinquency_score
```python
delinquency_score = (num_delinquent_accounts * 20) + ((1 - credit_utilization_ratio) * 30)
```
**Interpretation:** Higher values indicate better credit management (fewer delinquencies, lower utilization).

#### 3. loan_to_income_ratio
```python
loan_to_income_ratio = loan_amount / annual_income
```
**Interpretation:** Higher values indicate borrowing more relative to income (higher risk).

#### 4. debt_to_income_ratio
```python
monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term_months)
debt_to_income_ratio = (monthly_payment * 12) / annual_income
```
**Interpretation:** Percentage of annual income needed for debt payments. DTI > 0.43 considered high risk.

---

## Data Relationships

```
staging.customers (1) ──── (many) staging.credit_history
                 │
                 ├──── (many) staging.transactions
                 │
                 └──── (many) staging.loan_applications
                               │
                               └──── (1) analytics.risk_features
                                     │
                                     └──── (many) predictions.risk_scores
```

---

## Data Quality Standards

### Completeness
- Critical fields (customer_id, credit_score, loan_amount) must not be null
- Missing values in non-critical fields filled with: mean (numerical), mode (categorical)

### Accuracy
- All monetary values rounded to 2 decimal places
- Dates cannot be in the future
- Percentages and ratios must be between 0 and 1

### Consistency
- Customer IDs must exist in customers table before appearing in other tables
- All categorical values must match predefined sets

### Timeliness
- Data refreshed daily at 2 AM UTC
- Maximum data age: 7 days

---

## Glossary

| Term | Definition |
|------|------------|
| Credit Utilization | Percentage of available credit being used |
| DTI Ratio | Debt-to-Income ratio - monthly debt payments divided by gross monthly income |
| FICO Score | Credit score ranging from 300-850, higher is better |
| Default | Failure to repay a loan according to terms |
| Delinquency | Payment that is overdue |
| AUC-ROC | Area Under the Receiver Operating Characteristic curve - model performance metric |