-- Raw Data Schema (staging)
CREATE SCHEMA staging;

-- Customer Demographics
CREATE TABLE staging.customers (
    customer_id SERIAL PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    marital_status VARCHAR(20),
    education_level VARCHAR(50),
    employment_status VARCHAR(50),
    annual_income DECIMAL(12,2),
    years_employed INT,
    home_ownership VARCHAR(20),
    num_dependents INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Credit History
CREATE TABLE staging.credit_history (
    credit_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES staging.customers(customer_id),
    credit_score INT,
    num_credit_accounts INT,
    credit_utilization_ratio DECIMAL(5,4),
    num_delinquent_accounts INT,
    total_debt DECIMAL(12,2),
    bankruptcy_flag BOOLEAN,
    years_credit_history INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transaction History
CREATE TABLE staging.transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES staging.customers(customer_id),
    transaction_date DATE,
    transaction_amount DECIMAL(12,2),
    transaction_type VARCHAR(50),
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Loan Applications
CREATE TABLE staging.loan_applications (
    loan_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES staging.customers(customer_id),
    application_date DATE,
    loan_amount DECIMAL(12,2),
    loan_purpose VARCHAR(50),
    loan_term_months INT,
    interest_rate DECIMAL(5,4),
    monthly_payment DECIMAL(12,2),
    debt_to_income_ratio DECIMAL(5,4),
    default_flag BOOLEAN,
    days_past_due INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analytics Schema (processed data)
CREATE SCHEMA analytics;

-- Feature Store
CREATE TABLE analytics.risk_features (
    feature_id SERIAL PRIMARY KEY,
    customer_id INT,
    loan_id INT,
    
    -- Demographic features
    age INT,
    income_category VARCHAR(20),
    employment_stability_score DECIMAL(5,2),
    
    -- Credit features
    credit_score INT,
    credit_utilization DECIMAL(5,4),
    delinquency_score DECIMAL(5,2),
    
    -- Behavioral features
    avg_monthly_spending DECIMAL(12,2),
    transaction_volatility DECIMAL(5,2),
    payment_consistency_score DECIMAL(5,2),
    
    -- Derived features
    debt_to_income DECIMAL(5,4),
    loan_to_income_ratio DECIMAL(5,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Results Schema
CREATE SCHEMA predictions;

CREATE TABLE predictions.risk_scores (
    prediction_id SERIAL PRIMARY KEY,
    customer_id INT,
    loan_id INT,
    risk_score DECIMAL(5,4),
    risk_category VARCHAR(20), -- Low, Medium, High
    default_probability DECIMAL(5,4),
    model_version VARCHAR(50),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions.model_metadata (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    algorithm VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    training_date TIMESTAMP,
    features_used JSONB
);