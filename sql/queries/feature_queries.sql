-- Feature Queries for Risk Assessment

-- 1. Get complete customer risk profile
SELECT 
    c.customer_id,
    c.age,
    c.annual_income,
    c.years_employed,
    c.employment_status,
    c.home_ownership,
    ch.credit_score,
    ch.credit_utilization_ratio,
    ch.num_delinquent_accounts,
    ch.total_debt,
    ch.bankruptcy_flag,
    COUNT(DISTINCT t.transaction_id) as total_transactions,
    AVG(t.transaction_amount) as avg_transaction_amount,
    STDDEV(t.transaction_amount) as transaction_volatility,
    COUNT(DISTINCT la.loan_id) as total_loans,
    SUM(CASE WHEN la.default_flag = true THEN 1 ELSE 0 END) as defaulted_loans
FROM staging.customers c
LEFT JOIN staging.credit_history ch ON c.customer_id = ch.customer_id
LEFT JOIN staging.transactions t ON c.customer_id = t.customer_id
LEFT JOIN staging.loan_applications la ON c.customer_id = la.customer_id
WHERE c.customer_id = :customer_id
GROUP BY c.customer_id, c.age, c.annual_income, c.years_employed, 
         c.employment_status, c.home_ownership, ch.credit_score, 
         ch.credit_utilization_ratio, ch.num_delinquent_accounts,
         ch.total_debt, ch.bankruptcy_flag;

-- 2. Calculate aggregated transaction features
SELECT 
    customer_id,
    COUNT(*) as transaction_count,
    AVG(transaction_amount) as avg_monthly_spending,
    STDDEV(transaction_amount) as spending_volatility,
    MIN(transaction_amount) as min_transaction,
    MAX(transaction_amount) as max_transaction,
    SUM(CASE WHEN transaction_type = 'Debit' THEN transaction_amount ELSE 0 END) as total_debits,
    SUM(CASE WHEN transaction_type = 'Credit' THEN transaction_amount ELSE 0 END) as total_credits
FROM staging.transactions
WHERE customer_id = :customer_id
GROUP BY customer_id;

-- 3. Get recent loan application history
SELECT 
    loan_id,
    customer_id,
    application_date,
    loan_amount,
    loan_purpose,
    loan_term_months,
    interest_rate,
    debt_to_income_ratio,
    default_flag,
    days_past_due
FROM staging.loan_applications
WHERE customer_id = :customer_id
ORDER BY application_date DESC
LIMIT 5;

-- 4. Calculate credit utilization trend
SELECT 
    customer_id,
    AVG(credit_utilization_ratio) as avg_utilization,
    MAX(credit_utilization_ratio) as max_utilization,
    MIN(credit_utilization_ratio) as min_utilization
FROM staging.credit_history
WHERE customer_id = :customer_id
GROUP BY customer_id;

-- 5. Get high-risk customers
SELECT 
    rf.customer_id,
    rf.credit_score,
    rf.debt_to_income,
    rf.delinquency_score,
    rs.risk_score,
    rs.risk_category
FROM analytics.risk_features rf
JOIN predictions.risk_scores rs ON rf.customer_id = rs.customer_id
WHERE rs.risk_category = 'High'
    AND rs.prediction_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY rs.risk_score DESC
LIMIT 100;

-- 6. Calculate income-based loan capacity
SELECT 
    customer_id,
    annual_income,
    annual_income * 3 as max_recommended_loan,
    annual_income * 0.43 as max_monthly_payment,
    CASE 
        WHEN annual_income < 30000 THEN 'Low'
        WHEN annual_income < 60000 THEN 'Medium'
        WHEN annual_income < 100000 THEN 'High'
        ELSE 'Very High'
    END as income_category
FROM staging.customers
WHERE customer_id = :customer_id;

-- 7. Get customers for model retraining
SELECT 
    c.customer_id,
    c.age,
    c.annual_income,
    c.years_employed,
    c.num_dependents,
    ch.credit_score,
    ch.num_credit_accounts,
    ch.credit_utilization_ratio,
    ch.num_delinquent_accounts,
    ch.total_debt,
    ch.years_credit_history,
    la.loan_amount,
    la.loan_term_months,
    la.interest_rate,
    la.debt_to_income_ratio,
    la.default_flag
FROM staging.customers c
INNER JOIN staging.credit_history ch ON c.customer_id = ch.customer_id
INNER JOIN staging.loan_applications la ON c.customer_id = la.customer_id
WHERE la.application_date >= CURRENT_DATE - INTERVAL '2 years';

-- 8. Calculate employment stability metrics
SELECT 
    customer_id,
    age,
    years_employed,
    (years_employed::FLOAT / NULLIF(age, 0)) * 100 as employment_stability_score,
    CASE 
        WHEN years_employed >= 10 THEN 'Very Stable'
        WHEN years_employed >= 5 THEN 'Stable'
        WHEN years_employed >= 2 THEN 'Moderate'
        ELSE 'Unstable'
    END as employment_stability_category
FROM staging.customers
WHERE customer_id = :customer_id;

-- 9. Get customer risk trend over time
SELECT 
    customer_id,
    risk_score,
    risk_category,
    prediction_date,
    LAG(risk_score) OVER (PARTITION BY customer_id ORDER BY prediction_date) as previous_risk_score,
    risk_score - LAG(risk_score) OVER (PARTITION BY customer_id ORDER BY prediction_date) as risk_change
FROM predictions.risk_scores
WHERE customer_id = :customer_id
ORDER BY prediction_date DESC
LIMIT 12;

-- 10. Identify customers with improving credit
SELECT 
    customer_id,
    credit_score,
    num_delinquent_accounts,
    credit_utilization_ratio,
    created_at
FROM staging.credit_history
WHERE customer_id = :customer_id
ORDER BY created_at DESC
LIMIT 1;

-- 11. Get default rate by loan purpose
SELECT 
    loan_purpose,
    COUNT(*) as total_loans,
    SUM(CASE WHEN default_flag = true THEN 1 ELSE 0 END) as defaulted_loans,
    ROUND(SUM(CASE WHEN default_flag = true THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100, 2) as default_rate,
    AVG(loan_amount) as avg_loan_amount,
    AVG(interest_rate) as avg_interest_rate
FROM staging.loan_applications
GROUP BY loan_purpose
ORDER BY default_rate DESC;

-- 12. Get customers with high DTI but good credit
SELECT 
    c.customer_id,
    c.annual_income,
    ch.credit_score,
    la.debt_to_income_ratio,
    la.loan_amount
FROM staging.customers c
JOIN staging.credit_history ch ON c.customer_id = ch.customer_id
JOIN staging.loan_applications la ON c.customer_id = la.customer_id
WHERE ch.credit_score >= 700
    AND la.debt_to_income_ratio > 0.43
ORDER BY la.debt_to_income_ratio DESC
LIMIT 50;