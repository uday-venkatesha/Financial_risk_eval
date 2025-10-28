-- Reporting Queries for Risk Assessment Dashboard

-- 1. Daily Risk Assessment Summary
SELECT 
    DATE(prediction_date) as report_date,
    COUNT(*) as total_assessments,
    COUNT(CASE WHEN risk_category = 'Low' THEN 1 END) as low_risk_count,
    COUNT(CASE WHEN risk_category = 'Medium' THEN 1 END) as medium_risk_count,
    COUNT(CASE WHEN risk_category = 'High' THEN 1 END) as high_risk_count,
    ROUND(AVG(risk_score), 2) as avg_risk_score,
    ROUND(AVG(default_probability), 4) as avg_default_probability
FROM predictions.risk_scores
WHERE prediction_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(prediction_date)
ORDER BY report_date DESC;

-- 2. Model Performance Metrics
SELECT 
    model_name,
    model_version,
    algorithm,
    ROUND(accuracy::NUMERIC, 4) as accuracy,
    ROUND(precision_score::NUMERIC, 4) as precision,
    ROUND(recall::NUMERIC, 4) as recall,
    ROUND(f1_score::NUMERIC, 4) as f1_score,
    ROUND(auc_roc::NUMERIC, 4) as auc_roc,
    training_date
FROM predictions.model_metadata
ORDER BY training_date DESC
LIMIT 5;

-- 3. Risk Distribution by Demographics
SELECT 
    CASE 
        WHEN c.age < 30 THEN '18-29'
        WHEN c.age < 40 THEN '30-39'
        WHEN c.age < 50 THEN '40-49'
        WHEN c.age < 60 THEN '50-59'
        ELSE '60+'
    END as age_group,
    c.employment_status,
    COUNT(*) as customer_count,
    ROUND(AVG(rs.risk_score), 2) as avg_risk_score,
    COUNT(CASE WHEN rs.risk_category = 'High' THEN 1 END) as high_risk_count
FROM staging.customers c
JOIN predictions.risk_scores rs ON c.customer_id = rs.customer_id
WHERE rs.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY age_group, c.employment_status
ORDER BY age_group, c.employment_status;

-- 4. Loan Performance by Amount Range
SELECT 
    CASE 
        WHEN loan_amount < 25000 THEN '0-25K'
        WHEN loan_amount < 50000 THEN '25K-50K'
        WHEN loan_amount < 100000 THEN '50K-100K'
        WHEN loan_amount < 200000 THEN '100K-200K'
        ELSE '200K+'
    END as loan_range,
    COUNT(*) as total_loans,
    SUM(CASE WHEN default_flag = true THEN 1 ELSE 0 END) as defaults,
    ROUND(AVG(CASE WHEN default_flag = true THEN 1 ELSE 0 END)::NUMERIC * 100, 2) as default_rate,
    ROUND(AVG(interest_rate)::NUMERIC * 100, 2) as avg_interest_rate,
    ROUND(AVG(debt_to_income_ratio)::NUMERIC, 2) as avg_dti
FROM staging.loan_applications
GROUP BY loan_range
ORDER BY loan_range;

-- 5. Credit Score vs Default Rate
SELECT 
    CASE 
        WHEN ch.credit_score < 600 THEN '300-599'
        WHEN ch.credit_score < 650 THEN '600-649'
        WHEN ch.credit_score < 700 THEN '650-699'
        WHEN ch.credit_score < 750 THEN '700-749'
        ELSE '750+'
    END as credit_score_range,
    COUNT(DISTINCT la.loan_id) as total_loans,
    SUM(CASE WHEN la.default_flag = true THEN 1 ELSE 0 END) as defaults,
    ROUND(AVG(CASE WHEN la.default_flag = true THEN 1 ELSE 0 END)::NUMERIC * 100, 2) as default_rate
FROM staging.credit_history ch
JOIN staging.loan_applications la ON ch.customer_id = la.customer_id
GROUP BY credit_score_range
ORDER BY credit_score_range;

-- 6. Top Risk Factors Analysis
SELECT 
    'High DTI' as risk_factor,
    COUNT(*) as customer_count,
    ROUND(AVG(rs.risk_score), 2) as avg_risk_score
FROM analytics.risk_features rf
JOIN predictions.risk_scores rs ON rf.customer_id = rs.customer_id
WHERE rf.debt_to_income > 0.43
    AND rs.prediction_date >= CURRENT_DATE - INTERVAL '7 days'

UNION ALL

SELECT 
    'Low Credit Score' as risk_factor,
    COUNT(*) as customer_count,
    ROUND(AVG(rs.risk_score), 2) as avg_risk_score
FROM analytics.risk_features rf
JOIN predictions.risk_scores rs ON rf.customer_id = rs.customer_id
WHERE rf.credit_score < 650
    AND rs.prediction_date >= CURRENT_DATE - INTERVAL '7 days'

UNION ALL

SELECT 
    'High Delinquency' as risk_factor,
    COUNT(*) as customer_count,
    ROUND(AVG(rs.risk_score), 2) as avg_risk_score
FROM staging.credit_history ch
JOIN predictions.risk_scores rs ON ch.customer_id = rs.customer_id
WHERE ch.num_delinquent_accounts >= 2
    AND rs.prediction_date >= CURRENT_DATE - INTERVAL '7 days'

UNION ALL

SELECT 
    'Bankruptcy History' as risk_factor,
    COUNT(*) as customer_count,
    ROUND(AVG(rs.risk_score), 2) as avg_risk_score
FROM staging.credit_history ch
JOIN predictions.risk_scores rs ON ch.customer_id = rs.customer_id
WHERE ch.bankruptcy_flag = true
    AND rs.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY avg_risk_score DESC;

-- 7. Monthly Trend Analysis
SELECT 
    DATE_TRUNC('month', prediction_date) as month,
    COUNT(*) as total_predictions,
    ROUND(AVG(risk_score), 2) as avg_risk_score,
    COUNT(CASE WHEN risk_category = 'High' THEN 1 END) as high_risk_count,
    ROUND(COUNT(CASE WHEN risk_category = 'High' THEN 1 END)::NUMERIC / COUNT(*) * 100, 2) as high_risk_percentage
FROM predictions.risk_scores
WHERE prediction_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', prediction_date)
ORDER BY month DESC;

-- 8. Loan Purpose Risk Analysis
SELECT 
    la.loan_purpose,
    COUNT(*) as total_applications,
    SUM(CASE WHEN la.default_flag = true THEN 1 ELSE 0 END) as defaults,
    ROUND(AVG(CASE WHEN la.default_flag = true THEN 1 ELSE 0 END)::NUMERIC * 100, 2) as default_rate,
    ROUND(AVG(rs.risk_score), 2) as avg_predicted_risk,
    ROUND(AVG(la.loan_amount), 2) as avg_loan_amount,
    ROUND(AVG(la.interest_rate)::NUMERIC * 100, 2) as avg_interest_rate
FROM staging.loan_applications la
LEFT JOIN predictions.risk_scores rs ON la.customer_id = rs.customer_id
GROUP BY la.loan_purpose
ORDER BY default_rate DESC;

-- 9. Customer Segment Performance
SELECT 
    CASE 
        WHEN c.annual_income < 30000 THEN 'Low Income'
        WHEN c.annual_income < 60000 THEN 'Medium Income'
        WHEN c.annual_income < 100000 THEN 'High Income'
        ELSE 'Very High Income'
    END as income_segment,
    CASE 
        WHEN ch.credit_score < 650 THEN 'Poor Credit'
        WHEN ch.credit_score < 700 THEN 'Fair Credit'
        WHEN ch.credit_score < 750 THEN 'Good Credit'
        ELSE 'Excellent Credit'
    END as credit_segment,
    COUNT(DISTINCT c.customer_id) as customer_count,
    ROUND(AVG(rs.risk_score), 2) as avg_risk_score,
    COUNT(CASE WHEN rs.risk_category = 'High' THEN 1 END) as high_risk_count
FROM staging.customers c
JOIN staging.credit_history ch ON c.customer_id = ch.customer_id
JOIN predictions.risk_scores rs ON c.customer_id = rs.customer_id
WHERE rs.prediction_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY income_segment, credit_segment
ORDER BY income_segment, credit_segment;

-- 10. Default Prediction Accuracy (if actual outcomes available)
SELECT 
    rs.risk_category,
    COUNT(*) as predictions,
    SUM(CASE WHEN la.default_flag = true THEN 1 ELSE 0 END) as actual_defaults,
    ROUND(AVG(CASE WHEN la.default_flag = true THEN 1 ELSE 0 END)::NUMERIC * 100, 2) as actual_default_rate,
    ROUND(AVG(rs.default_probability)::NUMERIC * 100, 2) as predicted_default_rate
FROM predictions.risk_scores rs
JOIN staging.loan_applications la ON rs.customer_id = la.customer_id
WHERE rs.prediction_date >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY rs.risk_category
ORDER BY rs.risk_category;

-- 11. High Value At-Risk Customers
SELECT 
    c.customer_id,
    c.annual_income,
    ch.credit_score,
    la.loan_amount,
    rs.risk_score,
    rs.risk_category,
    la.loan_amount * rs.default_probability as potential_loss
FROM staging.customers c
JOIN staging.credit_history ch ON c.customer_id = ch.customer_id
JOIN staging.loan_applications la ON c.customer_id = la.customer_id
JOIN predictions.risk_scores rs ON c.customer_id = rs.customer_id
WHERE rs.risk_category IN ('Medium', 'High')
    AND la.loan_amount > 50000
    AND rs.prediction_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY potential_loss DESC
LIMIT 100;

-- 12. Feature Distribution Summary
SELECT 
    'Credit Score' as feature,
    ROUND(AVG(credit_score), 2) as mean_value,
    ROUND(STDDEV(credit_score), 2) as std_dev,
    MIN(credit_score) as min_value,
    MAX(credit_score) as max_value
FROM staging.credit_history

UNION ALL

SELECT 
    'Annual Income' as feature,
    ROUND(AVG(annual_income), 2) as mean_value,
    ROUND(STDDEV(annual_income), 2) as std_dev,
    MIN(annual_income) as min_value,
    MAX(annual_income) as max_value
FROM staging.customers

UNION ALL

SELECT 
    'Debt to Income' as feature,
    ROUND(AVG(debt_to_income_ratio)::NUMERIC, 2) as mean_value,
    ROUND(STDDEV(debt_to_income_ratio)::NUMERIC, 2) as std_dev,
    MIN(debt_to_income_ratio) as min_value,
    MAX(debt_to_income_ratio) as max_value
FROM staging.loan_applications;

-- 13. Weekly Risk Alert Report
SELECT 
    CURRENT_DATE as report_date,
    'New High Risk Customers' as alert_type,
    COUNT(*) as alert_count,
    STRING_AGG(customer_id::TEXT, ', ') as customer_ids
FROM predictions.risk_scores
WHERE risk_category = 'High'
    AND prediction_date >= CURRENT_DATE - INTERVAL '7 days'
HAVING COUNT(*) > 10

UNION ALL

SELECT 
    CURRENT_DATE as report_date,
    'Sudden Risk Increase' as alert_type,
    COUNT(*) as alert_count,
    STRING_AGG(customer_id::TEXT, ', ') as customer_ids
FROM (
    SELECT 
        customer_id,
        risk_score - LAG(risk_score) OVER (PARTITION BY customer_id ORDER BY prediction_date) as risk_change
    FROM predictions.risk_scores
    WHERE prediction_date >= CURRENT_DATE - INTERVAL '30 days'
) risk_changes
WHERE risk_change > 20
HAVING COUNT(*) > 0;

-- 14. Profitability vs Risk Analysis
SELECT 
    rs.risk_category,
    COUNT(*) as loan_count,
    ROUND(AVG(la.loan_amount), 2) as avg_loan_amount,
    ROUND(AVG(la.interest_rate)::NUMERIC * 100, 2) as avg_interest_rate,
    ROUND(SUM(la.loan_amount * la.interest_rate * (la.loan_term_months / 12.0)), 2) as total_interest_revenue,
    ROUND(SUM(CASE WHEN la.default_flag = true THEN la.loan_amount ELSE 0 END), 2) as total_loss_from_defaults,
    ROUND(SUM(la.loan_amount * la.interest_rate * (la.loan_term_months / 12.0)) - 
          SUM(CASE WHEN la.default_flag = true THEN la.loan_amount ELSE 0 END), 2) as net_profit
FROM predictions.risk_scores rs
JOIN staging.loan_applications la ON rs.customer_id = la.customer_id
WHERE rs.prediction_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY rs.risk_category
ORDER BY rs.risk_category;

-- 15. Data Quality Metrics
SELECT 
    'Customers' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as null_age,
    SUM(CASE WHEN annual_income IS NULL THEN 1 ELSE 0 END) as null_income,
    MAX(created_at) as last_updated
FROM staging.customers

UNION ALL

SELECT 
    'Credit History' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN credit_score IS NULL THEN 1 ELSE 0 END) as null_credit_score,
    SUM(CASE WHEN credit_utilization_ratio IS NULL THEN 1 ELSE 0 END) as null_utilization,
    MAX(created_at) as last_updated
FROM staging.credit_history

UNION ALL

SELECT 
    'Loan Applications' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN loan_amount IS NULL THEN 1 ELSE 0 END) as null_loan_amount,
    SUM(CASE WHEN interest_rate IS NULL THEN 1 ELSE 0 END) as null_interest_rate,
    MAX(created_at) as last_updated
FROM staging.loan_applications;