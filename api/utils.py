import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def validate_request(data, required_fields):
    """
    Validate API request data
    
    Args:
        data: Request JSON data
        required_fields: List of required field names
    
    Returns:
        dict: {'valid': bool, 'message': str}
    """
    if not data:
        return {'valid': False, 'message': 'Request body is required'}
    
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return {
            'valid': False, 
            'message': f'Missing required fields: {", ".join(missing_fields)}'
        }
    
    # Validate data types
    if 'customer_id' in data and not isinstance(data['customer_id'], int):
        return {'valid': False, 'message': 'customer_id must be an integer'}
    
    if 'loan_amount' in data:
        try:
            loan_amount = float(data['loan_amount'])
            if loan_amount <= 0:
                return {'valid': False, 'message': 'loan_amount must be positive'}
        except (ValueError, TypeError):
            return {'valid': False, 'message': 'loan_amount must be a number'}
    
    if 'loan_term_months' in data:
        try:
            loan_term = int(data['loan_term_months'])
            if loan_term not in [12, 24, 36, 48, 60, 72]:
                return {'valid': False, 'message': 'loan_term_months must be one of: 12, 24, 36, 48, 60, 72'}
        except (ValueError, TypeError):
            return {'valid': False, 'message': 'loan_term_months must be an integer'}
    
    return {'valid': True, 'message': 'Valid request'}

def load_customer_features(engine, customer_id, loan_amount, loan_term_months=36):
    """
    Load and prepare customer features for prediction
    
    Args:
        engine: SQLAlchemy engine
        customer_id: Customer ID
        loan_amount: Requested loan amount
        loan_term_months: Loan term in months
    
    Returns:
        DataFrame: Feature vector for prediction or None if customer not found
    """
    try:
        query = """
            SELECT 
                c.customer_id,
                c.age,
                c.annual_income,
                c.years_employed,
                c.num_dependents,
                c.employment_status,
                c.home_ownership,
                ch.credit_score,
                ch.num_credit_accounts,
                ch.credit_utilization_ratio,
                ch.num_delinquent_accounts,
                ch.total_debt,
                ch.years_credit_history,
                ch.bankruptcy_flag
            FROM staging.customers c
            LEFT JOIN staging.credit_history ch ON c.customer_id = ch.customer_id
            WHERE c.customer_id = :customer_id
        """
        
        customer_data = pd.read_sql(query, engine, params={'customer_id': customer_id})
        
        if customer_data.empty:
            logger.warning(f"Customer {customer_id} not found")
            return None
        
        # Add loan-specific features
        customer_data['loan_amount'] = loan_amount
        customer_data['loan_term_months'] = loan_term_months
        
        # Calculate interest rate based on credit score
        customer_data['interest_rate'] = customer_data['credit_score'].apply(
            calculate_interest_rate
        )
        
        # Calculate derived features
        customer_data['debt_to_income_ratio'] = calculate_dti(
            loan_amount,
            loan_term_months,
            customer_data['interest_rate'].iloc[0],
            customer_data['annual_income'].iloc[0]
        )
        
        customer_data['employment_stability_score'] = (
            customer_data['years_employed'] / customer_data['age']
        ) * 100
        
        customer_data['delinquency_score'] = (
            customer_data['num_delinquent_accounts'] * 20 + 
            (1 - customer_data['credit_utilization_ratio']) * 30
        )
        
        customer_data['loan_to_income_ratio'] = (
            customer_data['loan_amount'] / customer_data['annual_income']
        )
        
        # Get transaction features
        trans_query = """
            SELECT 
                AVG(transaction_amount) as avg_transaction,
                STDDEV(transaction_amount) as transaction_volatility,
                COUNT(*) as transaction_count
            FROM staging.transactions
            WHERE customer_id = :customer_id
        """
        
        trans_data = pd.read_sql(trans_query, engine, params={'customer_id': customer_id})
        
        if not trans_data.empty:
            customer_data['avg_monthly_spending'] = trans_data['avg_transaction'].iloc[0]
            customer_data['spending_volatility'] = trans_data['transaction_volatility'].iloc[0]
            customer_data['transaction_count'] = trans_data['transaction_count'].iloc[0]
        else:
            customer_data['avg_monthly_spending'] = 0
            customer_data['spending_volatility'] = 0
            customer_data['transaction_count'] = 0
        
        # Encode categorical variables
        employment_mapping = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}
        home_mapping = {'Own': 0, 'Rent': 1, 'Mortgage': 2}
        
        customer_data['employment_status_encoded'] = customer_data['employment_status'].map(
            employment_mapping
        ).fillna(0)
        
        customer_data['home_ownership_encoded'] = customer_data['home_ownership'].map(
            home_mapping
        ).fillna(0)
        
        return customer_data
        
    except Exception as e:
        logger.error(f"Error loading customer features: {str(e)}")
        return None

def calculate_interest_rate(credit_score):
    """Calculate interest rate based on credit score"""
    if credit_score >= 750:
        return np.random.uniform(0.03, 0.06)
    elif credit_score >= 700:
        return np.random.uniform(0.06, 0.10)
    elif credit_score >= 650:
        return np.random.uniform(0.10, 0.15)
    else:
        return np.random.uniform(0.15, 0.25)

def calculate_monthly_payment(principal, annual_rate, months):
    """Calculate monthly loan payment"""
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        return principal / months
    return principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)

def calculate_dti(loan_amount, loan_term_months, interest_rate, annual_income):
    """Calculate debt-to-income ratio"""
    monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term_months)
    annual_payment = monthly_payment * 12
    return annual_payment / annual_income if annual_income > 0 else 0

def calculate_risk_score(features, model):
    """
    Calculate risk score from model predictions
    
    Args:
        features: Feature DataFrame
        model: Trained model
    
    Returns:
        dict: Risk assessment results
    """
    try:
        # Get prediction probability
        prediction_proba = model.predict_proba(features)[0]
        default_probability = prediction_proba[1]  # Probability of class 1 (default)
        
        # Calculate risk score (0-100)
        risk_score = default_probability * 100
        
        # Categorize risk
        if risk_score < 30:
            risk_category = 'Low'
            recommendation = 'Approve'
        elif risk_score < 60:
            risk_category = 'Medium'
            recommendation = 'Review'
        else:
            risk_category = 'High'
            recommendation = 'Reject'
        
        return {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'default_probability': default_probability,
            'recommendation': recommendation
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk score: {str(e)}")
        raise

def format_response(data, status='success'):
    """Format API response consistently"""
    return {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }

def handle_error(error_message, status_code=500):
    """Format error response consistently"""
    return {
        'status': 'error',
        'timestamp': datetime.now().isoformat(),
        'error': error_message
    }, status_code