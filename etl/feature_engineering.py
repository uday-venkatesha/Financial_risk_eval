import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_income_category(income_series):
    """
    Categorize income into brackets
    
    Args:
        income_series: pandas Series of annual income values
    
    Returns:
        pandas Series of income categories
    """
    return pd.cut(
        income_series,
        bins=[0, 30000, 60000, 100000, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

def calculate_employment_stability(age, years_employed):
    """
    Calculate employment stability score
    
    Args:
        age: Customer age
        years_employed: Years in current employment
    
    Returns:
        Employment stability score (0-100)
    """
    # Avoid division by zero
    age = np.maximum(age, 1)
    return (years_employed / age) * 100

def calculate_delinquency_score(num_delinquent, credit_util):
    """
    Calculate delinquency risk score
    
    Args:
        num_delinquent: Number of delinquent accounts
        credit_util: Credit utilization ratio
    
    Returns:
        Delinquency score (higher is better)
    """
    # Penalize for delinquencies, reward for low utilization
    delinquency_penalty = num_delinquent * 20
    utilization_bonus = (1 - credit_util) * 30
    
    return utilization_bonus - delinquency_penalty

def calculate_loan_to_income_ratio(loan_amount, annual_income):
    """
    Calculate loan to income ratio
    
    Args:
        loan_amount: Requested loan amount
        annual_income: Annual income
    
    Returns:
        Loan to income ratio
    """
    # Avoid division by zero
    annual_income = np.maximum(annual_income, 1)
    return loan_amount / annual_income

def calculate_monthly_payment(principal, annual_rate, months):
    """
    Calculate monthly loan payment using amortization formula
    
    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Loan term in months
    
    Returns:
        Monthly payment amount
    """
    monthly_rate = annual_rate / 12
    
    if monthly_rate == 0:
        return principal / months
    
    # Amortization formula
    payment = principal * (monthly_rate * (1 + monthly_rate)**months) / \
              ((1 + monthly_rate)**months - 1)
    
    return payment

def calculate_debt_to_income_ratio(loan_amount, interest_rate, loan_term_months, annual_income):
    """
    Calculate debt-to-income ratio
    
    Args:
        loan_amount: Loan amount
        interest_rate: Annual interest rate
        loan_term_months: Loan term in months
        annual_income: Annual income
    
    Returns:
        Debt-to-income ratio
    """
    monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term_months)
    annual_payment = monthly_payment * 12
    
    # Avoid division by zero
    annual_income = np.maximum(annual_income, 1)
    
    return annual_payment / annual_income

def calculate_credit_age_score(years_credit_history, age):
    """
    Calculate credit history age score
    
    Args:
        years_credit_history: Length of credit history
        age: Customer age
    
    Returns:
        Credit age score (0-100)
    """
    max_possible_history = age - 18
    max_possible_history = np.maximum(max_possible_history, 1)
    
    return (years_credit_history / max_possible_history) * 100

def calculate_debt_burden(total_debt, annual_income):
    """
    Calculate overall debt burden ratio
    
    Args:
        total_debt: Total outstanding debt
        annual_income: Annual income
    
    Returns:
        Debt burden ratio
    """
    annual_income = np.maximum(annual_income, 1)
    return total_debt / annual_income

def calculate_credit_utilization_category(utilization_ratio):
    """
    Categorize credit utilization
    
    Args:
        utilization_ratio: Credit utilization ratio (0-1)
    
    Returns:
        Utilization category
    """
    conditions = [
        (utilization_ratio < 0.3),
        (utilization_ratio < 0.5),
        (utilization_ratio < 0.7),
        (utilization_ratio >= 0.7)
    ]
    
    categories = ['Excellent', 'Good', 'Fair', 'Poor']
    
    return np.select(conditions, categories, default='Unknown')

def calculate_payment_consistency_score(transaction_df, customer_id):
    """
    Calculate payment consistency based on transaction patterns
    
    Args:
        transaction_df: DataFrame with transaction history
        customer_id: Customer ID
    
    Returns:
        Payment consistency score (0-100)
    """
    customer_trans = transaction_df[transaction_df['customer_id'] == customer_id]
    
    if len(customer_trans) == 0:
        return 50  # Neutral score for no history
    
    # Calculate coefficient of variation (lower is more consistent)
    mean_amount = customer_trans['transaction_amount'].mean()
    std_amount = customer_trans['transaction_amount'].std()
    
    if mean_amount == 0:
        return 50
    
    cv = std_amount / mean_amount
    
    # Convert to 0-100 scale (lower CV = higher score)
    score = max(0, min(100, 100 - (cv * 50)))
    
    return score

def engineer_interaction_features(df):
    """
    Create interaction features between variables
    
    Args:
        df: DataFrame with base features
    
    Returns:
        DataFrame with additional interaction features
    """
    # Income x Credit Score interaction
    df['income_credit_interaction'] = (
        df['annual_income'] / 100000 * df['credit_score'] / 850
    )
    
    # DTI x Credit Utilization interaction
    df['dti_utilization_risk'] = (
        df['debt_to_income_ratio'] * df['credit_utilization_ratio']
    )
    
    # Age x Employment stability
    df['mature_stable_employment'] = (
        df['age'] / 100 * df['employment_stability_score'] / 100
    )
    
    # Loan size x Income interaction
    df['loan_affordability'] = (
        df['loan_to_income_ratio'] * (1 - df['debt_to_income_ratio'])
    )
    
    return df

def create_risk_flags(df):
    """
    Create binary risk flag features
    
    Args:
        df: DataFrame with features
    
    Returns:
        DataFrame with risk flags
    """
    # High DTI flag
    df['high_dti_flag'] = (df['debt_to_income_ratio'] > 0.43).astype(int)
    
    # Low credit score flag
    df['low_credit_flag'] = (df['credit_score'] < 650).astype(int)
    
    # High utilization flag
    df['high_utilization_flag'] = (df['credit_utilization_ratio'] > 0.7).astype(int)
    
    # Recent delinquency flag
    df['has_delinquency_flag'] = (df['num_delinquent_accounts'] > 0).astype(int)
    
    # Unemployed flag
    df['unemployed_flag'] = (df['employment_status'] == 'Unemployed').astype(int)
    
    # Combined high risk flag
    df['multiple_risk_factors'] = (
        df['high_dti_flag'] + 
        df['low_credit_flag'] + 
        df['high_utilization_flag'] + 
        df['has_delinquency_flag']
    )
    
    return df

def create_binned_features(df):
    """
    Create binned versions of continuous features
    
    Args:
        df: DataFrame with features
    
    Returns:
        DataFrame with binned features
    """
    # Age bins
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '60+']
    )
    
    # Credit score bins
    df['credit_tier'] = pd.cut(
        df['credit_score'],
        bins=[0, 600, 650, 700, 750, 850],
        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    )
    
    # Loan amount bins
    df['loan_size'] = pd.cut(
        df['loan_amount'],
        bins=[0, 25000, 50000, 100000, np.inf],
        labels=['Small', 'Medium', 'Large', 'Very Large']
    )
    
    return df

def calculate_aggregate_features(transactions_df):
    """
    Calculate aggregate features from transaction history
    
    Args:
        transactions_df: DataFrame with transaction history
    
    Returns:
        DataFrame with aggregated features per customer
    """
    agg_features = transactions_df.groupby('customer_id').agg({
        'transaction_amount': ['mean', 'std', 'min', 'max', 'sum'],
        'transaction_id': 'count'
    }).reset_index()
    
    # Flatten column names
    agg_features.columns = [
        'customer_id',
        'avg_transaction_amount',
        'transaction_volatility',
        'min_transaction_amount',
        'max_transaction_amount',
        'total_transaction_amount',
        'transaction_count'
    ]
    
    # Fill NaN volatility with 0 for customers with a single transaction
    agg_features['transaction_volatility'] = agg_features['transaction_volatility'].fillna(0)
    
    return agg_features