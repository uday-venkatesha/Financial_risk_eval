import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)

class FinancialDataGenerator:
    def __init__(self, num_customers=10000):
        self.num_customers = num_customers
        
    def generate_customers(self):
        """Generate synthetic customer data"""
        customers = []
        for i in range(self.num_customers):
            customer = {
                'customer_id': i + 1,
                'age': np.random.randint(21, 70),
                'gender': random.choice(['Male', 'Female', 'Other']),
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
                'education_level': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
                'employment_status': random.choice(['Employed', 'Self-Employed', 'Unemployed']),
                'annual_income': np.random.lognormal(10.5, 0.8),  # Log-normal distribution
                'years_employed': np.random.randint(0, 30),
                'home_ownership': random.choice(['Own', 'Rent', 'Mortgage']),
                'num_dependents': np.random.poisson(1.5)
            }
            customers.append(customer)
        return pd.DataFrame(customers)
    
    def generate_credit_history(self, customers_df):
        """Generate credit history for customers"""
        credit_data = []
        for customer_id in customers_df['customer_id']:
            # Correlate credit score with income and employment
            customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            base_score = 600 + (customer['annual_income'] / 1000) + (customer['years_employed'] * 2)
            credit_score = int(np.clip(base_score + np.random.normal(0, 50), 300, 850))
            
            credit = {
                'customer_id': customer_id,
                'credit_score': credit_score,
                'num_credit_accounts': np.random.randint(1, 15),
                'credit_utilization_ratio': np.random.beta(2, 5),  # Skewed towards lower utilization
                'num_delinquent_accounts': np.random.poisson(0.3),
                'total_debt': np.random.lognormal(9, 1.5),
                'bankruptcy_flag': random.choices([True, False], weights=[0.05, 0.95])[0],
                'years_credit_history': min(customer['age'] - 18, np.random.randint(1, 25))
            }
            credit_data.append(credit)
        return pd.DataFrame(credit_data)
    
    def generate_transactions(self, customers_df, num_transactions_per_customer=50):
        """Generate transaction history"""
        transactions = []
        transaction_id = 1
        
        for customer_id in customers_df['customer_id']:
            customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            num_trans = np.random.poisson(num_transactions_per_customer)
            
            for _ in range(num_trans):
                transaction = {
                    'transaction_id': transaction_id,
                    'customer_id': customer_id,
                    'transaction_date': fake.date_between(start_date='-2y', end_date='today'),
                    'transaction_amount': abs(np.random.normal(customer['annual_income']/12/20, 100)),
                    'transaction_type': random.choice(['Debit', 'Credit']),
                    'category': random.choice(['Groceries', 'Utilities', 'Entertainment', 
                                              'Healthcare', 'Transportation', 'Other'])
                }
                transactions.append(transaction)
                transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def generate_loans(self, customers_df, credit_df, default_rate=0.15):
        """Generate loan applications with realistic default patterns"""
        loans = []
        
        for idx, customer in customers_df.iterrows():
            credit = credit_df[credit_df['customer_id'] == customer['customer_id']].iloc[0]
            
            # Some customers don't have loans
            if random.random() > 0.7:
                continue
            
            loan_amount = np.random.uniform(5000, min(customer['annual_income'] * 3, 500000))
            interest_rate = self._calculate_interest_rate(credit['credit_score'])
            loan_term = random.choice([12, 24, 36, 48, 60, 72])
            monthly_payment = self._calculate_monthly_payment(loan_amount, interest_rate, loan_term)
            
            debt_to_income = (monthly_payment * 12) / customer['annual_income']
            
            # Calculate default probability based on risk factors
            default_prob = self._calculate_default_probability(
                credit['credit_score'],
                debt_to_income,
                customer['employment_status'],
                credit['num_delinquent_accounts']
            )
            
            default_flag = random.random() < default_prob
            
            loan = {
                'customer_id': customer['customer_id'],
                'application_date': fake.date_between(start_date='-1y', end_date='today'),
                'loan_amount': loan_amount,
                'loan_purpose': random.choice(['Home', 'Auto', 'Personal', 'Education', 'Business']),
                'loan_term_months': loan_term,
                'interest_rate': interest_rate,
                'monthly_payment': monthly_payment,
                'debt_to_income_ratio': debt_to_income,
                'default_flag': default_flag,
                'days_past_due': np.random.poisson(30) if default_flag else 0
            }
            loans.append(loan)
        
        return pd.DataFrame(loans)
    
    def _calculate_interest_rate(self, credit_score):
        """Calculate interest rate based on credit score"""
        if credit_score >= 750:
            return np.random.uniform(0.03, 0.06)
        elif credit_score >= 700:
            return np.random.uniform(0.06, 0.10)
        elif credit_score >= 650:
            return np.random.uniform(0.10, 0.15)
        else:
            return np.random.uniform(0.15, 0.25)
    
    def _calculate_monthly_payment(self, principal, annual_rate, months):
        """Calculate monthly loan payment"""
        monthly_rate = annual_rate / 12
        if monthly_rate == 0:
            return principal / months
        return principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    
    def _calculate_default_probability(self, credit_score, dti, employment_status, delinquencies):
        """Calculate probability of default based on risk factors"""
        prob = 0.05  # Base probability
        
        # Credit score impact
        if credit_score < 600:
            prob += 0.3
        elif credit_score < 700:
            prob += 0.15
        
        # DTI impact
        if dti > 0.43:
            prob += 0.2
        elif dti > 0.35:
            prob += 0.1
        
        # Employment impact
        if employment_status == 'Unemployed':
            prob += 0.25
        
        # Delinquency impact
        prob += delinquencies * 0.1
        
        return min(prob, 0.9) 