#!/usr/bin/env python3
"""
Main script to generate synthetic data and load to database
"""
import sys
sys.path.append('..')

from data.generator import FinancialDataGenerator
from data.loader import load_to_database
from sqlalchemy import create_engine
from config.database import get_db_connection_string

def main():
    print("Starting data generation...")
    
    # Initialize generator
    generator = FinancialDataGenerator(num_customers=10000)
    
    # Generate data
    customers = generator.generate_customers()
    credit = generator.generate_credit_history(customers)
    transactions = generator.generate_transactions(customers)
    loans = generator.generate_loans(customers, credit)
    
    # Load to database
    engine = create_engine(get_db_connection_string())
    
    data_dict = {
        'customers': customers,
        'credit_history': credit,
        'transactions': transactions,
        'loan_applications': loans
    }
    
    load_to_database(data_dict, engine)
    print("Data generation and loading completed!")

if __name__ == '__main__':
    main()