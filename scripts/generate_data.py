#!/usr/bin/env python3
"""
Main script to generate synthetic data and load to database
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.generator import FinancialDataGenerator
from data.loader import load_to_database
from sqlalchemy import create_engine
from config import get_db_connection_string

def main():
    print("Starting data generation...")
    
    # Initialize generator
    generator = FinancialDataGenerator(num_customers=10000)
    
    # Generate data
    print("Generating customers...")
    customers = generator.generate_customers()
    
    print("Generating credit history...")
    credit = generator.generate_credit_history(customers)
    
    print("Generating transactions...")
    transactions = generator.generate_transactions(customers)
    
    print("Generating loans...")
    loans = generator.generate_loans(customers, credit)
    
    # Load to database
    print("Connecting to database...")
    engine = create_engine(get_db_connection_string())
    
    data_dict = {
        'customers': customers,
        'credit_history': credit,
        'transactions': transactions,
        'loan_applications': loans
    }
    
    print("Loading data to database...")
    load_to_database(data_dict, engine)
    
    print("\n" + "="*50)
    print("Data generation completed successfully!")
    print("="*50)
    print(f"Customers: {len(customers)}")
    print(f"Credit records: {len(credit)}")
    print(f"Transactions: {len(transactions)}")
    print(f"Loan applications: {len(loans)}")
    print("="*50)

if __name__ == '__main__':
    main()