# etl_pipeline.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime

class RiskETLPipeline:
    def __init__(self, db_connection_string):
        self.engine = create_engine(db_connection_string)
    
    def extract_data(self):
        """Extract data from staging schema"""
        customers = pd.read_sql("SELECT * FROM staging.customers", self.engine)
        credit = pd.read_sql("SELECT * FROM staging.credit_history", self.engine)
        transactions = pd.read_sql("SELECT * FROM staging.transactions", self.engine)
        loans = pd.read_sql("SELECT * FROM staging.loan_applications", self.engine)
        
        return customers, credit, transactions, loans
    
    def transform_features(self, customers, credit, transactions, loans):
        """Feature engineering for risk modeling"""
        
        # Aggregate transaction features
        trans_features = transactions.groupby('customer_id').agg({
            'transaction_amount': ['mean', 'std', 'sum'],
            'transaction_id': 'count'
        }).reset_index()
        trans_features.columns = ['customer_id', 'avg_monthly_spending', 
                                  'spending_volatility', 'total_spending', 'transaction_count']
        
        # Merge all data
        feature_df = loans.merge(customers, on='customer_id', how='left')
        feature_df = feature_df.merge(credit, on='customer_id', how='left')
        feature_df = feature_df.merge(trans_features, on='customer_id', how='left')
        
        # Create derived features
        feature_df['income_category'] = pd.cut(feature_df['annual_income'], 
                                               bins=[0, 30000, 60000, 100000, np.inf],
                                               labels=['Low', 'Medium', 'High', 'Very High'])
        
        feature_df['employment_stability_score'] = (
            feature_df['years_employed'] / feature_df['age']
        ) * 100
        
        feature_df['delinquency_score'] = (
            feature_df['num_delinquent_accounts'] * 20 + 
            (1 - feature_df['credit_utilization_ratio']) * 30
        )
        
        feature_df['loan_to_income_ratio'] = (
            feature_df['loan_amount'] / feature_df['annual_income']
        )
        
        return feature_df
    
    def load_features(self, feature_df):
        """Load transformed features to analytics schema"""
        feature_df.to_sql('risk_features', self.engine, 
                         schema='analytics', if_exists='replace', index=False)
    
    def run_pipeline(self):
        """Execute full ETL pipeline"""
        print(f"[{datetime.now()}] Starting ETL Pipeline...")
        
        # Extract
        print("Extracting data...")
        customers, credit, transactions, loans = self.extract_data()
        
        # Transform
        print("Transforming and engineering features...")
        features = self.transform_features(customers, credit, transactions, loans)
        
        # Load
        print("Loading features to analytics schema...")
        self.load_features(features)
        
        print(f"[{datetime.now()}] ETL Pipeline completed successfully!")
        return features