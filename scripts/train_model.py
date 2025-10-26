#!/usr/bin/env python3
"""
Standalone model training script
"""
import sys
sys.path.append('..')

from models.training import RiskPredictionModel
from sqlalchemy import create_engine
import pandas as pd
from config.database import get_db_connection_string

def main():
    # Load features
    engine = create_engine(get_db_connection_string())
    features = pd.read_sql("SELECT * FROM analytics.risk_features", engine)
    
    # Train model
    model = RiskPredictionModel()
    X_train, X_test, y_train, y_test = model.prepare_data(features)
    model.train_models(X_train, y_train)
    results = model.evaluate_models(X_test, y_test)
    
    # Save best model
    model.save_model('random_forest', 'models/saved_models/risk_model.pkl')
    print("Model training completed!")

if __name__ == '__main__':
    main()