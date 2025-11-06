#!/usr/bin/env python3
"""
Standalone model training script
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.training import RiskPredictionModel
from sqlalchemy import create_engine
import pandas as pd
from config import get_db_connection_string

def main():
    print("Starting model training...")
    
    # Load features
    print("Loading features from database...")
    engine = create_engine(get_db_connection_string())
    features = pd.read_sql("SELECT * FROM analytics.risk_features", engine)
    
    print(f"Loaded {len(features)} feature records")
    
    # Train model
    model = RiskPredictionModel()
    X_train, X_test, y_train, y_test = model.prepare_data(features)
    model.train_models(X_train, y_train)
    results = model.evaluate_models(X_test, y_test)
    
    # Save best model
    print("\nSaving model...")
    model.save_model('random_forest', 'models/saved_models/risk_model.pkl')
    
    print("\n" + "="*50)
    print("Model training completed successfully!")
    print("="*50)
    print(f"Best model saved: models/saved_models/risk_model.pkl")
    print("="*50)

if __name__ == '__main__':
    main()