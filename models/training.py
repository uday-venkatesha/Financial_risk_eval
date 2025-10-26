# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import json

class RiskPredictionModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, feature_df):
        """Prepare data for modeling"""
        # Select features
        feature_cols = [
            'age', 'annual_income', 'years_employed', 'num_dependents',
            'credit_score', 'num_credit_accounts', 'credit_utilization_ratio',
            'num_delinquent_accounts', 'total_debt', 'years_credit_history',
            'loan_amount', 'loan_term_months', 'interest_rate',
            'debt_to_income_ratio', 'employment_stability_score',
            'delinquency_score', 'loan_to_income_ratio'
        ]
        
        # Handle categorical variables
        categorical_cols = ['employment_status', 'home_ownership', 'loan_purpose']
        for col in categorical_cols:
            if col in feature_df.columns:
                le = LabelEncoder()
                feature_df[f'{col}_encoded'] = le.fit_transform(feature_df[col].fillna('Unknown'))
                feature_cols.append(f'{col}_encoded')
        
        X = feature_df[feature_cols].fillna(0)
        y = feature_df['default_flag'].astype(int)
        
        self.feature_names = feature_cols
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        print("Training models...")
        
        # Logistic Regression
        self.models['logistic'] = LogisticRegression(max_iter=1000, random_state=42)
        self.models['logistic'].fit(X_train, y_train)
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.models['gradient_boosting'].fit(X_train, y_train)
        
        print("Models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"\n{name.upper()} Results:")
            for metric, value in results[name].items():
                print(f"  {metric}: {value:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from tree-based models"""
        if model_name in ['random_forest', 'gradient_boosting']:
            importance = self.models[model_name].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None
    
    def save_model(self, model_name, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)