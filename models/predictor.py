import pickle
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the pickled model file
    
    Returns:
        Loaded model object
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_risk(model, input_features):
    """
    Predict risk score for given features
    
    Args:
        model: Trained model
        input_features: DataFrame with customer features
    
    Returns:
        dict: Risk prediction results
    """
    try:
        # Select and order features as expected by model
        feature_cols = [
            'age', 'annual_income', 'years_employed', 'num_dependents',
            'credit_score', 'num_credit_accounts', 'credit_utilization_ratio',
            'num_delinquent_accounts', 'total_debt', 'years_credit_history',
            'loan_amount', 'loan_term_months', 'interest_rate',
            'debt_to_income_ratio', 'employment_stability_score',
            'delinquency_score', 'loan_to_income_ratio',
            'employment_status_encoded', 'home_ownership_encoded'
        ]
        
        # Ensure all required features are present
        X = input_features[feature_cols].fillna(0)
        
        # Get prediction probability
        prediction_proba = model.predict_proba(X)[0]
        default_probability = prediction_proba[1]  # Probability of class 1 (default)
        
        # Calculate risk score (0-100)
        risk_score = default_probability * 100
        
        # Categorize risk and provide recommendation
        if risk_score < 30:
            risk_category = 'Low'
            recommendation = 'Approve - Low risk of default'
        elif risk_score < 60:
            risk_category = 'Medium'
            recommendation = 'Review - Manual underwriting recommended'
        else:
            risk_category = 'High'
            recommendation = 'Reject - High risk of default'
        
        result = {
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'default_probability': float(default_probability),
            'recommendation': recommendation
        }
        
        logger.info(f"Prediction: {risk_category} (score: {risk_score:.2f})")
        return result
        
    except KeyError as e:
        logger.error(f"Missing required feature: {str(e)}")
        raise ValueError(f"Missing required feature: {str(e)}")
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def predict_batch(model, features_df):
    """
    Make predictions for multiple customers
    
    Args:
        model: Trained model
        features_df: DataFrame with features for multiple customers
    
    Returns:
        DataFrame: Predictions for all customers
    """
    try:
        predictions = []
        
        for idx, row in features_df.iterrows():
            row_df = pd.DataFrame([row])
            prediction = predict_risk(model, row_df)
            prediction['customer_id'] = row.get('customer_id')
            predictions.append(prediction)
        
        return pd.DataFrame(predictions)
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise

def explain_prediction(model, input_features, feature_names=None):
    """
    Provide explanation for model prediction using feature importance
    
    Args:
        model: Trained model
        input_features: DataFrame with customer features
        feature_names: List of feature names
    
    Returns:
        dict: Feature contributions to prediction
    """
    try:
        # Get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = input_features.columns.tolist()
            
            # Get feature values for this prediction
            feature_values = input_features.iloc[0].to_dict()
            
            # Combine importance with values
            explanations = []
            for name, importance in zip(feature_names, importances):
                if name in feature_values:
                    explanations.append({
                        'feature': name,
                        'value': feature_values[name],
                        'importance': float(importance)
                    })
            
            # Sort by importance
            explanations.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'top_features': explanations[:5],
                'all_features': explanations
            }
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
            
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        return None