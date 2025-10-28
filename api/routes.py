from flask import Blueprint, request, jsonify
from api.utils import validate_request, load_customer_features, calculate_risk_score
from models.predictor import load_model, predict_risk
from sqlalchemy import create_engine
from config import get_db_connection_string
import pandas as pd
import logging
import json
from datetime import datetime

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# Load model at startup
MODEL_PATH = 'models/saved_models/risk_model.pkl'
try:
    risk_model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    risk_model = None

# Database connection
engine = create_engine(get_db_connection_string())

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict risk score for a customer
    
    Request body:
    {
        "customer_id": int,
        "loan_amount": float,
        "loan_term_months": int,
        "loan_purpose": str (optional)
    }
    """
    try:
        # Validate request
        data = request.get_json()
        validation_result = validate_request(data, required_fields=['customer_id', 'loan_amount'])
        
        if not validation_result['valid']:
            return jsonify({'error': validation_result['message']}), 400
        
        customer_id = data['customer_id']
        loan_amount = data['loan_amount']
        loan_term_months = data.get('loan_term_months', 36)
        
        # Load customer features
        features = load_customer_features(engine, customer_id, loan_amount, loan_term_months)
        
        if features is None:
            return jsonify({'error': f'Customer {customer_id} not found'}), 404
        
        # Make prediction
        if risk_model is None:
            return jsonify({'error': 'Model not available'}), 503
        
        risk_result = predict_risk(risk_model, features)
        
        # Save prediction to database
        save_prediction(engine, customer_id, risk_result)
        
        response = {
            'customer_id': customer_id,
            'loan_amount': loan_amount,
            'risk_score': float(risk_result['risk_score']),
            'risk_category': risk_result['risk_category'],
            'default_probability': float(risk_result['default_probability']),
            'recommendation': risk_result['recommendation'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction generated for customer {customer_id}: {risk_result['risk_category']}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@api_bp.route('/risk-score/<int:customer_id>', methods=['GET'])
def get_risk_score(customer_id):
    """Get existing risk score for a customer"""
    try:
        query = """
            SELECT 
                customer_id,
                risk_score,
                risk_category,
                default_probability,
                prediction_date
            FROM predictions.risk_scores
            WHERE customer_id = :customer_id
            ORDER BY prediction_date DESC
            LIMIT 1
        """
        
        result = pd.read_sql(query, engine, params={'customer_id': customer_id})
        
        if result.empty:
            return jsonify({'error': f'No risk score found for customer {customer_id}'}), 404
        
        record = result.iloc[0].to_dict()
        record['prediction_date'] = record['prediction_date'].isoformat()
        
        return jsonify(record), 200
        
    except Exception as e:
        logger.error(f"Error retrieving risk score: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/model-metrics', methods=['GET'])
def get_model_metrics():
    """Get current model performance metrics"""
    try:
        query = """
            SELECT 
                model_name,
                model_version,
                algorithm,
                accuracy,
                precision_score,
                recall,
                f1_score,
                auc_roc,
                training_date
            FROM predictions.model_metadata
            ORDER BY training_date DESC
            LIMIT 1
        """
        
        result = pd.read_sql(query, engine)
        
        if result.empty:
            return jsonify({'error': 'No model metrics available'}), 404
        
        metrics = result.iloc[0].to_dict()
        metrics['training_date'] = metrics['training_date'].isoformat()
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple customers
    
    Request body:
    {
        "customers": [
            {"customer_id": 1, "loan_amount": 50000},
            {"customer_id": 2, "loan_amount": 75000}
        ]
    }
    """
    try:
        data = request.get_json()
        customers = data.get('customers', [])
        
        if not customers or len(customers) == 0:
            return jsonify({'error': 'No customers provided'}), 400
        
        if len(customers) > 100:
            return jsonify({'error': 'Batch size exceeds limit of 100'}), 400
        
        results = []
        for customer_data in customers:
            try:
                customer_id = customer_data['customer_id']
                loan_amount = customer_data['loan_amount']
                loan_term_months = customer_data.get('loan_term_months', 36)
                
                features = load_customer_features(engine, customer_id, loan_amount, loan_term_months)
                
                if features is not None:
                    risk_result = predict_risk(risk_model, features)
                    results.append({
                        'customer_id': customer_id,
                        'risk_score': float(risk_result['risk_score']),
                        'risk_category': risk_result['risk_category'],
                        'status': 'success'
                    })
                else:
                    results.append({
                        'customer_id': customer_id,
                        'status': 'error',
                        'message': 'Customer not found'
                    })
            except Exception as e:
                results.append({
                    'customer_id': customer_data.get('customer_id'),
                    'status': 'error',
                    'message': str(e)
                })
        
        return jsonify({
            'total': len(customers),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'error'),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the model"""
    try:
        # This would load feature importance from model metadata
        with open('models/saved_models/feature_importance.json', 'r') as f:
            feature_importance = json.load(f)
        
        return jsonify(feature_importance), 200
        
    except FileNotFoundError:
        return jsonify({'error': 'Feature importance data not available'}), 404
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def save_prediction(engine, customer_id, risk_result):
    """Save prediction to database"""
    try:
        query = """
            INSERT INTO predictions.risk_scores 
            (customer_id, risk_score, risk_category, default_probability, model_version)
            VALUES (:customer_id, :risk_score, :risk_category, :default_probability, :model_version)
        """
        
        with engine.connect() as conn:
            conn.execute(query, {
                'customer_id': customer_id,
                'risk_score': risk_result['risk_score'],
                'risk_category': risk_result['risk_category'],
                'default_probability': risk_result['default_probability'],
                'model_version': 'v1.0.0'
            })
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")