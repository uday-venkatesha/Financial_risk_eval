from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
import logging
from logging.config import dictConfig
import yaml

def create_app(config_name='development'):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configure logging
    with open('config/logging_config.yaml', 'r') as f:
        logging_config = yaml.safe_load(f)
        dictConfig(logging_config)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'risk-assessment-api'}, 200
    
    @app.route('/')
    def index():
        return {
            'message': 'Financial Risk Assessment API',
            'version': '1.0.0',
            'endpoints': {
                'predict': '/api/predict',
                'risk_score': '/api/risk-score/<customer_id>',
                'model_metrics': '/api/model-metrics',
                'health': '/health'
            }
        }, 200
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Endpoint not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Internal server error: {str(error)}')
        return {'error': 'Internal server error'}, 500
    
    return app

def main():
    """Run the Flask development server"""
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

if __name__ == '__main__':
    main()