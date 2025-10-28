#!/usr/bin/env python3
"""
Model Deployment Script
Handles model deployment to production environment
"""
import sys
import os
sys.path.append('..')

import pickle
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
from sqlalchemy import create_engine
from config import get_db_connection_string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeployer:
    """Handle model deployment operations"""
    
    def __init__(self, model_path, model_name, version):
        self.model_path = model_path
        self.model_name = model_name
        self.version = version
        self.deployment_dir = Path('models/saved_models/production')
        self.backup_dir = Path('models/saved_models/backups')
        
    def validate_model(self):
        """Validate model file before deployment"""
        logger.info("Validating model...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Check if model has required methods
            required_methods = ['predict', 'predict_proba']
            for method in required_methods:
                if not hasattr(model, method):
                    raise ValueError(f"Model missing required method: {method}")
            
            logger.info("Model validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise
    
    def backup_existing_model(self):
        """Backup currently deployed model"""
        logger.info("Backing up existing model...")
        
        production_model = self.deployment_dir / 'risk_model.pkl'
        
        if production_model.exists():
            # Create backup directory if it doesn't exist
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f'risk_model_{timestamp}.pkl'
            
            shutil.copy2(production_model, backup_path)
            logger.info(f"Backup created: {backup_path}")
        else:
            logger.info("No existing model to backup")
    
    def deploy_model(self):
        """Deploy model to production"""
        logger.info(f"Deploying model {self.model_name} version {self.version}...")
        
        # Create production directory if it doesn't exist
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model to production
        production_path = self.deployment_dir / 'risk_model.pkl'
        shutil.copy2(self.model_path, production_path)
        
        logger.info(f"Model deployed to: {production_path}")
        
        # Create version file
        version_info = {
            'model_name': self.model_name,
            'version': self.version,
            'deployment_date': datetime.now().isoformat(),
            'source_path': str(self.model_path)
        }
        
        version_path = self.deployment_dir / 'version.json'
        with open(version_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        logger.info(f"Version info saved to: {version_path}")
    
    def update_database_metadata(self, metrics):
        """Update model metadata in database"""
        logger.info("Updating database metadata...")
        
        try:
            engine = create_engine(get_db_connection_string())
            
            query = """
                INSERT INTO predictions.model_metadata 
                (model_name, model_version, algorithm, accuracy, precision_score, 
                 recall, f1_score, auc_roc, training_date, features_used)
                VALUES (:model_name, :model_version, :algorithm, :accuracy, 
                        :precision_score, :recall, :f1_score, :auc_roc, 
                        :training_date, :features_used)
            """
            
            with engine.connect() as conn:
                conn.execute(query, {
                    'model_name': self.model_name,
                    'model_version': self.version,
                    'algorithm': metrics.get('algorithm', 'RandomForestClassifier'),
                    'accuracy': metrics.get('accuracy', 0),
                    'precision_score': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'auc_roc': metrics.get('auc_roc', 0),
                    'training_date': datetime.now(),
                    'features_used': json.dumps(metrics.get('features', []))
                })
                conn.commit()
            
            logger.info("Database metadata updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update database metadata: {str(e)}")
            raise
    
    def create_deployment_report(self):
        """Generate deployment report"""
        logger.info("Creating deployment report...")
        
        report = {
            'deployment_date': datetime.now().isoformat(),
            'model_name': self.model_name,
            'version': self.version,
            'source_path': str(self.model_path),
            'deployment_path': str(self.deployment_dir / 'risk_model.pkl'),
            'status': 'success'
        }
        
        report_path = self.deployment_dir / 'deployment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Deployment report saved to: {report_path}")
        return report
    
    def rollback(self, backup_timestamp):
        """Rollback to a previous model version"""
        logger.info(f"Rolling back to backup: {backup_timestamp}")
        
        backup_path = self.backup_dir / f'risk_model_{backup_timestamp}.pkl'
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        production_path = self.deployment_dir / 'risk_model.pkl'
        shutil.copy2(backup_path, production_path)
        
        logger.info("Rollback completed successfully")
    
    def test_deployment(self):
        """Test deployed model"""
        logger.info("Testing deployed model...")
        
        production_path = self.deployment_dir / 'risk_model.pkl'
        
        try:
            with open(production_path, 'rb') as f:
                model = pickle.load(f)
            
            # Create dummy test data
            import numpy as np
            test_features = np.random.randn(1, 19)  # 19 features
            
            # Test prediction
            prediction = model.predict(test_features)
            prob = model.predict_proba(test_features)
            
            logger.info("Deployment test successful")
            logger.info(f"Test prediction: {prediction[0]}")
            logger.info(f"Test probability: {prob[0]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment test failed: {str(e)}")
            raise

def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy risk assessment model')
    parser.add_argument('--model-path', required=True, help='Path to model file')
    parser.add_argument('--model-name', default='Risk Prediction Model', 
                       help='Name of the model')
    parser.add_argument('--version', required=True, help='Model version (e.g., v1.0.0)')
    parser.add_argument('--skip-backup', action='store_true', 
                       help='Skip backing up existing model')
    parser.add_argument('--skip-test', action='store_true', 
                       help='Skip deployment test')
    
    args = parser.parse_args()
    
    try:
        # Initialize deployer
        deployer = ModelDeployer(args.model_path, args.model_name, args.version)
        
        # Validate model
        deployer.validate_model()
        
        # Backup existing model
        if not args.skip_backup:
            deployer.backup_existing_model()
        
        # Deploy model
        deployer.deploy_model()
        
        # Load model metrics (from training)
        metrics_path = Path(args.model_path).parent / 'model_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            deployer.update_database_metadata(metrics)
        else:
            logger.warning("Model metrics file not found, skipping database update")
        
        # Test deployment
        if not args.skip_test:
            deployer.test_deployment()
        
        # Create deployment report
        report = deployer.create_deployment_report()
        
        logger.info("="*60)
        logger.info("DEPLOYMENT SUCCESSFUL")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Version: {args.version}")
        logger.info(f"Deployment Date: {report['deployment_date']}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()