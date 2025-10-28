"""
Financial Risk Assessment ETL Pipeline DAG
Runs daily at 2 AM to process customer data and generate risk features
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Define the DAG
dag = DAG(
    'financial_risk_assessment',
    default_args=default_args,
    description='Daily risk assessment and feature engineering pipeline',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    catchup=False,  # Don't backfill on first run
    max_active_runs=1,  # Only one instance at a time
    tags=['risk', 'ml', 'etl']
)

# Task Functions

def check_data_freshness(**context):
    """Check if new data is available for processing"""
    logger.info("Checking data freshness...")
    
    hook = PostgresHook(postgres_conn_id='risk_db_connection')
    
    # Check latest data in staging
    query = """
        SELECT 
            MAX(created_at) as latest_customer,
            (SELECT MAX(created_at) FROM staging.credit_history) as latest_credit,
            (SELECT MAX(created_at) FROM staging.transactions) as latest_transaction,
            (SELECT MAX(created_at) FROM staging.loan_applications) as latest_loan
        FROM staging.customers
    """
    
    result = hook.get_first(query)
    
    if result and result[0]:
        logger.info(f"Latest customer data: {result[0]}")
        logger.info(f"Latest credit data: {result[1]}")
        logger.info(f"Latest transaction data: {result[2]}")
        logger.info(f"Latest loan data: {result[3]}")
        
        # Push to XCom for next tasks
        context['task_instance'].xcom_push(key='data_timestamp', value=str(result[0]))
        return True
    else:
        logger.warning("No data found in staging schema")
        return False

def run_data_quality_checks(**context):
    """Run comprehensive data quality checks"""
    logger.info("Running data quality checks...")
    
    from etl.data_quality import DataQualityChecker
    from sqlalchemy import create_engine
    from config import get_db_connection_string
    import pandas as pd
    
    engine = create_engine(get_db_connection_string())
    
    # Load data
    customers = pd.read_sql("SELECT * FROM staging.customers", engine)
    credit = pd.read_sql("SELECT * FROM staging.credit_history", engine)
    transactions = pd.read_sql("SELECT * FROM staging.transactions", engine)
    loans = pd.read_sql("SELECT * FROM staging.loan_applications", engine)
    
    # Initialize checker
    checker = DataQualityChecker()
    
    # Run checks
    checker.check_missing_values(customers, 'customers', threshold=0.1)
    checker.check_duplicates(customers, 'customers', ['customer_id'])
    checker.validate_business_rules(customers, 'customers')
    
    checker.check_missing_values(credit, 'credit_history', threshold=0.1)
    checker.validate_business_rules(credit, 'credit_history')
    
    checker.check_referential_integrity(
        customers, credit,
        'customer_id', 'customer_id',
        'customers -> credit_history'
    )
    
    # Get report
    report = checker.get_report()
    
    logger.info(f"Data Quality Summary: {report['summary']}")
    
    # Fail if there are critical errors
    if report['summary']['failures'] > 0:
        raise ValueError(f"Data quality checks failed: {report['summary']['failures']} failures")
    
    # Push report to XCom
    context['task_instance'].xcom_push(key='quality_report', value=report)
    
    return report['summary']

def run_etl_pipeline(**context):
    """Execute the ETL pipeline"""
    logger.info("Starting ETL pipeline...")
    
    from etl.pipeline import RiskETLPipeline
    from config import get_db_connection_string
    
    # Initialize pipeline
    pipeline = RiskETLPipeline(get_db_connection_string())
    
    # Run pipeline
    features = pipeline.run_pipeline()
    
    # Log statistics
    logger.info(f"ETL completed successfully")
    logger.info(f"Total features generated: {len(features)}")
    logger.info(f"Feature columns: {len(features.columns)}")
    
    # Push statistics to XCom
    context['task_instance'].xcom_push(key='feature_count', value=len(features))
    
    return len(features)

def train_model(**context):
    """Train machine learning models"""
    logger.info("Training models...")
    
    from models.training import RiskPredictionModel
    from sqlalchemy import create_engine
    from config import get_db_connection_string
    import pandas as pd
    import json
    
    # Load features
    engine = create_engine(get_db_connection_string())
    features = pd.read_sql("SELECT * FROM analytics.risk_features", engine)
    
    logger.info(f"Loaded {len(features)} feature records")
    
    # Initialize model
    model = RiskPredictionModel()
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(features)
    
    # Train models
    model.train_models(X_train, y_train)
    
    # Evaluate models
    results = model.evaluate_models(X_test, y_test)
    
    # Get feature importance
    feature_importance = model.get_feature_importance('random_forest')
    
    # Save best model
    model_path = 'models/saved_models/risk_model.pkl'
    model.save_model('random_forest', model_path)
    
    # Save feature importance
    importance_path = 'models/saved_models/feature_importance.json'
    feature_importance.to_json(importance_path, orient='records')
    
    # Save metrics
    metrics_path = 'models/saved_models/model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results['random_forest'], f, indent=2)
    
    logger.info(f"Model training completed")
    logger.info(f"Best model AUC-ROC: {results['random_forest']['auc_roc']:.4f}")
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(key='model_metrics', value=results['random_forest'])
    
    return results

def generate_batch_predictions(**context):
    """Generate predictions for all customers"""
    logger.info("Generating batch predictions...")
    
    from models.predictor import load_model
    from sqlalchemy import create_engine
    from config import get_db_connection_string
    import pandas as pd
    from datetime import datetime
    
    engine = create_engine(get_db_connection_string())
    
    # Load model
    model = load_model('models/saved_models/risk_model.pkl')
    
    # Load features for prediction
    query = """
        SELECT * FROM analytics.risk_features 
        WHERE customer_id NOT IN (
            SELECT customer_id FROM predictions.risk_scores 
            WHERE prediction_date >= CURRENT_DATE
        )
        LIMIT 1000
    """
    
    features = pd.read_sql(query, engine)
    
    if len(features) == 0:
        logger.info("No new customers to score")
        return 0
    
    logger.info(f"Generating predictions for {len(features)} customers")
    
    # Generate predictions
    feature_cols = [
        'age', 'annual_income', 'years_employed', 'num_dependents',
        'credit_score', 'num_credit_accounts', 'credit_utilization_ratio',
        'num_delinquent_accounts', 'total_debt', 'years_credit_history',
        'loan_amount', 'loan_term_months', 'interest_rate',
        'debt_to_income_ratio', 'employment_stability_score',
        'delinquency_score', 'loan_to_income_ratio',
        'employment_status_encoded', 'home_ownership_encoded'
    ]
    
    X = features[feature_cols].fillna(0)
    predictions = model.predict_proba(X)[:, 1]
    
    # Prepare results
    results_df = pd.DataFrame({
        'customer_id': features['customer_id'],
        'loan_id': features['loan_id'],
        'risk_score': predictions * 100,
        'default_probability': predictions,
        'risk_category': pd.cut(predictions * 100, 
                                bins=[0, 30, 60, 100],
                                labels=['Low', 'Medium', 'High']),
        'model_version': 'v1.0.0',
        'prediction_date': datetime.now()
    })
    
    # Save to database
    results_df.to_sql('risk_scores', engine, schema='predictions', 
                     if_exists='append', index=False)
    
    logger.info(f"Saved {len(results_df)} predictions to database")
    
    # Push count to XCom
    context['task_instance'].xcom_push(key='prediction_count', value=len(results_df))
    
    return len(results_df)

def send_summary_report(**context):
    """Send summary report of pipeline execution"""
    logger.info("Generating summary report...")
    
    ti = context['task_instance']
    
    # Retrieve metrics from XCom
    feature_count = ti.xcom_pull(task_ids='run_etl', key='feature_count')
    quality_report = ti.xcom_pull(task_ids='data_quality_check', key='quality_report')
    model_metrics = ti.xcom_pull(task_ids='train_model', key='model_metrics')
    prediction_count = ti.xcom_pull(task_ids='generate_predictions', key='prediction_count')
    
    summary = f"""
    ========================================
    Risk Assessment Pipeline Summary
    ========================================
    Execution Date: {datetime.now()}
    
    Data Quality:
    - Total Checks: {quality_report['summary']['total_checks'] if quality_report else 'N/A'}
    - Passed: {quality_report['summary']['passed'] if quality_report else 'N/A'}
    - Warnings: {quality_report['summary']['warnings'] if quality_report else 'N/A'}
    
    ETL:
    - Features Generated: {feature_count}
    
    Model Training:
    - Algorithm: Random Forest
    - Accuracy: {model_metrics['accuracy']:.4f if model_metrics else 'N/A'}
    - AUC-ROC: {model_metrics['auc_roc']:.4f if model_metrics else 'N/A'}
    
    Predictions:
    - New Predictions: {prediction_count}
    
    Status: SUCCESS ✓
    ========================================
    """
    
    logger.info(summary)
    
    # In production, send email or Slack notification here
    # send_email(summary)
    # send_slack_notification(summary)
    
    return summary

# Define Tasks

check_freshness = PythonOperator(
    task_id='check_data_freshness',
    python_callable=check_data_freshness,
    dag=dag
)

data_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=run_data_quality_checks,
    dag=dag
)

etl_task = PythonOperator(
    task_id='run_etl',
    python_callable=run_etl_pipeline,
    dag=dag
)

model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

prediction_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_batch_predictions,
    dag=dag
)

report_task = PythonOperator(
    task_id='send_summary_report',
    python_callable=send_summary_report,
    dag=dag,
    trigger_rule='all_done'  # Run even if upstream tasks fail
)

# Define task dependencies (pipeline flow)
check_freshness >> data_quality_check >> etl_task >> model_task >> prediction_task >> report_task