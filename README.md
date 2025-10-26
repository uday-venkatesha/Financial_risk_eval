# Financial Risk Assessment Model

## Project Overview

The Financial Risk Assessment Model is an end-to-end machine learning solution designed to enhance loan default prediction accuracy and automate risk assessment workflows for financial institutions. This project leverages advanced data analytics, machine learning, and interactive visualization to help credit teams make faster, data-driven lending decisions.

## Key Features

- **Predictive Risk Modeling**: ML models (Logistic Regression, Random Forest, Gradient Boosting) for accurate loan default prediction
- **Automated ETL Pipeline**: Apache Airflow-orchestrated data processing workflows
- **Real-time Risk API**: Flask-based REST API for on-demand risk scoring
- **Interactive Dashboards**: Tableau visualizations for risk monitoring and insights
- **Data Governance**: Robust security, compliance, and data quality controls
- **Scalable Architecture**: PostgreSQL database with staging and analytics schemas

## Business Impact

- Improved loan default prediction accuracy
- Faster credit decision-making through automation
- Proactive identification of high-risk accounts
- Reduced financial losses through targeted interventions
- Enhanced regulatory compliance and reporting

## Project Structure

```
financial_risk_eval/
├── airflow/                    # Airflow DAGs and configuration
│   ├── dags/                   # Pipeline orchestration
│   └── config/                 # Airflow settings
├── api/                        # Flask REST API
│   ├── app.py                  # API application
│   ├── routes.py               # API endpoints
│   └── utils.py                # Helper functions
├── config/                     # Configuration files
│   ├── database.yaml           # Database connections
│   ├── logging_config.yaml     # Logging setup
│   └── model_config.yaml       # Model parameters
├── data/                       # Data generation
│   ├── generator.py            # Synthetic data generator
│   └── loader.py               # Database loader
├── etl/                        # ETL pipeline
│   ├── pipeline.py             # Main ETL logic
│   ├── feature_engineering.py  # Feature creation
│   └── data_quality.py         # Data validation
├── models/                     # ML models
│   ├── training.py             # Model training
│   ├── predictor.py            # Prediction utilities
│   ├── evaluation.py           # Model evaluation
│   └── saved_models/           # Serialized models
├── sql/                        # Database schemas and queries
│   ├── schema.sql              # Database structure
│   └── queries/                # Reusable SQL queries
├── notebooks/                  # Jupyter notebooks for analysis
├── dashboards/                 # Tableau dashboards
├── docs/                       # Documentation
└── scripts/                    # Utility scripts

```

## Tech Stack

- **Languages**: Python 3.9+
- **ML Libraries**: scikit-learn, XGBoost, imbalanced-learn
- **Data Processing**: pandas, NumPy, PySpark
- **Database**: PostgreSQL, SQLAlchemy
- **Orchestration**: Apache Airflow
- **API**: Flask, Flask-CORS
- **Visualization**: Tableau, Matplotlib, Seaborn
- **Monitoring**: MLflow

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 12+
- Apache Airflow 2.6+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd financial_risk_eval
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure database:
```bash
# Update config/database.yaml with your credentials
# Run database setup
chmod +x scripts/setup_database.sh
./scripts/setup_database.sh
```

5. Set environment variables:
```bash
export DB_PASSWORD=your_password
export FLASK_ENV=development
```

## Usage

### 1. Generate Synthetic Data

```bash
python scripts/generate_data.py
```

### 2. Run ETL Pipeline

```bash
# Standalone execution
python scripts/run_etl.py

# Or via Airflow
airflow dags trigger financial_risk_assessment
```

### 3. Train Models

```bash
python scripts/train_model.py
```

### 4. Start API Server

```bash
python api/app.py
# API available at http://localhost:5000
```

### 5. Make Predictions

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 123, "loan_amount": 50000}'
```

## API Endpoints

- `POST /api/predict` - Get risk prediction for a customer
- `GET /api/risk-score/{customer_id}` - Retrieve existing risk score
- `GET /api/model-metrics` - View model performance metrics
- `GET /api/health` - API health check

See [API Documentation](docs/api_documentation.md) for details.

## Model Performance

Current production model (Random Forest):
- **Accuracy**: 87.3%
- **Precision**: 84.6%
- **Recall**: 79.2%
- **F1-Score**: 81.8%
- **AUC-ROC**: 0.91

## Data Pipeline

1. **Extraction**: Raw data from staging schema
2. **Transformation**: Feature engineering, data quality checks
3. **Loading**: Processed features to analytics schema
4. **Training**: Model training and evaluation
5. **Deployment**: Model serving via API

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## Data Governance

- **Access Control**: Role-based access to sensitive data
- **Audit Logging**: All data access and model predictions logged
- **Encryption**: Data encrypted at rest and in transit
- **Compliance**: GDPR and financial regulation adherence
- **Data Quality**: Automated validation and monitoring

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add new feature'`
3. Push branch: `git push origin feature/your-feature`
4. Submit pull request

## Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check credentials in `config/database.yaml`
- Ensure database and schemas exist

### Airflow DAG Not Appearing
- Check DAG file syntax
- Verify Airflow home directory
- Review Airflow logs

### Model Training Failures
- Ensure sufficient data in analytics schema
- Check for missing values in features
- Verify model configuration

## License

Proprietary - Internal Use Only

## Contact

For questions or support, contact the Data Science Team.

## Acknowledgments

- Data Engineering Team
- Risk Management Department
- Credit Analysis Team