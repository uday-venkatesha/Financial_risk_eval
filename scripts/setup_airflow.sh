#!/bin/bash
# Airflow Quick Setup Script for Financial Risk Assessment

set -e  # Exit on error

echo "======================================"
echo "Airflow Setup for Risk Assessment"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="$PROJECT_ROOT/airflow"

echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}Airflow home: $AIRFLOW_HOME${NC}"

# Step 1: Set AIRFLOW_HOME
echo ""
echo "Step 1: Setting AIRFLOW_HOME environment variable..."
export AIRFLOW_HOME="$AIRFLOW_HOME"

# Create AIRFLOW_HOME and common subdirectories
mkdir -p "$AIRFLOW_HOME/dags" "$AIRFLOW_HOME/plugins" "$AIRFLOW_HOME/logs"
echo -e "${GREEN}✓ Directories created${NC}"

# Step 2: Check if Airflow is installed
echo ""
echo "Step 2: Checking Airflow installation..."
if ! command -v airflow &> /dev/null; then
    echo "Airflow not found. Installing..."
    python3 -m pip install --user apache-airflow==2.6.3
    python3 -m pip install --user apache-airflow-providers-postgres==5.5.1
    echo -e "${GREEN}✓ Airflow installed${NC}"
else
    echo -e "${GREEN}✓ Airflow already installed${NC}"
fi

# Step 3: Initialize Airflow database
echo ""
echo "Step 3: Initializing Airflow database..."
airflow db init
echo -e "${GREEN}✓ Airflow database initialized${NC}"

# Step 4: Create admin user
echo ""
echo "Step 4: Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email udayvenkatesh2015@gmail.com \
    --password admin 2>/dev/null || echo -e "${GREEN}✓ Admin user already exists${NC}"

# Step 5: Create Airflow configuration
echo ""
echo "Step 5: Configuring Airflow..."
cat > "$AIRFLOW_HOME/airflow.cfg" << EOF
[core]
dags_folder = $AIRFLOW_HOME/dags
plugins_folder = $AIRFLOW_HOME/plugins
load_examples = False
default_timezone = America/Chicago

[logging]
base_log_folder = $AIRFLOW_HOME/logs

[webserver]
base_url = http://localhost:8080
web_server_port = 8080
expose_config = True

[scheduler]
dag_dir_list_interval = 60
catchup_by_default = False

[api]
auth_backends = airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session
EOF
echo -e "${GREEN}✓ Airflow configuration created${NC}"

# Step 6: Create PostgreSQL connection
echo ""
echo "Step 6: Setting up PostgreSQL connection..."
echo "Please enter your PostgreSQL details:"
read -p "Host [localhost]: " DB_HOST
DB_HOST=${DB_HOST:-localhost}

read -p "Port [5432]: " DB_PORT
DB_PORT=${DB_PORT:-5432}

read -p "Database [risk_db]: " DB_NAME
DB_NAME=${DB_NAME:-risk_db}

read -p "Username [postgres]: " DB_USER
DB_USER=${DB_USER:-postgres}

read -sp "Password: " DB_PASSWORD
echo ""

# Add connection
airflow connections delete 'risk_db_connection' 2>/dev/null || true
airflow connections add 'risk_db_connection' \
    --conn-type 'postgres' \
    --conn-host "$DB_HOST" \
    --conn-login "$DB_USER" \
    --conn-password "$DB_PASSWORD" \
    --conn-schema "$DB_NAME" \
    --conn-port "$DB_PORT"
echo -e "${GREEN}✓ PostgreSQL connection configured${NC}"

# Step 7: Test DAG
echo ""
echo "Step 7: Testing DAG file..."
if [ -f "$AIRFLOW_HOME/dags/risk_assessment_dag.py" ]; then
    if python3 -m py_compile "$AIRFLOW_HOME/dags/risk_assessment_dag.py" 2>/dev/null; then
        echo -e "${GREEN}✓ DAG file is syntactically valid${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: DAG file has syntax errors. Check $AIRFLOW_HOME/dags/risk_assessment_dag.py${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Warning: DAG file not found at $AIRFLOW_HOME/dags/risk_assessment_dag.py${NC}"
fi

# Step 8: Create logs directory
echo ""
echo "Step 8: Creating logs directory..."
mkdir -p "$PROJECT_ROOT/logs"
echo -e "${GREEN}✓ Logs directory created${NC}"

# Step 9: Test database connection
echo ""
echo "Step 9: Testing database connection..."
if airflow connections test risk_db_connection 2>&1 | grep -q "Connection success"; then
    echo -e "${GREEN}✓ Database connection successful${NC}"
else
    echo -e "${RED}✗ Database connection failed${NC}"
    echo -e "${YELLOW}Please check your PostgreSQL credentials and ensure the database is running${NC}"
fi

# Print summary
echo ""
echo "======================================"
echo -e "${GREEN}Airflow Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Start the Airflow scheduler:"
echo "   airflow scheduler"
echo ""
echo "2. In a new terminal, start the webserver:"
echo "   airflow webserver --port 8080"
echo ""
echo "3. Access Airflow UI:"
echo "   http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "4. Your DAG 'financial_risk_assessment' should appear in the UI"
echo ""
echo "To trigger the DAG manually:"
echo "   airflow dags trigger financial_risk_assessment"
echo ""
echo "======================================"