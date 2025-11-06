#!/usr/bin/env python3
"""
Standalone ETL pipeline execution
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from etl.pipeline import RiskETLPipeline
from config import get_db_connection_string

def main():
    print("Starting ETL pipeline...")
    pipeline = RiskETLPipeline(get_db_connection_string())
    features = pipeline.run_pipeline()
    print(f"\nETL completed successfully!")
    print(f"Generated {len(features)} feature records.")

if __name__ == '__main__':
    main()