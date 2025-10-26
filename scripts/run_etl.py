#!/usr/bin/env python3
"""
Standalone ETL pipeline execution
"""
import sys
sys.path.append('..')

from etl.pipeline import RiskETLPipeline
from config.database import get_db_connection_string

def main():
    pipeline = RiskETLPipeline(get_db_connection_string())
    features = pipeline.run_pipeline()
    print(f"ETL completed. Generated {len(features)} feature records.")

if __name__ == '__main__':
    main()