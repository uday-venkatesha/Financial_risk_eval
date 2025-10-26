from sqlalchemy import create_engine
import pandas as pd

def load_to_database(dataframes_dict, engine):
    """Load generated data to PostgreSQL"""
    for table_name, df in dataframes_dict.items():
        df.to_sql(table_name, engine, schema='staging', 
                 if_exists='replace', index=False)