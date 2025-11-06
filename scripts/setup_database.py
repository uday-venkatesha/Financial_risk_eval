#!/usr/bin/env python3
"""
Database setup script - Creates database and schemas
"""
import sys
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_database():
    """Create the risk_db database if it doesn't exist"""
    print("Setting up database...")
    
    # Get password
    import getpass
    password = getpass.getpass("Enter PostgreSQL password for user 'postgres': ")
    
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            password=password,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname='risk_db'")
        exists = cursor.fetchone()
        
        if not exists:
            print("Creating database 'risk_db'...")
            cursor.execute("CREATE DATABASE risk_db")
            print("✓ Database created successfully")
        else:
            print("✓ Database 'risk_db' already exists")
        
        cursor.close()
        conn.close()
        
        # Now connect to risk_db and create schemas
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            password=password,
            database='risk_db'
        )
        cursor = conn.cursor()
        
        # Read and execute schema.sql
        schema_file = project_root / 'sql' / 'schema.sql'
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        print("Creating schemas and tables...")
        cursor.execute(schema_sql)
        conn.commit()
        print("✓ Schemas and tables created successfully")
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*50)
        print("Database setup completed!")
        print("="*50)
        print("Database: risk_db")
        print("Schemas: staging, analytics, predictions")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == '__main__':
    create_database()