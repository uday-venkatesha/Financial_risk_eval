import yaml
import os

def get_db_connection_string(env='development'):
    with open('config/database.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config[env]
    return f"postgresql://{db_config['username']}:{db_config['password']}@" \
           f"{db_config['host']}:{db_config['port']}/{db_config['database']}"