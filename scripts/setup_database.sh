#!/bin/bash
# Database setup script

echo "Setting up PostgreSQL database..."

# Create database
psql -U postgres -c "CREATE DATABASE risk_db;"

# Run schema
psql -U postgres -d risk_db -f ../sql/schema.sql

echo "Database setup completed!"