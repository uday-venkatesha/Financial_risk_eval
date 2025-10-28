import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Data quality validation and monitoring"""
    
    def __init__(self):
        self.quality_report = {
            'timestamp': datetime.now(),
            'checks': [],
            'warnings': [],
            'errors': []
        }
    
    def check_missing_values(self, df, table_name, threshold=0.3):
        """Check for missing values in dataframe"""
        logger.info(f"Checking missing values in {table_name}...")
        
        missing_pct = df.isnull().sum() / len(df)
        cols_with_missing = missing_pct[missing_pct > 0]
        
        for col, pct in cols_with_missing.items():
            if pct > threshold:
                error_msg = f"{table_name}.{col} has {pct:.2%} missing values (threshold: {threshold:.2%})"
                logger.error(error_msg)
                self.quality_report['errors'].append(error_msg)
            else:
                warning_msg = f"{table_name}.{col} has {pct:.2%} missing values"
                logger.warning(warning_msg)
                self.quality_report['warnings'].append(warning_msg)
        
        self.quality_report['checks'].append({
            'check': 'missing_values',
            'table': table_name,
            'status': 'pass' if len(cols_with_missing) == 0 else 'warning',
            'details': cols_with_missing.to_dict()
        })
        
        return len(cols_with_missing) == 0
    
    def check_duplicates(self, df, table_name, key_columns):
        """Check for duplicate records"""
        logger.info(f"Checking duplicates in {table_name}...")
        
        duplicates = df.duplicated(subset=key_columns, keep=False).sum()
        
        if duplicates > 0:
            error_msg = f"{table_name} has {duplicates} duplicate records"
            logger.error(error_msg)
            self.quality_report['errors'].append(error_msg)
            status = 'fail'
        else:
            logger.info(f"{table_name}: No duplicates found")
            status = 'pass'
        
        self.quality_report['checks'].append({
            'check': 'duplicates',
            'table': table_name,
            'status': status,
            'duplicate_count': int(duplicates)
        })
        
        return duplicates == 0
    
    def check_value_ranges(self, df, table_name, range_specs):
        """
        Check if values are within expected ranges
        
        range_specs: dict like {'column': {'min': 0, 'max': 100}}
        """
        logger.info(f"Checking value ranges in {table_name}...")
        
        range_violations = []
        
        for col, specs in range_specs.items():
            if col not in df.columns:
                continue
            
            if 'min' in specs:
                violations = (df[col] < specs['min']).sum()
                if violations > 0:
                    msg = f"{table_name}.{col}: {violations} values below minimum {specs['min']}"
                    logger.warning(msg)
                    range_violations.append(msg)
            
            if 'max' in specs:
                violations = (df[col] > specs['max']).sum()
                if violations > 0:
                    msg = f"{table_name}.{col}: {violations} values above maximum {specs['max']}"
                    logger.warning(msg)
                    range_violations.append(msg)
        
        self.quality_report['checks'].append({
            'check': 'value_ranges',
            'table': table_name,
            'status': 'pass' if len(range_violations) == 0 else 'warning',
            'violations': range_violations
        })
        
        return len(range_violations) == 0
    
    def check_data_types(self, df, table_name, expected_types):
        """
        Verify data types match expectations
        
        expected_types: dict like {'column': 'int64'}
        """
        logger.info(f"Checking data types in {table_name}...")
        
        type_mismatches = []
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue
            
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                msg = f"{table_name}.{col}: expected {expected_type}, got {actual_type}"
                logger.warning(msg)
                type_mismatches.append(msg)
        
        self.quality_report['checks'].append({
            'check': 'data_types',
            'table': table_name,
            'status': 'pass' if len(type_mismatches) == 0 else 'warning',
            'mismatches': type_mismatches
        })
        
        return len(type_mismatches) == 0
    
    def check_referential_integrity(self, parent_df, child_df, parent_key, child_key, relation_name):
        """Check foreign key relationships"""
        logger.info(f"Checking referential integrity: {relation_name}...")
        
        parent_ids = set(parent_df[parent_key].unique())
        child_ids = set(child_df[child_key].unique())
        
        orphaned_records = child_ids - parent_ids
        
        if len(orphaned_records) > 0:
            error_msg = f"{relation_name}: {len(orphaned_records)} orphaned records found"
            logger.error(error_msg)
            self.quality_report['errors'].append(error_msg)
            status = 'fail'
        else:
            logger.info(f"{relation_name}: All foreign keys valid")
            status = 'pass'
        
        self.quality_report['checks'].append({
            'check': 'referential_integrity',
            'relation': relation_name,
            'status': status,
            'orphaned_count': len(orphaned_records)
        })
        
        return len(orphaned_records) == 0
    
    def check_statistical_anomalies(self, df, table_name, numerical_columns):
        """Detect statistical outliers"""
        logger.info(f"Checking statistical anomalies in {table_name}...")
        
        anomalies = []
        
        for col in numerical_columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            # Z-score method
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > 3).sum()
            
            if outliers > 0:
                pct = outliers / len(df) * 100
                if pct > 5:  # More than 5% outliers
                    msg = f"{table_name}.{col}: {outliers} outliers ({pct:.2f}%)"
                    logger.warning(msg)
                    anomalies.append(msg)
        
        self.quality_report['checks'].append({
            'check': 'statistical_anomalies',
            'table': table_name,
            'status': 'pass' if len(anomalies) == 0 else 'warning',
            'anomalies': anomalies
        })
        
        return len(anomalies) == 0
    
    def check_data_freshness(self, df, table_name, date_column, max_age_days=7):
        """Check if data is recent enough"""
        logger.info(f"Checking data freshness in {table_name}...")
        
        if date_column not in df.columns:
            logger.warning(f"Date column {date_column} not found in {table_name}")
            return True
        
        df[date_column] = pd.to_datetime(df[date_column])
        max_date = df[date_column].max()
        age_days = (datetime.now() - max_date).days
        
        if age_days > max_age_days:
            warning_msg = f"{table_name}: Data is {age_days} days old (threshold: {max_age_days})"
            logger.warning(warning_msg)
            self.quality_report['warnings'].append(warning_msg)
            status = 'warning'
        else:
            status = 'pass'
        
        self.quality_report['checks'].append({
            'check': 'data_freshness',
            'table': table_name,
            'status': status,
            'age_days': age_days,
            'max_age_days': max_age_days
        })
        
        return age_days <= max_age_days
    
    def validate_business_rules(self, df, table_name):
        """Validate domain-specific business rules"""
        logger.info(f"Validating business rules for {table_name}...")
        
        violations = []
        
        if table_name == 'customers':
            # Age should be between 18 and 100
            if 'age' in df.columns:
                invalid_age = ((df['age'] < 18) | (df['age'] > 100)).sum()
                if invalid_age > 0:
                    violations.append(f"{invalid_age} customers with invalid age")
            
            # Annual income should be positive
            if 'annual_income' in df.columns:
                invalid_income = (df['annual_income'] <= 0).sum()
                if invalid_income > 0:
                    violations.append(f"{invalid_income} customers with non-positive income")
        
        elif table_name == 'credit_history':
            # Credit score should be between 300 and 850
            if 'credit_score' in df.columns:
                invalid_score = ((df['credit_score'] < 300) | (df['credit_score'] > 850)).sum()
                if invalid_score > 0:
                    violations.append(f"{invalid_score} records with invalid credit score")
            
            # Credit utilization should be between 0 and 1
            if 'credit_utilization_ratio' in df.columns:
                invalid_util = ((df['credit_utilization_ratio'] < 0) | 
                               (df['credit_utilization_ratio'] > 1)).sum()
                if invalid_util > 0:
                    violations.append(f"{invalid_util} records with invalid credit utilization")
        
        elif table_name == 'loan_applications':
            # Loan amount should be positive
            if 'loan_amount' in df.columns:
                invalid_amount = (df['loan_amount'] <= 0).sum()
                if invalid_amount > 0:
                    violations.append(f"{invalid_amount} loans with non-positive amount")
            
            # Interest rate should be positive and reasonable
            if 'interest_rate' in df.columns:
                invalid_rate = ((df['interest_rate'] <= 0) | (df['interest_rate'] > 0.5)).sum()
                if invalid_rate > 0:
                    violations.append(f"{invalid_rate} loans with invalid interest rate")
        
        for violation in violations:
            logger.warning(violation)
            self.quality_report['warnings'].append(violation)
        
        self.quality_report['checks'].append({
            'check': 'business_rules',
            'table': table_name,
            'status': 'pass' if len(violations) == 0 else 'warning',
            'violations': violations
        })
        
        return len(violations) == 0
    
    def get_report(self):
        """Get comprehensive quality report"""
        total_checks = len(self.quality_report['checks'])
        passed = sum(1 for c in self.quality_report['checks'] if c['status'] == 'pass')
        warnings = sum(1 for c in self.quality_report['checks'] if c['status'] == 'warning')
        failures = sum(1 for c in self.quality_report['checks'] if c['status'] == 'fail')
        
        summary = {
            'total_checks': total_checks,
            'passed': passed,
            'warnings': warnings,
            'failures': failures,
            'success_rate': f"{(passed/total_checks*100):.2f}%" if total_checks > 0 else "0%"
        }
        
        return {
            'summary': summary,
            'details': self.quality_report
        }


def run_quality_checks(customers_df, credit_df, transactions_df, loans_df):
    """Run all quality checks on datasets"""
    checker = DataQualityChecker()
    
    # Customers checks
    checker.check_missing_values(customers_df, 'customers', threshold=0.1)
    checker.check_duplicates(customers_df, 'customers', ['customer_id'])
    checker.check_value_ranges(customers_df, 'customers', {
        'age': {'min': 18, 'max': 100},
        'annual_income': {'min': 0}
    })
    checker.validate_business_rules(customers_df, 'customers')
    
    # Credit history checks
    checker.check_missing_values(credit_df, 'credit_history', threshold=0.1)
    checker.check_duplicates(credit_df, 'credit_history', ['customer_id'])
    checker.check_value_ranges(credit_df, 'credit_history', {
        'credit_score': {'min': 300, 'max': 850},
        'credit_utilization_ratio': {'min': 0, 'max': 1}
    })
    checker.validate_business_rules(credit_df, 'credit_history')
    
    # Referential integrity
    checker.check_referential_integrity(
        customers_df, credit_df,
        'customer_id', 'customer_id',
        'customers -> credit_history'
    )
    
    checker.check_referential_integrity(
        customers_df, transactions_df,
        'customer_id', 'customer_id',
        'customers -> transactions'
    )
    
    checker.check_referential_integrity(
        customers_df, loans_df,
        'customer_id', 'customer_id',
        'customers -> loans'
    )
    
    # Loans checks
    checker.check_missing_values(loans_df, 'loan_applications', threshold=0.1)
    checker.validate_business_rules(loans_df, 'loan_applications')
    
    # Get final report
    report = checker.get_report()
    logger.info(f"Data Quality Report: {report['summary']}")
    
    return report