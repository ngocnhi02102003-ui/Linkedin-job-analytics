import sys
from pathlib import Path
import pandas as pd

# Đảm bảo import được config
sys.path.append(str(Path(__file__).resolve().parent))
import config

def main():
    print("Starting data audit...")
    data_dir = config.DATA_RAW_DIR
    out_dir = config.TABLES_DIR
    
    # 1. Load tất cả bảng
    tables = {}
    csv_files = {
        'postings': 'postings.csv',
        'companies': 'companies.csv',
        'employee_counts': 'employee_counts.csv',
        'job_skills': 'job_skills.csv',
        'job_industries': 'job_industries.csv',
        'salaries': 'salaries.csv',
        'industries': 'industries.csv',
        'skills': 'skills.csv',
        'benefits': 'benefits.csv',
        'company_industries': 'company_industries.csv',
        'company_specialities': 'company_specialities.csv'
    }
    
    for name, file_name in list(csv_files.items()):
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"Loading {name}...")
            # Dùng low_memory=False để tránh cảnh báo Mixed Types khi load file lớn
            tables[name] = pd.read_csv(file_path, low_memory=False)
        else:
            print(f"Warning: {file_name} not found.")

    # 2, 3, 4, 5. Kiểm tra shape, missing, dtype, duplicate full rows
    audit_results = []
    
    for name, df in tables.items():
        print(f"Auditing {name}...")
        n_rows, n_cols = df.shape
        full_dups = df.duplicated().sum()
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / n_rows if n_rows > 0 else 0
            audit_results.append({
                'table_name': name,
                'column_name': col,
                'n_rows': n_rows,
                'n_cols': n_cols,
                'dtype': str(df[col].dtype),
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'full_row_duplicates': full_dups
            })
            
    df_audit_summary = pd.DataFrame(audit_results)
    df_audit_summary.to_csv(out_dir / 'data_audit_summary.csv', index=False)
    print("Saved data_audit_summary.csv")

    # 6. Kiểm tra duplicate theo key
    print("Checking key duplicates...")
    keys_to_check = {
        'postings': ['job_id'],
        'companies': ['company_id'],
        'employee_counts': ['company_id'],
        'job_skills': ['job_id', 'skill_abr'],
        'job_industries': ['job_id', 'industry_id'],
        'salaries': ['job_id'],
        'industries': ['industry_id'],
        'skills': ['skill_abr']
    }
    
    key_dup_results = []
    for table_name, keys in keys_to_check.items():
        if table_name in tables:
            df = tables[table_name]
            n_rows = len(df)
            dup_count = df.duplicated(subset=keys).sum()
            key_dup_results.append({
                'table_name': table_name,
                'key_columns': ", ".join(keys),
                'total_rows': n_rows,
                'key_duplicates': dup_count,
                'duplicate_pct': dup_count / n_rows if n_rows > 0 else 0
            })
            
    df_key_dup = pd.DataFrame(key_dup_results)
    df_key_dup.to_csv(out_dir / 'key_dup_report.csv', index=False)
    print("Saved key_dup_report.csv")
    
    # 7. Kiểm tra integrity sơ bộ
    print("Checking referential integrity...")
    integrity_checks = [
        {'child': 'postings', 'child_key': 'company_id', 'parent': 'companies', 'parent_key': 'company_id'},
        {'child': 'job_skills', 'child_key': 'job_id', 'parent': 'postings', 'parent_key': 'job_id'},
        {'child': 'job_industries', 'child_key': 'job_id', 'parent': 'postings', 'parent_key': 'job_id'},
        {'child': 'salaries', 'child_key': 'job_id', 'parent': 'postings', 'parent_key': 'job_id'}
    ]
    
    integrity_results = []
    for check in integrity_checks:
        c_name = check['child']
        p_name = check['parent']
        c_key = check['child_key']
        p_key = check['parent_key']
        
        if c_name in tables and p_name in tables:
            child_df = tables[c_name]
            parent_df = tables[p_name]
            
            # Keys
            child_keys = child_df[c_key].dropna().unique()
            parent_keys = parent_df[p_key].dropna().unique()
            
            invalid_keys = set(child_keys) - set(parent_keys)
            invalid_rows = child_df[child_df[c_key].isin(invalid_keys)]
            
            invalid_rows_count = len(invalid_rows)
            total_child_rows = len(child_df)
            
            integrity_results.append({
                'child_table': c_name,
                'child_key': c_key,
                'parent_table': p_name,
                'parent_key': p_key,
                'invalid_records_count': invalid_rows_count,
                'invalid_records_pct': invalid_rows_count / total_child_rows if total_child_rows > 0 else 0
            })
            
    df_integrity = pd.DataFrame(integrity_results)
    df_integrity.to_csv(out_dir / 'referential_integrity_report.csv', index=False)
    print("Saved referential_integrity_report.csv")
    
    print("\n=== AUDIT RESULTS ===")
    
    print("\n[KEY DUPLICATES REPORT]")
    print(df_key_dup.to_string(index=False))
    
    print("\n[REFERENTIAL INTEGRITY REPORT]")
    print(df_integrity.to_string(index=False))

    print("\n[MISSING DATA SUMMARY BY TABLE]")
    missing_by_table = df_audit_summary.groupby('table_name').agg(
        total_rows=('n_rows', 'first'),
        avg_missing_pct=('missing_pct', 'mean'),
        cols_with_missing=('missing_count', lambda x: (x > 0).sum())
    ).reset_index()
    # Sort for better summary reading
    missing_by_table = missing_by_table.sort_values(by='avg_missing_pct', ascending=False)
    print(missing_by_table.to_string(index=False))

if __name__ == "__main__":
    main()
