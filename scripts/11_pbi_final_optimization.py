"""
11_pbi_final_optimization.py
Final export script for Power BI Dashboard with Data Integrity Fixes.
Handles metric calculation, Star Schema, and Referential Integrity.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent))
import config

# ── Paths ─────────────────────────────────────────────────────
PRO_DIR    = config.DATA_PROCESSED_DIR
RAW_DIR    = config.DATA_RAW_DIR
PBI_DIR    = PRO_DIR / "powerbi"
PBI_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. LOAD & ENRICH
# ════════════════════════════════════════════════════════════
def load_and_enrich():
    print("Loading master and predictions …")
    df = pd.read_csv(PRO_DIR / "jobs_master.csv", low_memory=False)
    
    # Load ML predictions
    try:
        df_sal_p = pd.read_csv(PRO_DIR / "salary_predictions.csv")
        sal_pred_map = df_sal_p.set_index("job_id")["pred_salary_best"].to_dict()
    except:
        sal_pred_map = {}

    try:
        df_hot_p = pd.read_csv(PRO_DIR / "hot_job_predictions.csv")
        hot_prob_map = df_hot_p.set_index("job_id")["hot_job_prob"].to_dict()
    except:
        hot_prob_map = {}

    # --- METRIC CALCULATIONS ---
    print("Calculating final metrics …")
    
    # Salary Metrics
    df["has_salary"] = (df["normalized_salary"] > 0).astype(int)
    df["salary_disclosure_status"] = df["has_salary"].map({1: "Disclosed", 0: "Hidden"})
    df["pred_salary"] = df["job_id"].map(sal_pred_map)
    
    # Performance Metrics
    df["applies_per_view"] = (df["applies"] / df["views"].replace(0, np.nan)).fillna(0)
    df["engagement_score"] = df["views"].fillna(0) + (df["applies"].fillna(0) * 10)
    
    # Categorical Standardization
    df["work_type_clean"] = df["formatted_work_type"].fillna("Other").astype(str)
    
    exp_map = {
        "Internship": "Entry",
        "Entry level": "Entry",
        "Associate": "Associate",
        "Mid-Senior level": "Mid-Senior",
        "Director": "Executive",
        "Executive": "Executive"
    }
    df["experience_group_clean"] = df["formatted_experience_level"].map(exp_map).fillna("Unknown")
    
    def get_band(s):
        if pd.isna(s) or s <= 0: return "Unknown"
        if s < 50000: return "<50k"
        if s < 100000: return "50k-100k"
        if s < 150000: return "100k-150k"
        return "150k+"
    df["salary_band_clean"] = df["normalized_salary"].apply(get_band)
    
    df["hot_job_prob"] = df["job_id"].map(hot_prob_map)
    
    return df

# ════════════════════════════════════════════════════════════
# 2. REFERENTIAL INTEGRITY FIXES
# ════════════════════════════════════════════════════════════
def apply_integrity_fixes(df):
    print("\n--- Applying Data Integrity Fixes ---")
    
    # [1] Fact Table: Fill NULL IDs with -1
    # We must ensure company_id and primary_industry_id don't have NaNs
    df["company_id"] = df["company_id"].fillna(-1).astype(int)
    df["primary_industry_id"] = df["primary_industry_id"].fillna(-1).astype(int)
    print("✓ Handled NULL Foreign Keys in Fact table (replaced with -1)")
    
    # [2] Dimension Company (cleanup and append Unknown)
    try:
        df_comp = pd.read_csv(RAW_DIR / "companies.csv")
        df_comp = df_comp.drop_duplicates(subset=["company_id"])
        
        # Add 'Unknown' company
        unknown_row = pd.DataFrame([{
            "company_id": -1, 
            "name": "(Unknown Company)",
            "description": "Information not available in LinkedIn dataset.",
            "company_size": 0,
            "url": "none"
        }])
        df_comp = pd.concat([df_comp, unknown_row], ignore_index=True)
        
        # Merge company size from master for consistency if needed
        sz_map = df.drop_duplicates("company_id").set_index("company_id")["company_size"].to_dict()
        df_comp["company_size_clean"] = df_comp["company_id"].map(sz_map).fillna(0)
        
        df_comp.to_csv(PBI_DIR / "dim_company.csv", index=False)
        print("✓ Created dim_company.csv (with ID: -1 row)")
    except Exception as e:
        print(f"  Warning loading companies: {e}")

    # [3] Dimension Industry (append Unknown)
    try:
        df_ind = pd.read_csv(RAW_DIR / "industries.csv")
        unknown_ind = pd.DataFrame([{
            "industry_id": -1, 
            "industry_name": "(Unknown Industry)"
        }])
        df_ind = pd.concat([df_ind, unknown_ind], ignore_index=True)
        df_ind.to_csv(PBI_DIR / "dim_industry.csv", index=False)
        print("✓ Created dim_industry.csv (with ID: -1 row)")
    except:
        print("  Warning loading industries.")

    # [4] Dimension Calendar (COMPLETE daily range)
    dates = pd.to_datetime(df["listed_date"]).dropna()
    min_date, max_date = dates.min(), dates.max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    df_cal = pd.DataFrame({"date": full_date_range})
    df_cal["year"] = df_cal["date"].dt.year
    df_cal["month"] = df_cal["date"].dt.month
    df_cal["month_name"] = df_cal["date"].dt.month_name()
    df_cal["day"] = df_cal["date"].dt.day
    df_cal["day_name"] = df_cal["date"].dt.day_name()
    df_cal.to_csv(PBI_DIR / "dim_calendar.csv", index=False)
    print(f"✓ Created dim_calendar.csv (Continuous from {min_date.date()} to {max_date.date()})")

    # [5] Junction Tables (Filter to only existing Fact IDs)
    valid_jobs = set(df["job_id"])
    
    df_skill_map = pd.read_csv(RAW_DIR / "job_skills.csv")
    df_skill_map = df_skill_map[df_skill_map["job_id"].isin(valid_jobs)]
    df_skill_map.to_csv(PBI_DIR / "fact_job_skill.csv", index=False)
    
    df_ben = pd.read_csv(RAW_DIR / "benefits.csv")
    df_ben = df_ben[df_ben["job_id"].isin(valid_jobs)]
    df_ben[["job_id", "type"]].to_csv(PBI_DIR / "fact_job_benefit.csv", index=False)
    print("✓ Scrubbed Junction tables to only include jobs present in Fact table.")
    
    return df

# ════════════════════════════════════════════════════════════
# 3. EXPORT FACT
# ════════════════════════════════════════════════════════════
def export_final_fact(df):
    fact_cols = [
        "job_id", "listed_date", "company_id", "primary_industry_id",
        "normalized_salary", "has_salary", "salary_disclosure_status",
        "skill_count", "benefit_count", "views", "applies", "applies_per_view",
        "engagement_score", "work_type_clean", "experience_group_clean",
        "is_remote", "hot_job_prob", "salary_band_clean"
    ]
    df_summary = df[fact_cols].copy()
    df_summary.to_csv(PBI_DIR / "summary_for_dashboard.csv", index=False)
    print(f"✓ Saved summary_for_dashboard.csv ({len(df_summary):,} rows)")
    
    # Export dim_skill (constant)
    df_dim_skill = pd.read_csv(RAW_DIR / "skills.csv")
    df_dim_skill.to_csv(PBI_DIR / "dim_skill.csv", index=False)

if __name__ == "__main__":
    df_final = load_and_enrich()
    df_fixed = apply_integrity_fixes(df_final)
    export_final_fact(df_fixed)
    print("\n✓ DATA INTEGRITY REPAIR COMPLETE: Repository is Power BI Ready.")
