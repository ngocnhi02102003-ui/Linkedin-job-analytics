"""
etl_create_powerbi_files.py
Phase 1b: ETL — Create Power BI Star Schema.
Verbose output showing every export step.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import config

# ════════════════════════════════════════════════════════════
# STEP 1: LOAD MASTER & VALIDATE
# ════════════════════════════════════════════════════════════
def load_and_validate():
    print("\n" + "="*60)
    print("STEP 1: LOAD MASTER DATASET")
    print("="*60)
    
    df = pd.read_csv(config.DATA_PROCESSED_DIR / "jobs_master.csv")
    print(f"  Loaded jobs_master.csv: {len(df):,} rows x {len(df.columns)} cols")
    
    # Quick validation
    null_company = df["company_id"].isna().sum()
    null_industry = df["primary_industry_id"].isna().sum()
    print(f"  NULL company_id: {null_company}")
    print(f"  NULL primary_industry_id: {null_industry}")
    
    return df

# ════════════════════════════════════════════════════════════
# STEP 2: EXPORT FACT TABLE
# ════════════════════════════════════════════════════════════
def export_fact_table(df):
    print("\n" + "="*60)
    print("STEP 2: EXPORT FACT TABLE (summary_for_dashboard)")
    print("="*60)
    
    fact_cols = [
        "job_id", "listed_date", "company_id", "primary_industry_id",
        "normalized_salary", "has_salary", "salary_disclosure_status",
        "skill_count", "benefit_count", "views", "applies", "applies_per_view",
        "engagement_score", "work_type_clean", "experience_group_clean",
        "is_remote", "is_sponsored", "is_hot_job", "salary_band_clean",
        "company_known_flag", "industry_known_flag"
    ]
    
    # Check which columns actually exist
    available = [c for c in fact_cols if c in df.columns]
    missing_cols = [c for c in fact_cols if c not in df.columns]
    if missing_cols:
        print(f"  ⚠ Missing columns (skipped): {missing_cols}")
    
    df_fact = df[available].copy()
    out_path = config.PBI_DIR / "summary_for_dashboard.csv"
    df_fact.to_csv(out_path, index=False)
    print(f"  ✓ Saved summary_for_dashboard.csv")
    print(f"    shape: {df_fact.shape}")
    print(f"    columns: {len(available)}")
    
    # Also save full master to powerbi dir
    df.to_csv(config.PBI_DIR / "jobs_master.csv", index=False)
    print(f"  ✓ Saved jobs_master.csv (full) to powerbi/")
    
    return set(df["job_id"])

# ════════════════════════════════════════════════════════════
# STEP 3: EXPORT DIMENSION TABLES
# ════════════════════════════════════════════════════════════
def export_dimensions(df):
    print("\n" + "="*60)
    print("STEP 3: EXPORT DIMENSION TABLES")
    print("="*60)
    
    # --- dim_company ---
    print("\n=== dim_company ===")
    df_comp = pd.read_csv(config.DATA_RAW_DIR / "companies.csv")
    n0 = len(df_comp)
    df_comp = df_comp.drop_duplicates("company_id")
    # Add Unknown row
    unknown = pd.DataFrame([{"company_id": -1, "name": "(Unknown Company)", "company_size": 0, "country": "Unknown"}])
    df_comp = pd.concat([df_comp, unknown], ignore_index=True)
    df_comp.to_csv(config.PBI_DIR / "dim_company.csv", index=False)
    print(f"  raw: {n0:,} -> dedup: {len(df_comp):,} (includes Unknown row)")
    print(f"  ✓ Saved dim_company.csv  shape={df_comp.shape}")
    
    # --- dim_industry ---
    print("\n=== dim_industry ===")
    df_ind = pd.read_csv(config.DATA_RAW_DIR / "industries.csv")
    n0 = len(df_ind)
    unknown_ind = pd.DataFrame([{"industry_id": -1, "industry_name": "(Unknown Industry)"}])
    df_ind = pd.concat([df_ind, unknown_ind], ignore_index=True)
    df_ind.to_csv(config.PBI_DIR / "dim_industry.csv", index=False)
    print(f"  raw: {n0:,} -> final: {len(df_ind):,} (includes Unknown row)")
    print(f"  ✓ Saved dim_industry.csv  shape={df_ind.shape}")
    
    # --- dim_skill ---
    print("\n=== dim_skill ===")
    df_skill = pd.read_csv(config.DATA_RAW_DIR / "skills.csv")
    df_skill.to_csv(config.PBI_DIR / "dim_skill.csv", index=False)
    print(f"  rows: {len(df_skill):,}")
    print(f"  ✓ Saved dim_skill.csv  shape={df_skill.shape}")
    
    # --- dim_calendar ---
    print("\n=== dim_calendar ===")
    dates = pd.to_datetime(df["listed_date"]).dropna()
    min_d, max_d = dates.min(), dates.max()
    full_range = pd.date_range(start=min_d, end=max_d, freq="D")
    df_cal = pd.DataFrame({"date": full_range})
    df_cal["year"] = df_cal["date"].dt.year
    df_cal["quarter"] = df_cal["date"].dt.quarter
    df_cal["month"] = df_cal["date"].dt.month
    df_cal["month_name"] = df_cal["date"].dt.month_name()
    df_cal["week"] = df_cal["date"].dt.isocalendar().week.astype(int)
    df_cal["day"] = df_cal["date"].dt.day
    df_cal["day_name"] = df_cal["date"].dt.day_name()
    df_cal["is_weekend"] = df_cal["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df_cal.to_csv(config.PBI_DIR / "dim_calendar.csv", index=False)
    print(f"  date range: {min_d.date()} to {max_d.date()} ({len(full_range)} days)")
    print(f"  ✓ Saved dim_calendar.csv  shape={df_cal.shape}")

# ════════════════════════════════════════════════════════════
# STEP 4: EXPORT JUNCTION / BRIDGE TABLES
# ════════════════════════════════════════════════════════════
def export_junctions(valid_jobs):
    print("\n" + "="*60)
    print("STEP 4: EXPORT JUNCTION TABLES (Bridge)")
    print("="*60)
    
    # --- fact_job_skill ---
    print("\n=== fact_job_skill ===")
    df_js = pd.read_csv(config.DATA_RAW_DIR / "job_skills.csv")
    n0 = len(df_js)
    df_js = df_js[df_js["job_id"].isin(valid_jobs)]
    df_js.to_csv(config.PBI_DIR / "fact_job_skill.csv", index=False)
    orphan = n0 - len(df_js)
    print(f"  raw: {n0:,} -> filtered: {len(df_js):,} (removed {orphan:,} orphan rows)")
    print(f"  ✓ Saved fact_job_skill.csv  shape={df_js.shape}")
    
    # --- fact_job_benefit ---
    print("\n=== fact_job_benefit ===")
    df_ben = pd.read_csv(config.DATA_RAW_DIR / "benefits.csv")
    n0 = len(df_ben)
    df_ben = df_ben[df_ben["job_id"].isin(valid_jobs)]
    df_ben = df_ben[["job_id", "type"]]
    orphan = n0 - len(df_ben)
    df_ben.to_csv(config.PBI_DIR / "fact_job_benefit.csv", index=False)
    print(f"  raw: {n0:,} -> filtered: {len(df_ben):,} (removed {orphan:,} orphan rows)")
    print(f"  ✓ Saved fact_job_benefit.csv  shape={df_ben.shape}")

# ════════════════════════════════════════════════════════════
# STEP 5: INTEGRITY VALIDATION
# ════════════════════════════════════════════════════════════
def validate_outputs(valid_jobs):
    print("\n" + "="*60)
    print("STEP 5: POST-EXPORT INTEGRITY VALIDATION")
    print("="*60)
    
    pbi = config.PBI_DIR
    fact = pd.read_csv(pbi / "summary_for_dashboard.csv")
    skill = pd.read_csv(pbi / "fact_job_skill.csv")
    benefit = pd.read_csv(pbi / "fact_job_benefit.csv")
    cal = pd.read_csv(pbi / "dim_calendar.csv")
    
    fact_ids = set(fact["job_id"])
    
    skill_orphan = skill[~skill["job_id"].isin(fact_ids)]
    benefit_orphan = benefit[~benefit["job_id"].isin(fact_ids)]
    null_company = fact["company_id"].isna().sum()
    null_industry = fact["primary_industry_id"].isna().sum()
    
    print(f"  NULL company_id in fact:       {null_company}")
    print(f"  NULL industry_id in fact:      {null_industry}")
    print(f"  Orphan rows in fact_job_skill:  {len(skill_orphan)}")
    print(f"  Orphan rows in fact_job_benefit:{len(benefit_orphan)}")
    print(f"  Calendar continuous days:       {len(cal)}")
    
    if null_company == 0 and null_industry == 0 and len(skill_orphan) == 0 and len(benefit_orphan) == 0:
        print("  ✓ ALL CHECKS PASSED — Data is Power BI ready.")
    else:
        print("  ⚠ SOME CHECKS FAILED — Review above.")

# ════════════════════════════════════════════════════════════
# STEP 6: CREATE DATA DICTIONARY
# ════════════════════════════════════════════════════════════
def create_data_dictionary():
    print("\n" + "="*60)
    print("STEP 6: CREATE DATA DICTIONARY")
    print("="*60)
    
    content = """# Data Dictionary: LinkedIn Jobs Project

## Fact Table: `summary_for_dashboard.csv`

| Column | Type | Description |
|---|---|---|
| `job_id` | int | Unique job posting identifier |
| `listed_date` | date | Date job was posted |
| `company_id` | int | FK to dim_company (-1 = Unknown) |
| `primary_industry_id` | int | FK to dim_industry (-1 = Unknown) |
| `normalized_salary` | float | Annual salary in USD (NaN = not disclosed) |
| `has_salary` | int | 1 = salary disclosed, 0 = hidden |
| `salary_disclosure_status` | str | "Disclosed" or "Hidden" |
| `skill_count` | int | Number of skills required |
| `benefit_count` | int | Number of benefits listed |
| `views` | float | Total views on posting |
| `applies` | float | Total applications received |
| `applies_per_view` | float | Conversion rate (applies/views) |
| `engagement_score` | float | Weighted: views + 10*applies |
| `work_type_clean` | str | Standardized work type |
| `experience_group_clean` | str | Entry/Associate/Mid-Senior/Executive/Unknown |
| `is_remote` | int | 1 = remote allowed |
| `is_sponsored` | int | 1 = sponsored posting |
| `is_hot_job` | int | 1 = views >= 75th percentile |
| `salary_band_clean` | str | Low/Mid/High/Very High/Unknown |
| `company_known_flag` | int | 1 = company info available |
| `industry_known_flag` | int | 1 = industry info available |

## Dimension Tables

| Table | Key | Description |
|---|---|---|
| `dim_company` | company_id | Company name, size, country |
| `dim_industry` | industry_id | Industry name |
| `dim_skill` | skill_abr | Skill abbreviation and name |
| `dim_calendar` | date | Full date table for Time Intelligence |

## Bridge Tables

| Table | Keys | Description |
|---|---|---|
| `fact_job_skill` | job_id, skill_abr | Many-to-many: jobs ↔ skills |
| `fact_job_benefit` | job_id, type | Many-to-many: jobs ↔ benefits |
"""
    path = config.REPORTS_DIR / "data_dictionary.md"
    with open(path, "w") as f:
        f.write(content)
    print(f"  ✓ Saved {path}")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ETL PIPELINE: LinkedIn Jobs — Power BI Export      ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    try:
        df = load_and_validate()
        valid_jobs = export_fact_table(df)
        export_dimensions(df)
        export_junctions(valid_jobs)
        validate_outputs(valid_jobs)
        create_data_dictionary()
        
        # Summary
        print("\n" + "="*60)
        print("EXPORT SUMMARY")
        print("="*60)
        import os
        for f in sorted(config.PBI_DIR.iterdir()):
            size_kb = os.path.getsize(f) / 1024
            print(f"  {f.name:<35} {size_kb:>10,.1f} KB")
        
        print("\n" + "="*60)
        print("✓ SUCCESS: Power BI Export Complete.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
