"""
08_export_powerbi.py
Export cleaned and enriched datasets into Power BI Star Schema.
Creates Fact and Dimension tables in data_processed/powerbi/.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import config

# ── Paths ─────────────────────────────────────────────────────
PRO_DIR    = config.DATA_PROCESSED_DIR
RAW_DIR    = config.DATA_RAW_DIR
PBI_DIR    = PRO_DIR / "powerbi"
PBI_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════
print("Loading datasets …")
df_master = pd.read_csv(PRO_DIR / "jobs_master.csv", low_memory=False)

# Predictions (if available)
try:
    df_sal_p = pd.read_csv(PRO_DIR / "salary_predictions.csv")
    sal_map = df_sal_p.set_index("job_id")["pred_salary_best"].to_dict()
except:
    print("Warning: salary_predictions.csv not found.")
    sal_map = {}

try:
    df_hot_p = pd.read_csv(PRO_DIR / "hot_job_predictions.csv")
    hot_map = df_hot_p.set_index("job_id")["hot_job_prob"].to_dict()
except:
    print("Warning: hot_job_predictions.csv not found.")
    hot_map = {}

# ════════════════════════════════════════════════════════════
# 2. FACT: DASHBOARD JOBS
# ════════════════════════════════════════════════════════════
print("Building fact_job_dashboard (Main Fact) …")
# Merge predictions
df_master["pred_salary"]   = df_master["job_id"].map(sal_map)
df_master["hot_job_prob"]  = df_master["job_id"].map(hot_map)

# Select core columns for the main fact table (keep it lean for PBI)
fact_cols = [
    "job_id", "company_id", "title", "location", "listed_date", "listed_month",
    "normalized_salary", "pred_salary", "hot_job_prob", "formatted_work_type",
    "formatted_experience_level", "is_remote", "is_sponsored", "views", "applies",
    "description_length", "skill_count", "benefit_count", "salary_band",
    "primary_industry_id"
]
# Ensure only existing columns are used
fact_cols = [col for col in fact_cols if col in df_master.columns]
df_fact_jobs = df_master[fact_cols].copy()

# ════════════════════════════════════════════════════════════
# 3. DIMENSION: COMPANY
# ════════════════════════════════════════════════════════════
print("Building dim_company …")
# Companies are in jobs_master, but let's get the latest metadata from companies.csv if possible
try:
    df_comp_ref = pd.read_csv(RAW_DIR / "companies.csv")
    # Join with jobs_master to get company attributes used in analysis
    comp_ids = df_master["company_id"].unique()
    df_dim_comp = df_comp_ref[df_comp_ref["company_id"].isin(comp_ids)].copy()
except:
    print("Warning: companies.csv not found. Re-extracting from master.")
    df_dim_comp = df_master[["company_id", "company_name", "company_size", "url", "employee_count", "follower_count"]].drop_duplicates(subset=["company_id"])

# ════════════════════════════════════════════════════════════
# 4. DIMENSIONS: SKILLS & INDUSTRIES
# ════════════════════════════════════════════════════════════
print("Building dimensions (Skills, Industries) …")
df_dim_skills = pd.read_csv(RAW_DIR / "skills.csv")
df_dim_industries = pd.read_csv(RAW_DIR / "industries.csv")

# ════════════════════════════════════════════════════════════
# 5. MAPPING FACTS (Multiple Skills / Benefits per Job)
# ════════════════════════════════════════════════════════════
print("Building mapping facts (Skill, Benefit) …")
df_fact_job_skill = pd.read_csv(RAW_DIR / "job_skills.csv")

# Benefits cleaning
df_ben = pd.read_csv(RAW_DIR / "benefits.csv")
df_fact_job_benefit = df_ben[["job_id", "type"]].copy()

# ════════════════════════════════════════════════════════════
# 6. EXPORT
# ════════════════════════════════════════════════════════════
def export_csv(df, name):
    df.to_csv(PBI_DIR / name, index=False)
    print(f"✓ Saved {name} (Rows: {len(df):,})")

print("\n--- Exporting Files ---")
export_csv(df_fact_jobs,      "dashboard_jobs.csv")
export_csv(df_dim_comp,       "dim_company.csv")
export_csv(df_dim_skills,     "dim_skill.csv")
export_csv(df_dim_industries, "dim_industry.csv")
export_csv(df_fact_job_skill, "fact_job_skill.csv")
export_csv(df_fact_job_benefit,"fact_job_benefit.csv")

# Optional: Simple Calendar Dimension
print("\nBuilding dim_calendar …")
dates = pd.to_datetime(df_master["listed_date"]).dropna().unique()
if len(dates) > 0:
    df_cal = pd.DataFrame({"date": sorted(dates)})
    df_cal["year"]     = df_cal["date"].dt.year
    df_cal["month"]    = df_cal["date"].dt.month
    df_cal["month_name"] = df_cal["date"].dt.month_name()
    df_cal["day"]      = df_cal["date"].dt.day
    df_cal["weekday"]  = df_cal["date"].dt.day_name()
    export_csv(df_cal, "dim_calendar.csv")

print("\nDone! All files ready for Power BI Import.")
