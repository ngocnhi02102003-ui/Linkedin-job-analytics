"""
etl_build_master.py
Phase 1: ETL — Building the Master Dataset.
Verbose output showing every step: Audit → Clean → Merge → Feature Engineering.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import config

# ════════════════════════════════════════════════════════════
# HELPER: Verbose logger
# ════════════════════════════════════════════════════════════
def log_step(name, before, after, note=""):
    extra = f"  {note}" if note else ""
    print(f"  [{name}] {before} -> {after} rows{extra}")

# ════════════════════════════════════════════════════════════
# STEP 1: LOAD RAW DATA
# ════════════════════════════════════════════════════════════
def load_all_raw():
    """Đọc tất cả file nguồn từ data_raw/"""
    print("\n" + "="*60)
    print("STEP 1: LOAD RAW DATA")
    print("="*60)
    
    p = config.DATA_RAW_DIR
    tables = {
        "postings":             pd.read_csv(p / "postings.csv"),
        "companies":            pd.read_csv(p / "companies.csv"),
        "industries":           pd.read_csv(p / "industries.csv"),
        "job_industries":       pd.read_csv(p / "job_industries.csv"),
        "job_skills":           pd.read_csv(p / "job_skills.csv"),
        "benefits":             pd.read_csv(p / "benefits.csv"),
        "salaries":             pd.read_csv(p / "salaries.csv"),
        "skills":               pd.read_csv(p / "skills.csv"),
        "employee_counts":      pd.read_csv(p / "employee_counts.csv"),
        "company_industries":   pd.read_csv(p / "company_industries.csv"),
        "company_specialities": pd.read_csv(p / "company_specialities.csv"),
    }
    
    print(f"\n{'table_name':<25} {'rows':>10} {'columns':>10}")
    print("-" * 50)
    for name, df in tables.items():
        print(f"  {name:<23} {len(df):>10,} {len(df.columns):>10}")
    
    return tables

# ════════════════════════════════════════════════════════════
# STEP 2: DATA QUALITY CHECK (AUDIT)
# ════════════════════════════════════════════════════════════
def run_data_audit(tables):
    """Kiểm tra duplicates, referential integrity, missing values"""
    print("\n" + "="*60)
    print("STEP 2: DATA QUALITY CHECK")
    print("="*60)
    
    # --- 2A: KEY DUPLICATES REPORT ---
    print("\n[KEY DUPLICATES REPORT]")
    key_map = {
        "postings":        ["job_id"],
        "companies":       ["company_id"],
        "employee_counts": ["company_id"],
        "job_skills":      ["job_id", "skill_abr"],
        "job_industries":  ["job_id", "industry_id"],
        "salaries":        ["job_id"],
        "industries":      ["industry_id"],
        "skills":          ["skill_abr"],
        "benefits":        ["job_id", "type"],
    }
    
    print(f"  {'table_name':<20} {'key_columns':<25} {'total_rows':>12} {'key_duplicates':>16} {'duplicate_pct':>14}")
    for tbl, keys in key_map.items():
        if tbl not in tables:
            continue
        df = tables[tbl]
        n = len(df)
        dups = df.duplicated(subset=keys).sum()
        pct = dups / n if n > 0 else 0
        print(f"  {tbl:<20} {', '.join(keys):<25} {n:>12,} {dups:>16,} {pct:>14.6f}")
    
    # --- 2B: REFERENTIAL INTEGRITY REPORT ---
    print("\n[REFERENTIAL INTEGRITY REPORT]")
    fk_checks = [
        ("postings",       "company_id",  "companies",  "company_id"),
        ("job_skills",     "job_id",      "postings",   "job_id"),
        ("job_industries", "job_id",      "postings",   "job_id"),
        ("salaries",       "job_id",      "postings",   "job_id"),
        ("benefits",       "job_id",      "postings",   "job_id"),
    ]
    
    print(f"  {'child_table':<18} {'child_key':<14} {'parent_table':<14} {'parent_key':<12} {'invalid_count':>15} {'invalid_pct':>12}")
    for child_tbl, child_key, parent_tbl, parent_key in fk_checks:
        if child_tbl not in tables or parent_tbl not in tables:
            continue
        child_df = tables[child_tbl]
        parent_df = tables[parent_tbl]
        invalid = child_df[~child_df[child_key].isin(parent_df[parent_key])]
        n = len(child_df)
        pct = len(invalid) / n if n > 0 else 0
        print(f"  {child_tbl:<18} {child_key:<14} {parent_tbl:<14} {parent_key:<12} {len(invalid):>15,} {pct:>12.6f}")
    
    # --- 2C: MISSING DATA SUMMARY BY TABLE ---
    print("\n[MISSING DATA SUMMARY BY TABLE]")
    print(f"  {'table_name':<25} {'total_rows':>12} {'avg_missing_pct':>16} {'cols_with_missing':>18}")
    for name, df in tables.items():
        n = len(df)
        missing_pcts = df.isnull().mean()
        avg_miss = missing_pcts.mean()
        cols_miss = (missing_pcts > 0).sum()
        print(f"  {name:<25} {n:>12,} {avg_miss:>16.6f} {cols_miss:>18}")

# ════════════════════════════════════════════════════════════
# STEP 3: CLEAN EACH TABLE
# ════════════════════════════════════════════════════════════
def clean_tables(tables):
    """Làm sạch từng bảng riêng lẻ và lưu vào data_interim/"""
    print("\n" + "="*60)
    print("STEP 3: CLEAN INDIVIDUAL TABLES")
    print("="*60)
    
    interim = config.DATA_INTERIM_DIR
    
    # --- 3A: CLEAN POSTINGS ---
    print("\n=== CLEAN POSTINGS ===")
    df_post = tables["postings"].copy()
    n0 = len(df_post)
    df_post = df_post.drop_duplicates(subset=["job_id"])
    log_step("drop_dup_job_id", n0, len(df_post), "duplicate job_id removed")
    
    # Date standardization
    df_post["listed_date"] = pd.to_datetime(df_post["listed_time"], unit="ms").dt.date
    log_step("postings_final", n0, len(df_post))
    df_post.to_csv(interim / "postings_clean.csv", index=False)
    print(f"  Saved postings_clean.csv  shape={df_post.shape}")
    
    # --- 3B: CLEAN SALARIES ---
    print("\n=== CLEAN SALARIES ===")
    df_sal = tables["salaries"].copy()
    n0 = len(df_sal)
    df_sal = df_sal.drop_duplicates(subset=["job_id"])
    log_step("salaries_final", n0, len(df_sal))
    df_sal.to_csv(interim / "salaries_clean.csv", index=False)
    print(f"  Saved salaries_clean.csv  shape={df_sal.shape}")
    
    # --- 3C: CLEAN BENEFITS ---
    print("\n=== CLEAN BENEFITS ===")
    df_ben = tables["benefits"].copy()
    n0 = len(df_ben)
    df_ben_agg = df_ben.groupby("job_id").agg(
        benefit_count=("type", "count"),
        benefit_types=("type", lambda x: ", ".join(sorted(x.unique())))
    ).reset_index()
    log_step("benefits_agg", n0, len(df_ben_agg))
    df_ben_agg.to_csv(interim / "benefits_agg.csv", index=False)
    print(f"  Saved benefits_agg.csv  shape={df_ben_agg.shape}")
    
    # --- 3D: CLEAN JOB_SKILLS ---
    print("\n=== CLEAN JOB_SKILLS ===")
    df_js = tables["job_skills"].copy()
    n0 = len(df_js)
    df_js_agg = df_js.groupby("job_id").agg(
        skill_count=("skill_abr", "count"),
        skill_list=("skill_abr", lambda x: ", ".join(sorted(x.unique())))
    ).reset_index()
    log_step("job_skills_agg", n0, len(df_js_agg))
    df_js_agg.to_csv(interim / "job_skills_agg.csv", index=False)
    print(f"  Saved job_skills_agg.csv  shape={df_js_agg.shape}")
    
    # --- 3E: CLEAN JOB_INDUSTRIES ---
    print("\n=== CLEAN JOB_INDUSTRIES ===")
    df_ji = tables["job_industries"].copy()
    n0 = len(df_ji)
    # Keep first industry per job as primary
    df_ji_primary = df_ji.drop_duplicates(subset=["job_id"], keep="first")
    # Merge industry name
    df_ji_primary = df_ji_primary.merge(
        tables["industries"][["industry_id", "industry_name"]], 
        on="industry_id", how="left"
    )
    log_step("job_industries_primary", n0, len(df_ji_primary))
    df_ji_primary.to_csv(interim / "job_industries_primary.csv", index=False)
    print(f"  Saved job_industries_primary.csv  shape={df_ji_primary.shape}")
    
    # --- 3F: CLEAN COMPANIES ---
    print("\n=== CLEAN COMPANIES ===")
    df_comp = tables["companies"].copy()
    n0 = len(df_comp)
    df_comp = df_comp.drop_duplicates(subset=["company_id"])
    log_step("companies_final", n0, len(df_comp))
    df_comp.to_csv(interim / "companies_clean.csv", index=False)
    print(f"  Saved companies_clean.csv  shape={df_comp.shape}")
    
    # --- 3G: CLEAN EMPLOYEE_COUNTS ---
    print("\n=== CLEAN EMPLOYEE_COUNTS ===")
    df_ec = tables["employee_counts"].copy()
    n0 = len(df_ec)
    # Keep latest record per company
    if "time_recorded" in df_ec.columns:
        df_ec = df_ec.sort_values("time_recorded", ascending=False)
    df_ec = df_ec.drop_duplicates(subset=["company_id"], keep="first")
    log_step("employee_counts_latest", n0, len(df_ec))
    df_ec.to_csv(interim / "employee_counts_latest.csv", index=False)
    print(f"  Saved employee_counts_latest.csv  shape={df_ec.shape}")
    
    return {
        "postings": df_post,
        "salaries": df_sal,
        "benefits_agg": df_ben_agg,
        "skills_agg": df_js_agg,
        "industries_primary": df_ji_primary,
        "companies": df_comp,
        "employee_counts": df_ec,
    }

# ════════════════════════════════════════════════════════════
# STEP 4: JOIN / MERGE INTO MASTER
# ════════════════════════════════════════════════════════════
def build_master(cleaned):
    """Merge tất cả bảng sạch thành jobs_master"""
    print("\n" + "="*60)
    print("STEP 4: BUILD JOBS_MASTER (Merge)")
    print("="*60)
    
    df = cleaned["postings"].copy()
    n = len(df)
    
    # --- Salary normalization ---
    def annualize(val, period):
        if pd.isna(val) or pd.isna(period): return np.nan
        period = str(period).upper()
        if period == "YEARLY": return val
        if period == "MONTHLY": return val * 12
        if period == "WEEKLY": return val * 52
        if period == "HOURLY": return val * 2080
        return val
    
    if "normalized_salary" not in df.columns:
        df["normalized_salary"] = np.nan
    df["salary_raw"] = df["med_salary"].fillna((df["min_salary"] + df["max_salary"]) / 2)
    df["calc_salary"] = df.apply(lambda r: annualize(r["salary_raw"], r["pay_period"]), axis=1)
    df["normalized_salary"] = df["normalized_salary"].fillna(df["calc_salary"])
    
    # --- Join Benefits ---
    df = df.merge(cleaned["benefits_agg"][["job_id", "benefit_count"]], on="job_id", how="left")
    df["benefit_count"] = df["benefit_count"].fillna(0).astype(int)
    log_step("join_benefits", n, len(df))
    
    # --- Join Skills ---
    df = df.merge(cleaned["skills_agg"][["job_id", "skill_count"]], on="job_id", how="left")
    df["skill_count"] = df["skill_count"].fillna(0).astype(int)
    log_step("join_skills", n, len(df))
    
    # --- Join Industry ---
    df = df.merge(
        cleaned["industries_primary"][["job_id", "industry_id", "industry_name"]], 
        on="job_id", how="left"
    )
    df.rename(columns={"industry_id": "primary_industry_id", "industry_name": "primary_industry_name"}, inplace=True)
    log_step("join_industry", n, len(df))
    
    # --- Join Company info ---
    comp_cols = ["company_id"]
    if "follower_count" in cleaned["companies"].columns:
        comp_cols.append("follower_count")
    df = df.merge(cleaned["companies"][comp_cols].drop_duplicates("company_id"), on="company_id", how="left", suffixes=("", "_comp"))
    log_step("join_company", n, len(df))
    
    # --- Join Employee Counts ---
    ec_cols = ["company_id", "employee_count"] if "employee_count" in cleaned["employee_counts"].columns else ["company_id"]
    df = df.merge(cleaned["employee_counts"][ec_cols].drop_duplicates("company_id"), on="company_id", how="left", suffixes=("", "_ec"))
    log_step("join_employee_counts", n, len(df))
    
    # --- Dedup final ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["job_id"], keep="first")
    log_step("dedup_final", before_dedup, len(df), "ensure 1-row-per-job_id")
    
    return df

# ════════════════════════════════════════════════════════════
# STEP 5: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
def add_features(df):
    """Tạo các cột phân tích mới"""
    print("\n" + "="*60)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*60)
    
    # 1. Salary features
    df["has_salary"] = (df["normalized_salary"].notna() & (df["normalized_salary"] > 0)).astype(int)
    df["salary_disclosure_status"] = df["has_salary"].map({1: "Disclosed", 0: "Hidden"})
    print(f"  [has_salary] Disclosed: {df['has_salary'].sum():,} | Hidden: {(df['has_salary']==0).sum():,}")
    
    # 2. Performance metrics
    df["applies_per_view"] = (df["applies"] / df["views"].replace(0, np.nan)).fillna(0)
    df["engagement_score"] = df["views"].fillna(0) + (df["applies"].fillna(0) * 10)
    print(f"  [engagement_score] mean={df['engagement_score'].mean():.1f} | median={df['engagement_score'].median():.1f}")
    
    # 3. NULL handling for IDs
    df["company_id"] = df["company_id"].fillna(-1).astype(int)
    df["primary_industry_id"] = df["primary_industry_id"].fillna(-1).astype(int)
    df["company_known_flag"] = (df["company_id"] != -1).astype(int)
    df["industry_known_flag"] = (df["primary_industry_id"] != -1).astype(int)
    print(f"  [company_known_flag] Known: {df['company_known_flag'].sum():,} | Unknown: {(df['company_known_flag']==0).sum():,}")
    print(f"  [industry_known_flag] Known: {df['industry_known_flag'].sum():,} | Unknown: {(df['industry_known_flag']==0).sum():,}")
    
    # 4. Categorical cleaning
    df["work_type_clean"] = df["formatted_work_type"].fillna("Other")
    exp_map = {
        "Internship": "Entry", "Entry level": "Entry",
        "Associate": "Associate",
        "Mid-Senior level": "Mid-Senior",
        "Director": "Executive", "Executive": "Executive"
    }
    df["experience_group_clean"] = df["formatted_experience_level"].map(exp_map).fillna("Unknown")
    print(f"  [experience_group_clean] distribution:")
    for grp, cnt in df["experience_group_clean"].value_counts().items():
        print(f"    {grp:<15} {cnt:>8,}")
    
    # 5. Salary bands
    def get_band(s):
        if pd.isna(s) or s <= 0: return "Unknown"
        if s < 40000: return "Low (<40k)"
        if s < 80000: return "Mid (40-80k)"
        if s < 150000: return "High (80-150k)"
        return "Very High (>150k)"
    df["salary_band_clean"] = df["normalized_salary"].apply(get_band)
    print(f"  [salary_band_clean] distribution:")
    for band, cnt in df["salary_band_clean"].value_counts().items():
        print(f"    {band:<20} {cnt:>8,}")
    
    # 6. Hot job flag
    view_threshold = df["views"].quantile(0.75)
    df["is_hot_job"] = (df["views"] >= view_threshold).astype(int)
    print(f"  [is_hot_job] threshold={view_threshold:.0f} views | hot_jobs={df['is_hot_job'].sum():,}")
    
    # 7. Remote / Sponsored
    df["is_remote"] = df["remote_allowed"].fillna(0).astype(int)
    df["is_sponsored"] = df["sponsored"].fillna(0).astype(int)
    print(f"  [is_remote] Remote: {df['is_remote'].sum():,} | On-site: {(df['is_remote']==0).sum():,}")
    print(f"  [is_sponsored] Sponsored: {df['is_sponsored'].sum():,} | Organic: {(df['is_sponsored']==0).sum():,}")
    
    # 8. Description length
    df["description_length"] = df["description"].fillna("").str.len()
    print(f"  [description_length] mean={df['description_length'].mean():.0f} chars")
    
    return df

# ════════════════════════════════════════════════════════════
# STEP 6: SAVE MASTER + REPORT
# ════════════════════════════════════════════════════════════
def save_master_and_report(df):
    """Lưu jobs_master.csv và tạo ETL pipeline report"""
    print("\n" + "="*60)
    print("STEP 6: SAVE MASTER DATASET")
    print("="*60)
    
    out_path = config.DATA_PROCESSED_DIR / "jobs_master.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    # ETL Pipeline Report
    report_path = config.REPORTS_DIR / "etl_pipeline_report.md"
    with open(report_path, "w") as f:
        f.write("# ETL Pipeline Report\n\n")
        f.write(f"- **Total Rows**: {len(df):,}\n")
        f.write(f"- **Total Columns**: {len(df.columns)}\n")
        f.write(f"- **Salary Disclosed**: {df['has_salary'].sum():,} ({df['has_salary'].mean()*100:.1f}%)\n")
        f.write(f"- **Salary Hidden**: {(df['has_salary']==0).sum():,} ({(df['has_salary']==0).mean()*100:.1f}%)\n")
        f.write(f"- **Remote Jobs**: {df['is_remote'].sum():,} ({df['is_remote'].mean()*100:.1f}%)\n")
        f.write(f"- **Hot Jobs**: {df['is_hot_job'].sum():,}\n")
        f.write(f"- **Companies Known**: {df['company_known_flag'].sum():,}\n")
        f.write(f"- **Industries Known**: {df['industry_known_flag'].sum():,}\n")
        f.write(f"- **Date Range**: {df['listed_date'].min()} to {df['listed_date'].max()}\n")
        f.write(f"\n## Feature List\n\n")
        for col in sorted(df.columns):
            f.write(f"- `{col}` ({df[col].dtype})\n")
    print(f"  Saved {report_path}")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ETL PIPELINE: LinkedIn Jobs — Build Master         ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    try:
        # Step 1: Load
        tables = load_all_raw()
        
        # Step 2: Audit
        run_data_audit(tables)
        
        # Step 3: Clean
        cleaned = clean_tables(tables)
        
        # Step 4: Merge
        df_master = build_master(cleaned)
        
        # Step 5: Feature Engineering
        df_master = add_features(df_master)
        
        # Step 6: Save
        save_master_and_report(df_master)
        
        print("\n" + "="*60)
        print("✓ SUCCESS: ETL Phase 1 Complete.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR in ETL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
