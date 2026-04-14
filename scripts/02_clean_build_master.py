"""
02_clean_build_master.py
Clean all main tables and build one master table (jobs_master) with one row per job_id.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config

import pandas as pd
import numpy as np

RAW = config.DATA_RAW_DIR
INT = config.DATA_INTERIM_DIR
PRO = config.DATA_PROCESSED_DIR
TBL = config.TABLES_DIR

build_report = []   # list of dicts summarising each step


def log(step, rows_in, rows_out, msg=""):
    print(f"  [{step}] {rows_in} -> {rows_out} rows  {msg}")
    build_report.append({"step": step, "rows_in": rows_in, "rows_out": rows_out, "note": msg})


# ─────────────────────────────────────────────
# 1. CLEAN POSTINGS
# ─────────────────────────────────────────────
def clean_postings():
    print("\n=== CLEAN POSTINGS ===")
    df = pd.read_csv(RAW / "postings.csv", low_memory=False)
    n0 = len(df)

    # Drop duplicate job_id (keep first)
    df = df.drop_duplicates(subset="job_id", keep="first")
    log("drop_dup_job_id", n0, len(df), "duplicate job_id removed")

    # Normalize text columns
    text_cols = ["title", "company_name", "location", "formatted_work_type",
                 "formatted_experience_level", "application_type", "pay_period",
                 "work_type", "currency", "compensation_type"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace("Nan", pd.NA)

    # Numeric: views, applies
    for col in ["views", "applies"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamps (seconds since epoch → datetime)
    ts_cols = ["listed_time", "original_listed_time", "expiry", "closed_time"]
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")

    # Derived columns
    if "listed_time" in df.columns:
        df["listed_date"]  = df["listed_time"].dt.date
        df["listed_month"] = df["listed_time"].dt.to_period("M").astype(str)

    if "description" in df.columns:
        df["description_length"] = df["description"].astype(str).str.len()

    if "remote_allowed" in df.columns:
        df["is_remote"] = df["remote_allowed"].fillna(0).astype(int).astype(bool)
    else:
        df["is_remote"] = False

    if "sponsored" in df.columns:
        df["is_sponsored"] = df["sponsored"].fillna(0).astype(int).astype(bool)
    else:
        df["is_sponsored"] = False

    log("postings_final", n0, len(df))
    df.to_csv(INT / "postings_clean.csv", index=False)
    print(f"  Saved postings_clean.csv  shape={df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. CLEAN SALARIES
# ─────────────────────────────────────────────
def clean_salaries():
    print("\n=== CLEAN SALARIES ===")
    df = pd.read_csv(RAW / "salaries.csv", low_memory=False)
    n0 = len(df)

    # Numeric salary columns
    for col in ["min_salary", "max_salary", "med_salary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize pay_period: keep only standard ones
    if "pay_period" in df.columns:
        df["pay_period"] = df["pay_period"].astype(str).str.upper().str.strip()
        multiplier = {
            "YEARLY": 1, "ANNUAL": 1, "ANNUALLY": 1,
            "MONTHLY": 12, "WEEKLY": 52, "HOURLY": 2080,
            "NAN": np.nan
        }
        df["annual_mult"] = df["pay_period"].map(multiplier)

        # Compute normalized_salary = mean of (min,max,med) * annualiser
        if "max_salary" in df.columns and "min_salary" in df.columns:
            df["mid_salary_raw"] = df[["min_salary", "max_salary", "med_salary"]].mean(axis=1)
        elif "med_salary" in df.columns:
            df["mid_salary_raw"] = df["med_salary"]
        else:
            df["mid_salary_raw"] = np.nan

        df["normalized_salary"] = df["mid_salary_raw"] * df["annual_mult"]

    # Clip extreme outliers (keep 1st–99th percentile, only for rows with values)
    if "normalized_salary" in df.columns:
        lo = df["normalized_salary"].quantile(0.01)
        hi = df["normalized_salary"].quantile(0.99)
        df["normalized_salary"] = df["normalized_salary"].clip(lo, hi)

    # Keep one row per job_id (highest normalized_salary wins; else first)
    if "job_id" in df.columns:
        df = df.sort_values("normalized_salary", ascending=False, na_position="last")
        df = df.drop_duplicates(subset="job_id", keep="first")
    log("salaries_final", n0, len(df))

    df.to_csv(INT / "salaries_clean.csv", index=False)
    print(f"  Saved salaries_clean.csv  shape={df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. CLEAN BENEFITS  →  benefit_count per job
# ─────────────────────────────────────────────
def clean_benefits():
    print("\n=== CLEAN BENEFITS ===")
    df = pd.read_csv(RAW / "benefits.csv", low_memory=False)
    n0 = len(df)

    df = df.dropna(subset=["job_id", "type"])
    df = df.drop_duplicates(subset=["job_id", "type"])

    agg = df.groupby("job_id", as_index=False).agg(benefit_count=("type", "count"))
    log("benefits_agg", n0, len(agg))

    agg.to_csv(INT / "benefits_agg.csv", index=False)
    print(f"  Saved benefits_agg.csv  shape={agg.shape}")
    return agg


# ─────────────────────────────────────────────
# 4. CLEAN JOB_SKILLS  →  skill_count per job
# ─────────────────────────────────────────────
def clean_job_skills():
    print("\n=== CLEAN JOB_SKILLS ===")
    df = pd.read_csv(RAW / "job_skills.csv", low_memory=False)
    n0 = len(df)

    df = df.dropna(subset=["job_id", "skill_abr"])
    df = df.drop_duplicates(subset=["job_id", "skill_abr"])

    agg = df.groupby("job_id", as_index=False).agg(skill_count=("skill_abr", "count"))
    log("job_skills_agg", n0, len(agg))

    agg.to_csv(INT / "job_skills_agg.csv", index=False)
    print(f"  Saved job_skills_agg.csv  shape={agg.shape}")
    return agg


# ─────────────────────────────────────────────
# 5. CLEAN JOB_INDUSTRIES  →  primary industry per job
# ─────────────────────────────────────────────
def clean_job_industries():
    print("\n=== CLEAN JOB_INDUSTRIES ===")
    df = pd.read_csv(RAW / "job_industries.csv", low_memory=False)
    industries = pd.read_csv(RAW / "industries.csv", low_memory=False)
    n0 = len(df)

    df = df.dropna(subset=["job_id", "industry_id"])
    df = df.drop_duplicates(subset=["job_id", "industry_id"])

    # Join industry names
    df = df.merge(industries, on="industry_id", how="left")

    # Keep first industry per job as "primary"
    primary = df.drop_duplicates(subset="job_id", keep="first")[["job_id", "industry_id", "industry_name"]]
    primary = primary.rename(columns={"industry_id": "primary_industry_id",
                                       "industry_name": "primary_industry_name"})
    log("job_industries_primary", n0, len(primary))

    primary.to_csv(INT / "job_industries_primary.csv", index=False)
    print(f"  Saved job_industries_primary.csv  shape={primary.shape}")
    return primary


# ─────────────────────────────────────────────
# 6. CLEAN COMPANIES
# ─────────────────────────────────────────────
def clean_companies():
    print("\n=== CLEAN COMPANIES ===")
    df = pd.read_csv(RAW / "companies.csv", low_memory=False)
    n0 = len(df)

    df = df.drop_duplicates(subset="company_id", keep="first")

    # Normalize text
    for col in ["name", "state", "country", "city"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title().replace("Nan", pd.NA)

    if "company_size" in df.columns:
        df["company_size"] = pd.to_numeric(df["company_size"], errors="coerce")

    log("companies_final", n0, len(df))
    df.to_csv(INT / "companies_clean.csv", index=False)
    print(f"  Saved companies_clean.csv  shape={df.shape}")
    return df


# ─────────────────────────────────────────────
# 7. CLEAN EMPLOYEE_COUNTS → latest snapshot per company
# ─────────────────────────────────────────────
def clean_employee_counts():
    print("\n=== CLEAN EMPLOYEE_COUNTS ===")
    df = pd.read_csv(RAW / "employee_counts.csv", low_memory=False)
    n0 = len(df)

    df = df.dropna(subset=["company_id"])
    df["time_recorded"] = pd.to_numeric(df["time_recorded"], errors="coerce")
    df = df.sort_values("time_recorded", ascending=False)
    df = df.drop_duplicates(subset="company_id", keep="first")

    for col in ["employee_count", "follower_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log("employee_counts_latest", n0, len(df))
    df.to_csv(INT / "employee_counts_latest.csv", index=False)
    print(f"  Saved employee_counts_latest.csv  shape={df.shape}")
    return df


# ─────────────────────────────────────────────
# 8. BUILD JOBS_MASTER
# ─────────────────────────────────────────────
def build_jobs_master(postings, salaries, benefits_agg, skills_agg, industries_primary,
                      companies, emp_counts):
    print("\n=== BUILD JOBS_MASTER ===")
    n0 = len(postings)

    # --- Join sequence (all left joins to preserve all postings) ---
    master = postings.copy()

    # Salary
    sal_cols = ["job_id", "normalized_salary", "min_salary", "max_salary", "med_salary",
                "pay_period", "currency", "compensation_type"]
    sal_cols = [c for c in sal_cols if c in salaries.columns]
    master = master.merge(salaries[sal_cols], on="job_id", how="left",
                          suffixes=("", "_sal"))
    log("join_salary", n0, len(master))

    # Benefits count
    master = master.merge(benefits_agg[["job_id", "benefit_count"]], on="job_id", how="left")
    master["benefit_count"] = master["benefit_count"].fillna(0).astype(int)
    log("join_benefits", n0, len(master))

    # Skills count
    master = master.merge(skills_agg[["job_id", "skill_count"]], on="job_id", how="left")
    master["skill_count"] = master["skill_count"].fillna(0).astype(int)
    log("join_skills", n0, len(master))

    # Primary industry
    master = master.merge(industries_primary, on="job_id", how="left")
    log("join_industry", n0, len(master))

    # Company info (select columns to avoid bloat)
    comp_cols = ["company_id", "name", "state", "country", "city",
                 "company_size", "url"]
    comp_cols = [c for c in comp_cols if c in companies.columns]
    master = master.merge(companies[comp_cols], on="company_id", how="left",
                          suffixes=("", "_comp"))
    log("join_company", n0, len(master))

    # Latest employee counts
    emp_cols = ["company_id", "employee_count", "follower_count"]
    emp_cols = [c for c in emp_cols if c in emp_counts.columns]
    master = master.merge(emp_counts[emp_cols], on="company_id", how="left",
                          suffixes=("", "_emp"))
    log("join_employee_counts", n0, len(master))

    # --- Derived flag columns ---
    # salary_band
    def salary_band(s):
        if pd.isna(s):
            return "Unknown"
        elif s < 40_000:
            return "Low (<40k)"
        elif s < 80_000:
            return "Mid (40-80k)"
        elif s < 150_000:
            return "High (80-150k)"
        else:
            return "Very High (>150k)"

    if "normalized_salary" in master.columns:
        master["salary_band"] = master["normalized_salary"].apply(salary_band)
    else:
        master["salary_band"] = "Unknown"

    # is_high_views / is_high_applies  (above 75th percentile)
    for col, flag in [("views", "is_high_views"), ("applies", "is_high_applies")]:
        if col in master.columns:
            threshold = master[col].quantile(0.75)
            master[flag] = master[col] > threshold
        else:
            master[flag] = False

    # Final de-dup guard
    before = len(master)
    master = master.drop_duplicates(subset="job_id", keep="first")
    log("dedup_final", before, len(master), "ensure 1-row-per-job_id")

    # Save
    master.to_csv(PRO / "jobs_master.csv", index=False)
    print(f"\n  ✓ jobs_master saved  shape={master.shape}")
    return master


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Starting 02_clean_build_master.py ...")

    postings       = clean_postings()
    salaries       = clean_salaries()
    benefits_agg   = clean_benefits()
    skills_agg     = clean_job_skills()
    ind_primary    = clean_job_industries()
    companies      = clean_companies()
    emp_counts     = clean_employee_counts()

    master = build_jobs_master(
        postings, salaries, benefits_agg, skills_agg, ind_primary,
        companies, emp_counts
    )

    # Build report
    report_df = pd.DataFrame(build_report)
    report_df.to_csv(TBL / "jobs_master_build_report.csv", index=False)
    print(f"\n  ✓ Build report saved  shape={report_df.shape}")

    # Quick summary print
    print("\n=== jobs_master QUICK SUMMARY ===")
    print(f"  Total jobs           : {len(master):,}")
    print(f"  Total columns        : {len(master.columns)}")
    print(f"  Has salary           : {master['normalized_salary'].notna().sum():,} ({master['normalized_salary'].notna().mean():.1%})")
    print(f"  With benefits        : {(master['benefit_count'] > 0).sum():,}")
    print(f"  With skills          : {(master['skill_count'] > 0).sum():,}")
    print(f"  Remote jobs          : {master['is_remote'].sum():,}")
    if "salary_band" in master.columns:
        print("\n  Salary band distribution:")
        print(master["salary_band"].value_counts().to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()
