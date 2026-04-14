"""
10_simulate_and_test.py
Mock data simulation to test Salary and Hot Job models.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config

import pandas as pd
import joblib
import numpy as np

# ── Paths ─────────────────────────────────────────────────────
MODELS_DIR = config.MODELS_DIR

def main():
    print("Loading models …")
    try:
        sal_model = joblib.load(MODELS_DIR / "salary_model.pkl")
        hot_model = joblib.load(MODELS_DIR / "hot_job_model.pkl")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # ════════════════════════════════════════════════════════════
    # 1. CREATE MOCK DATA
    # ════════════════════════════════════════════════════════════
    # We combine features required by both models
    mock_postings = [
        {
            "title": "Senior Software Engineer (AI/ML)",
            "primary_industry_name": "IT Services and IT Consulting",
            "formatted_work_type": "Full-Time",
            "formatted_experience_level": "Mid-Senior Level",
            "is_remote": "True",         # string — matches script 05 training pipeline
            "is_sponsored": "False",
            "country": "US",
            "skill_count": 12,
            "benefit_count": 8,
            "description_length": 3500,
            "company_size": 5.0,
            "employee_count": 10000,
            "follower_count": 250000
        },
        {
            "title": "Registered Nurse - ICU",
            "primary_industry_name": "Hospitals and Health Care",
            "formatted_work_type": "Full-Time",
            "formatted_experience_level": "Entry Level",
            "is_remote": "False",
            "is_sponsored": "True",
            "country": "US",
            "skill_count": 6,
            "benefit_count": 4,
            "description_length": 1500,
            "company_size": 4.0,
            "employee_count": 2000,
            "follower_count": 5000
        },
        {
            "title": "Marketing Coordinator",
            "primary_industry_name": "Retail",
            "formatted_work_type": "Full-Time",
            "formatted_experience_level": "Associate",
            "is_remote": "True",
            "is_sponsored": "False",
            "country": "US",
            "skill_count": 4,
            "benefit_count": 2,
            "description_length": 1100,
            "company_size": 2.0,
            "employee_count": 150,
            "follower_count": 800
        },
        {
            "title": "Data Analyst",
            "primary_industry_name": "Financial Services",
            "formatted_work_type": "Full-Time",
            "formatted_experience_level": "Mid-Senior Level",
            "is_remote": "True",
            "is_sponsored": "True",
            "country": "US",
            "skill_count": 9,
            "benefit_count": 6,
            "description_length": 2200,
            "company_size": 4.0,
            "employee_count": 3000,
            "follower_count": 50000
        },
        {
            "title": "Customer Service Representative",
            "primary_industry_name": "Retail",
            "formatted_work_type": "Contract",
            "formatted_experience_level": "Entry Level",
            "is_remote": "False",
            "is_sponsored": "False",
            "country": "US",
            "skill_count": 2,
            "benefit_count": 1,
            "description_length": 700,
            "company_size": 3.0,
            "employee_count": 500,
            "follower_count": 2000
        },
    ]

    df_mock = pd.DataFrame(mock_postings)

    # Script 04 (salary) uses is_remote / is_sponsored as integers in CAT_COLS
    # Script 05 (hot_job) uses them as strings in CAT_COLS
    # We need two separate versions of the DataFrame:
    df_sal_mock = df_mock.copy()
    df_sal_mock["is_remote"]    = df_sal_mock["is_remote"].map({"True": 1, "False": 0})
    df_sal_mock["is_sponsored"] = df_sal_mock["is_sponsored"].map({"True": 1, "False": 0})
    
    # ════════════════════════════════════════════════════════════
    # 2. PREPROCESS & RUN PREDICTIONS
    # ════════════════════════════════════════════════════════════
    
    # Pre-process df_mock for Hot Job (Model 05) - needs strings for cat cols
    df_hot_mock = df_mock.copy()
    HOT_CAT_COLS = ["primary_industry_name", "formatted_work_type", "formatted_experience_level", "is_remote", "is_sponsored"]
    for col in HOT_CAT_COLS:
        if col in df_hot_mock.columns:
            df_hot_mock[col] = df_hot_mock[col].astype(str).replace("nan", "Unknown").replace("None", "Unknown")

    # Metrics for salary model logic
    try:
        metrics = pd.read_csv(config.TABLES_DIR / "salary_model_metrics.csv")
        best_name = metrics.loc[metrics["R2"].idxmax(), "model"]
    except:
        best_name = "RandomForest" # fallback

    # Run Salary Predictions
    sal_raw = sal_model.predict(df_sal_mock)
    if best_name == "Ridge":
        pred_salaries = np.expm1(sal_raw)
    else:
        pred_salaries = sal_raw

    # Run Hot Job Predictions
    hot_probs = hot_model.predict_proba(df_hot_mock)[:, 1]

    # ════════════════════════════════════════════════════════════
    # 3. DISPLAY RESULTS
    # ════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"{'JOB TITLE':<35} | {'SALARY':<12} | {'HOT PROB':<10}")
    print("-" * 70)

    for i, row in df_mock.iterrows():
        title = row['title']
        sal = pred_salaries[i]
        hp = hot_probs[i]
        print(f"{title:<35} | ${sal:>10,.0f} | {hp:>9.1%}")

    print("="*70)
    print("\n✓ Simulation Complete.")

if __name__ == "__main__":
    main()
