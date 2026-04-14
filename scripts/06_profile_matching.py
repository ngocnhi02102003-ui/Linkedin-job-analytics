"""
06_profile_matching.py
Career orientation prototype: Match candidate profile to LinkedIn job data.
Uses Jaccard Similarity on skills and hard-filtering on experience/industry.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────
PRO_DIR     = config.DATA_PROCESSED_DIR
RAW_DIR     = config.DATA_RAW_DIR

# ════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS DATA
# ════════════════════════════════════════════════════════════
def load_data():
    print("Loading datasets …")
    # jobs_master: title, industry, salary, exp_level, work_type
    df_jobs = pd.read_csv(PRO_DIR / "jobs_master.csv", low_memory=False)
    
    # job_skills: job_id, skill_abr
    df_job_skills = pd.read_csv(RAW_DIR / "job_skills.csv")
    
    # skills: skill_abr, skill_name
    df_skills_ref = pd.read_csv(RAW_DIR / "skills.csv")
    
    # Map job_ids to sets of skill names (not just abr)
    job_skills_joined = df_job_skills.merge(df_skills_ref, on="skill_abr", how="left")
    job_skills_map = job_skills_joined.groupby("job_id")["skill_name"].apply(set).to_dict()
    
    # Merge skill names back into df_jobs for easier handling if needed
    # but using a dictionary is faster for matching.
    return df_jobs, job_skills_map, df_skills_ref

# ════════════════════════════════════════════════════════════
# 2. MATCHING ENGINE
# ════════════════════════════════════════════════════════════
def match_profile(user_skills, exp_level=None, industry=None, work_type=None, 
                  df_jobs=None, job_skills_map=None, df_skills_ref=None):
    
    # Step 1: Pre-Filtering (Alignment)
    # ────────────────────────────────
    df = df_jobs.copy()
    
    if exp_level:
        # Flexible matching for Experience Level strings
        df = df[df["formatted_experience_level"].str.contains(exp_level, case=False, na=False)]
    
    if industry:
        df = df[df["primary_industry_name"].str.contains(industry, case=False, na=False)]
        
    if work_type:
        df = df[df["formatted_work_type"].str.contains(work_type, case=False, na=False)]
    
    if df.empty:
        return pd.DataFrame()

    # Step 2: Semantic Mapping for User Skills
    # ──────────────────────────────────────
    # Map raw user input to standard LinkedIn skill names
    standard_skill_names = df_skills_ref["skill_name"].tolist()
    mapped_user_skills = set()
    
    for u_skill in user_skills:
        # Simple keyword matching (e.g., 'Python' matches 'Information Technology')
        # In this dataset, skills are broad. Let's make a mini-map for demo purposes.
        it_keywords = ["python", "sql", "excel", "data", "software", "it", "code"]
        if any(k in u_skill.lower() for k in it_keywords):
            mapped_user_skills.add("Information Technology")
            # In a real app, this would be a large NLP map.
            # Let's also check direct matches for generic categories.
        for s_name in standard_skill_names:
            if u_skill.lower() == s_name.lower():
                mapped_user_skills.add(s_name)
    
    print(f"Mapped user skills: {mapped_user_skills}")

    # Step 3: Calculate Similarity (Jaccard)
    # ──────────────────────────────────────
    results = []
    
    for idx, row in df.iterrows():
        jid = row["job_id"]
        job_skills = job_skills_map.get(jid, set())
        
        # Jaccard = Intersection / Union
        intersection = mapped_user_skills.intersection(job_skills)
        union = mapped_user_skills.union(job_skills)
        
        score = len(intersection) / len(union) if union else 0
        
        missing = job_skills - mapped_user_skills
        
        results.append({
            "job_id": jid,
            "title": row["title"],
            "company": row["company_name"],
            "industry": row["primary_industry_name"],
            "experience": row["formatted_experience_level"],
            "salary": row["normalized_salary"],
            "matching_score": round(score, 4),
            "required_skills": list(job_skills),
            "missing_skills": list(missing)
        })
    
    return pd.DataFrame(results).sort_values("matching_score", ascending=False)

# ════════════════════════════════════════════════════════════
# 3. RUN SAMPLE & EXPORT
# ════════════════════════════════════════════════════════════
def main():
    df_jobs, job_skills_map, df_skills_ref = load_data()
    
    # ───────────────────────────────────────────
    # SAMPLE PROFILE
    # ───────────────────────────────────────────
    user_skills = ["Python", "SQL", "Excel", "Project Management"]
    exp_level   = "Mid-Senior Level"
    industry    = None  # Open to all industries
    work_type   = None  # Open to all work types
    
    print(f"\nMatching profile for: {user_skills} | {exp_level}")
    
    matches = match_profile(
        user_skills, exp_level, industry, work_type,
        df_jobs, job_skills_map, df_skills_ref
    )
    
    if matches.empty:
        print("No matches found with current filters.")
        return

    # Top 5
    top_5 = matches.head(5).copy()
    
    # REFERENCE SALARY AGGREGATION
    # ────────────────────────────
    # Look at the top 100 matches to get a broader salary range
    salary_pool = matches[matches["salary"].notna() & (matches["salary"] > 0)].head(100)
    if not salary_pool.empty:
        suggested_min = salary_pool["salary"].quantile(0.25)
        suggested_med = salary_pool["salary"].median()
        suggested_max = salary_pool["salary"].quantile(0.75)
    else:
        suggested_min = suggested_med = suggested_max = 0

    print("\n" + "="*50)
    print("TOP 5 RECOMMENDED ROLES")
    print("="*50)
    for i, row in enumerate(top_5.itertuples(), 1):
        print(f"{i}. {row.title} @ {row.company}")
        print(f"   Score: {row.matching_score:.2%} | Industry: {row.industry}")
        print(f"   Missing skills: {row.missing_skills if row.missing_skills else 'None'}")
        print("-" * 30)

    print("\n" + "="*50)
    print("SALARY SUGGESTION BASED ON YOUR PROFILE")
    print("="*50)
    print(f"Reference Median: ${suggested_med:,.0f} / year")
    print(f"Projected Range : ${suggested_min:,.0f} - ${suggested_max:,.0f}")
    
    # Save the full matching table for reference
    top_save = matches.head(20).copy()
    top_save.to_csv(PRO_DIR / "profile_match_results.csv", index=False)
    print(f"\n✓ Saved top 20 matches to {PRO_DIR / 'profile_match_results.csv'}")

if __name__ == "__main__":
    main()
