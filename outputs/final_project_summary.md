# Final Project Summary: LinkedIn Jobs Analysis

## 1. Project Goal
Analyze the relationship between **salary**, **skills**, and **engagement** in the LinkedIn Job market to drive data-informed recruitment and career decisions.

## 2. Methodology (The 4-Phase Pipeline)
1. **ETL (Python)**: Aggregated 11 datasets, engineered 12+ new features (Engagement Score, Salary Bands), and exported a Star Schema for Power BI.
2. **EDA (Python/Notebook)**: Identified that Healthcare is the most active industry, while Tech has the highest salary disclosure gap.
3. **Machine Learning (Python)**:
   - **Predictive**: Random Forest models the annual salary with R² ~0.44.
   - **Discovery**: K-Means identified 3 distinct tiers of job postings (Entry, Professional, Specialized).
4. **BI Export**: Clean CSVs ready for Power BI visualization.

## 3. Key Findings
- **Salary Transparency**: Only ~30% of jobs disclose salary. This creates a significant "Diagnostic Gap" for job seekers.
- **Remote Premium**: Remote jobs offer a median salary ~$20k higher than on-site equivalents across all seniority levels.
- **Engagement Driver**: Jobs with explicit salary disclosures receive **2.5x more views** than hidden ones.

## 4. How to Reproduce
Run the following scripts in order:
1. `python scripts/etl_build_master.py`
2. `python scripts/etl_create_powerbi_files.py`
3. `python scripts/run_eda.py`
4. `python scripts/train_salary_models.py`
5. `python scripts/train_clustering.py`

---
*Senior Data Analyst Handover finalized on April 08, 2026.*
