# Power BI Modeling Guide

This guide ensures your Power BI dashboard correctly represents the Machine Learning and ETL outputs.

## 1. Data Model (Star Schema)
Connect the tables in the **Model View** as follows:

- **Fact Table**: `summary_for_dashboard` (Main)
- **Dimension Tables**:
  - `dim_company` (Join on `company_id`)
  - `dim_industry` (Join on `primary_industry_id`)
  - `dim_calendar` (Join on `listed_date`)
- **Bridge Tables (Many-to-Many)**:
  - `fact_job_skill` (Link to `dim_skill` and `summary_for_dashboard`)
  - `fact_job_benefit` (Link to `summary_for_dashboard`)

## 2. Incorporating ML Results
To add Predictive and Discovery layers:
1. **Join `salary_predictions`**: Left-join with the Fact table on `job_id`. Use `salary_pred` to show "Estimated Market Value".
2. **Join `cluster_results`**: Left-join with the Fact table on `job_id`. Create a Slicer for `cluster` to allow segment-based filtering.

## 3. Recommended DAX Measures
- **`Actual vs Predicted Gap`** = `AVERAGE(summary_for_dashboard[normalized_salary]) - AVERAGE(salary_predictions[salary_pred])`
- **`Conversion Rate`** = `DIVIDE(SUM(summary_for_dashboard[applies]), SUM(summary_for_dashboard[views]))`
- **`Skills Intensity Index`** = `AVERAGE(summary_for_dashboard[skill_count])`

## 4. Visualization Best Practices
- **Diagnostic Chart**: Use a scatter plot with `normalized_salary` vs `engagement_score`, colored by `cluster`.
- **Descriptive Chart**: Use a Tree Map for `Total Jobs` by `primary_industry_name`.
- **Top Skills**: Use a Bar Chart on `fact_job_skill` filtered by the top N skills.
