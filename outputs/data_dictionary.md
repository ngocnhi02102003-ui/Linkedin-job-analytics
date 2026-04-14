# Data Dictionary: LinkedIn Jobs Project

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
