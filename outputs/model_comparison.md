# ML Model Comparison Report

This report compares the performance of regression and clustering models used to analyze LinkedIn Job postings.

## 1. Regression Analysis (Salary Prediction)
**Target Variable**: `normalized_salary`

| Model Name | MAE (USD) | RMSE (USD) | R² | Pros | Cons |
|---|---|---|---|---|---|
| **Decision Tree** | 31,050 | 44,214 | 0.3378 | Fast, easy to interpret. | Overfits easily. |
| **Random Forest** | 28,214 | 40,487 | 0.4448 | Stable, robust. | Computationally heavy. |
| **Gradient Boosting**| **26,279** | **37,751** | **0.5173** | Best accuracy. | Slow to train. |

**Conclusion**: **Gradient Boosting** is the new champion model, achieving an R² of **0.52**, which is a significant improvement over the initial benchmarks.

---

## 2. Clustering Analysis (Market Discovery)
**Model**: K-Means

- **Optimal Clusters (k)**: 3
- **Silhouette Score**: 0.5290

### Cluster Profiles:
1. **Cluster 0: Entry-Level Low-Skill**
   - Low median salary, minimal skill requirement, high volume.
2. **Cluster 1: Modern Tech/Professional**
   - High median salary, high skill count, high engagement scores.
3. **Cluster 2: Specialized/Industrial**
   - Mid-range salary, specific benefits, lower view counts.

**Conclusion**: K-Means clustering effectively identifies three distinct tiers of job postings, allowing for better-targeted talent acquisition analysis.

---

## 3. Final Recommendation
- Use the **Random Forest** predictions for the "Predicted Salary" feature in Power BI.
- Use the **K-Means Cluster Labels** to allow users to filter the dashboard by "Market Segment" rather than just by Industry.
