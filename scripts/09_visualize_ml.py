"""
09_visualize_ml.py
Generate performance plots for Salary and Hot Job models.
Saves PNG files to outputs/figures/ for presentation.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score

# ── Paths ─────────────────────────────────────────────────────
PRO_DIR     = config.DATA_PROCESSED_DIR
FIG_DIR     = config.FIGURES_DIR
TABLES_DIR  = config.TABLES_DIR

# Set style for bright high-contrast presentation
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#eff9ff",
    "axes.edgecolor": "#003859",
    "axes.labelcolor": "#003859",
    "axes.titlecolor": "#003859",
    "text.color": "#003859",
    "grid.color": "#c4d8e2",
    "grid.linewidth": 0.8,
    "xtick.color": "#003859",
    "ytick.color": "#003859"
})
ACCENT_COLOR = '#184683'
HOT_COLOR    = '#003859'

# ════════════════════════════════════════════════════════════
# 1. VISUALIZE SALARY MODEL
# ════════════════════════════════════════════════════════════
def plot_salary_performance():
    print("Plotting Salary Model performance …")
    df = pd.read_csv(PRO_DIR / "salary_predictions.csv")
    
    # --- 1.1 Actual vs Predicted ---
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df, x='normalized_salary', y='pred_salary_hgb', 
        scatter_kws={'alpha':0.4, 'color':ACCENT_COLOR, 's':15},
        line_kws={'color':'#ff4444', 'lw':3, 'ls':'-'}
    )
    plt.title("Salary Model: Actual vs Predicted (HistGradientBoosting)", fontsize=14, pad=15)
    plt.xlabel("Actual Salary ($)", fontsize=12)
    plt.ylabel("Predicted Salary ($)", fontsize=12)
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_salary_actual_vs_pred.png", dpi=200)

    # --- 1.2 Residual Plot (Check for Bias) ---
    plt.figure(figsize=(10, 6))
    residuals = df['normalized_salary'] - df['pred_salary_hgb']
    sns.histplot(residuals, kde=True, color=ACCENT_COLOR, bins=50)
    plt.axvline(0, color='white', linestyle='--', lw=2)
    plt.title("Error Distribution (Residuals Analysis)", fontsize=14, pad=15)
    plt.xlabel("Prediction Error ($)", fontsize=12)
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_salary_residuals.png", dpi=200)
    print(f"✓ Saved Salary viz")

# ════════════════════════════════════════════════════════════
# 2. VISUALIZE CLUSTERING (PCA)
# ════════════════════════════════════════════════════════════
def plot_clustering():
    print("Plotting Market Clusters …")
    # Join with master to get features
    df_clusters = pd.read_csv(PRO_DIR / "powerbi/cluster_results.csv")
    # Note: Using views as a proxy for engagement
    df_master = pd.read_csv(PRO_DIR / "jobs_master.csv", usecols=['job_id', 'skill_count', 'views'])
    df = pd.merge(df_clusters, df_master, on='job_id')
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='skill_count', y='views', hue='cluster', palette=['#184683', '#eff9ff', '#337ab7'], alpha=0.7, s=40, edgecolor='white', linewidth=0.5)
    plt.yscale('log')
    plt.title("Market Segmentation: K-Means Clustering Results", fontsize=14, pad=15)
    plt.xlabel("Skill Count", fontsize=12)
    plt.ylabel("Views (Log Scale)", fontsize=12)
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_clusters_view.png", dpi=200)
    print(f"✓ Saved Market Clusters viz")

# ════════════════════════════════════════════════════════════
# 3. VISUALIZE HOT JOB MODEL
# ════════════════════════════════════════════════════════════
def plot_hot_job_performance():
    print("Plotting Hot Job Model performance …")
    path = PRO_DIR / "hot_job_predictions.csv"
    if not path.exists():
        print("  ⚠ hot_job_predictions.csv not found. Skipping.")
        return
        
    df = pd.read_csv(path)
    
    # 3.1 Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df['is_hot_job'], df['hot_job_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Hot Job: Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("Actual Labels", fontsize=12)
    plt.xticks([0.5, 1.5], ['Not Hot', 'Hot'])
    plt.yticks([0.5, 1.5], ['Not Hot', 'Hot'])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_hot_job_confusion_matrix.png", dpi=200)

    # 3.2 ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(df['is_hot_job'], df['hot_job_prob'])
    roc_auc_full = auc(fpr, tpr)
    
    # Lấy chỉ số AUC của tập Test (để đồng bộ với báo cáo)
    try:
        metrics_df = pd.read_csv(TABLES_DIR / 'hot_job_model_metrics.csv')
        # Lấy row có RandomForest (Champion model cho Hot Job)
        rf_auc = metrics_df[metrics_df['model'] == 'RandomForest']['roc_auc'].values[0]
        display_auc = rf_auc
    except:
        display_auc = roc_auc_full

    plt.plot(fpr, tpr, color='#ff4444', lw=3, label=f'ROC Curve (AUC = {display_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#003859', linestyle='--', alpha=0.5)
    plt.title("Hot Job: ROC Curve", fontsize=14, pad=15)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_hot_job_roc_curve.png", dpi=200)
    print(f"✓ Saved Hot Job viz")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_salary_performance()
    plt.close('all')
    
    plot_clustering()
    plt.close('all')
    
    plot_hot_job_performance()
    plt.close('all')
        
    print("\nAll ML visualizations generated successfully!")
