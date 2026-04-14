"""
train_salary_models.py
Phase 3a: ML — Salary Prediction Benchmarking.
Verbose output: Data Prep → Train DT/RF/GB → Compare → Save.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent))
import config

# ════════════════════════════════════════════════════════════
# STEP 1: PREPARE DATA
# ════════════════════════════════════════════════════════════
def step1_prepare():
    print("\n" + "="*60)
    print("ML STEP 1: PREPARE DATA FOR REGRESSION")
    print("="*60)
    
    df = pd.read_csv(config.DATA_PROCESSED_DIR / "jobs_master.csv")
    print(f"  Loaded jobs_master.csv: {len(df):,} rows")
    
    # Filter: only rows with valid salary in reasonable range
    df_sal = df[(df["has_salary"] == 1) & 
                (df["normalized_salary"] >= 20000) & 
                (df["normalized_salary"] <= 400000)].copy()
    print(f"  After salary filter (20k-400k): {len(df_sal):,} rows")
    
    # Define features
    cat_features = ["work_type_clean", "experience_group_clean", "is_remote", "is_sponsored"]
    num_features = ["skill_count", "benefit_count", "description_length"]
    target = "normalized_salary"
    
    print(f"\n  Features selected:")
    print(f"    Categorical: {cat_features}")
    print(f"    Numerical:   {num_features}")
    print(f"    Target:      {target}")
    
    # Train/Test split
    X = df_sal[cat_features + num_features]
    y = df_sal[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n  Train/Test Split:")
    print(f"    Train: {len(X_train):,} rows")
    print(f"    Test:  {len(X_test):,} rows")
    
    # Build preprocessor
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, cat_features),
        ("num", num_pipe, num_features)
    ])
    
    return X_train, X_test, y_train, y_test, preprocessor, df_sal

# ════════════════════════════════════════════════════════════
# STEP 2-4: TRAIN EACH MODEL
# ════════════════════════════════════════════════════════════
def train_model(name, model, preprocessor, X_train, X_test, y_train, y_test, step_num):
    print(f"\n" + "="*60)
    print(f"ML STEP {step_num}: TRAIN {name.upper()}")
    print("="*60)
    
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    
    print(f"  Training {name} …")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n  Evaluation Results:")
    print(f"    MAE:   ${mae:,.0f}")
    print(f"    RMSE:  ${rmse:,.0f}")
    print(f"    R²:    {r2:.4f}")
    
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "pipeline": pipe}

# ════════════════════════════════════════════════════════════
# STEP 5: COMPARE MODELS
# ════════════════════════════════════════════════════════════
def step5_compare(results):
    print("\n" + "="*60)
    print("ML STEP 5: MODEL COMPARISON")
    print("="*60)
    
    print(f"\n  {'Model':<22} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
    print("  " + "-"*58)
    for r in results:
        print(f"  {r['model']:<22} ${r['MAE']:>10,.0f} ${r['RMSE']:>10,.0f} {r['R2']:>10.4f}")
    
    best = max(results, key=lambda x: x["R2"])
    print(f"\n  ★ BEST MODEL: {best['model']} (R² = {best['R2']:.4f})")
    
    # Comparison chart
    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, metric in enumerate(["MAE", "RMSE", "R2"]):
        ax = axes[i]
        colors = ["#2ecc71" if r["model"] == best["model"] else "#95a5a6" for r in results]
        ax.bar([r["model"] for r in results], metrics_df[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Regression Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "regression_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✓ Saved regression_comparison.png")
    
    return best, metrics_df

# ════════════════════════════════════════════════════════════
# STEP 6: SAVE OUTPUTS
# ════════════════════════════════════════════════════════════
def step6_save(best, metrics_df, df_sal):
    print("\n" + "="*60)
    print("ML STEP 6: SAVE OUTPUTS")
    print("="*60)
    
    # 1. Metrics CSV
    metrics_df.to_csv(config.METRICS_DIR / "regression_metrics.csv", index=False)
    print("  ✓ Saved metrics/regression_metrics.csv")
    
    # 2. Feature Importance
    pipe = best["pipeline"]
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False).head(15)
    
    print(f"\n  Top 10 Feature Importances ({best['model']}):")
    for _, row in fi.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:<35} {row['importance']:.4f} {bar}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=fi, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=ax)
    ax.set_title(f"Feature Importance — {best['model']}")
    fig.savefig(config.CHARTS_DIR / "feature_importance_salary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved feature_importance_salary.png")
    
    # 3. Predictions CSV
    df_sal["salary_pred"] = pipe.predict(df_sal[pipe.feature_names_in_])
    df_sal[["job_id", "normalized_salary", "salary_pred"]].to_csv(
        config.PBI_DIR / "salary_predictions.csv", index=False
    )
    print(f"  ✓ Saved powerbi/salary_predictions.csv ({len(df_sal):,} rows)")
    
    # 4. Save model
    joblib.dump(pipe, config.MODELS_DIR / "salary_model.pkl")
    print("  ✓ Saved models/salary_model.pkl")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ML PIPELINE: Salary Prediction Benchmarking        ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    try:
        X_train, X_test, y_train, y_test, prep, df_sal = step1_prepare()
        
        models = [
            ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=42)),
            ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)),
        ]
        
        results = []
        for i, (name, model) in enumerate(models):
            r = train_model(name, model, prep, X_train, X_test, y_train, y_test, step_num=i+2)
            results.append(r)
        
        best, metrics_df = step5_compare(results)
        step6_save(best, metrics_df, df_sal)
        
        print("\n" + "="*60)
        print("✓ SUCCESS: ML Phase 3a (Regression) Complete.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
