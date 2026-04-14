"""
train_transparency_classification.py
Phase 3c: ML — Salary Transparency Prediction.
Verbose output: Data Prep → Train Logistic → Evaluate → Drivers → Save.
Predicts whether a job posting will disclose salary (has_salary = 1 or 0).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

sys.path.append(str(Path(__file__).resolve().parent))
import config

# ════════════════════════════════════════════════════════════
# STEP 1: PREPARE DATA
# ════════════════════════════════════════════════════════════
def step1_prepare():
    print("\n" + "="*60)
    print("CLASS STEP 1: PREPARE DATA FOR CLASSIFICATION")
    print("="*60)
    
    df = pd.read_csv(config.DATA_PROCESSED_DIR / "jobs_master.csv")
    print(f"  Loaded jobs_master.csv: {len(df):,} rows")
    
    # Target distribution
    target = "has_salary"
    print(f"\n  Target: {target}")
    print(f"    {'Class':<15} {'Count':>10} {'Pct':>8}")
    print("    " + "-"*35)
    for val, cnt in df[target].value_counts().sort_index().items():
        label = "Disclosed" if val == 1 else "Hidden"
        pct = cnt / len(df) * 100
        print(f"    {label:<15} {cnt:>10,} {pct:>7.1f}%")
    
    # Features
    cat_features = ["primary_industry_name", "work_type_clean", 
                     "experience_group_clean", "is_remote", "is_sponsored"]
    num_features = ["skill_count", "benefit_count", "description_length"]
    
    print(f"\n  Features:")
    print(f"    Categorical: {cat_features}")
    print(f"    Numerical:   {num_features}")
    
    X = df[cat_features + num_features].copy()
    y = df[target].copy()
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Train/Test Split (stratified):")
    print(f"    Train: {len(X_train):,} rows (Disclosed: {y_train.sum():,})")
    print(f"    Test:  {len(X_test):,} rows (Disclosed: {y_test.sum():,})")
    
    # Preprocessor
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
    
    return X_train, X_test, y_train, y_test, preprocessor

# ════════════════════════════════════════════════════════════
# STEP 2: TRAIN LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════
def step2_train(X_train, X_test, y_train, y_test, preprocessor):
    print("\n" + "="*60)
    print("CLASS STEP 2: TRAIN LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=1000, random_state=42, 
        class_weight="balanced",  # Handle imbalanced classes
        solver="lbfgs"
    )
    
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    
    print("  Training Logistic Regression (class_weight=balanced) …")
    pipe.fit(X_train, y_train)
    print("  ✓ Training complete.")
    
    return pipe

# ════════════════════════════════════════════════════════════
# STEP 3: EVALUATE MODEL
# ════════════════════════════════════════════════════════════
def step3_evaluate(pipe, X_test, y_test):
    print("\n" + "="*60)
    print("CLASS STEP 3: EVALUATE MODEL")
    print("="*60)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n  {'Metric':<20} {'Value':>10}")
    print("  " + "-"*32)
    print(f"  {'Accuracy':<20} {acc:>10.4f}")
    print(f"  {'Precision':<20} {prec:>10.4f}")
    print(f"  {'Recall':<20} {rec:>10.4f}")
    print(f"  {'F1-Score':<20} {f1:>10.4f}")
    print(f"  {'ROC-AUC':<20} {auc:>10.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Hidden    Disclosed")
    print(f"    Actual Hidden   {cm[0][0]:>8,}   {cm[0][1]:>8,}")
    print(f"    Actual Disclosed{cm[1][0]:>8,}   {cm[1][1]:>8,}")
    
    # Chart: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Hidden", "Disclosed"],
                yticklabels=["Hidden", "Disclosed"], ax=ax)
    ax.set_title("Confusion Matrix: Salary Transparency Prediction")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.savefig(config.CHARTS_DIR / "transparency_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved transparency_confusion_matrix.png")
    
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": auc}
    return metrics

# ════════════════════════════════════════════════════════════
# STEP 4: EXTRACT TRANSPARENCY DRIVERS
# ════════════════════════════════════════════════════════════
def step4_drivers(pipe):
    print("\n" + "="*60)
    print("CLASS STEP 4: TRANSPARENCY DRIVERS (Coefficients)")
    print("="*60)
    
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    coefs = model.coef_[0]
    
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    }).sort_values(by="coefficient", ascending=False)
    
    # Top 10 positive (encourage disclosure)
    print(f"\n  Top 10 POSITIVE drivers (encourage salary disclosure):")
    print(f"    {'Feature':<45} {'Coef':>8}")
    print("    " + "-"*55)
    for _, row in coef_df.head(10).iterrows():
        direction = "+" if row["coefficient"] > 0 else ""
        bar = "█" * int(abs(row["coefficient"]) * 5)
        print(f"    {row['feature']:<45} {direction}{row['coefficient']:>7.4f} {bar}")
    
    # Top 10 negative (encourage hiding)
    print(f"\n  Top 10 NEGATIVE drivers (encourage salary hiding):")
    print(f"    {'Feature':<45} {'Coef':>8}")
    print("    " + "-"*55)
    for _, row in coef_df.tail(10).iterrows():
        bar = "█" * int(abs(row["coefficient"]) * 5)
        print(f"    {row['feature']:<45} {row['coefficient']:>8.4f} {bar}")
    
    # Chart: Top/Bottom drivers
    top_bottom = pd.concat([coef_df.head(10), coef_df.tail(10)])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top_bottom["coefficient"]]
    ax.barh(top_bottom["feature"], top_bottom["coefficient"], color=colors)
    ax.set_title("Salary Transparency Drivers\n(Green = encourages disclosure, Red = encourages hiding)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "transparency_drivers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  ✓ Saved transparency_drivers.png")
    
    return coef_df

# ════════════════════════════════════════════════════════════
# STEP 5: SAVE OUTPUTS
# ════════════════════════════════════════════════════════════
def step5_save(pipe, metrics, coef_df):
    print("\n" + "="*60)
    print("CLASS STEP 5: SAVE OUTPUTS")
    print("="*60)
    
    # 1. Metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(config.METRICS_DIR / "classification_metrics.csv", index=False)
    print("  ✓ Saved metrics/classification_metrics.csv")
    
    # 2. Coefficients CSV
    coef_df.to_csv(config.METRICS_DIR / "transparency_coefficients.csv", index=False)
    print("  ✓ Saved metrics/transparency_coefficients.csv")
    
    # 3. Model pickle
    joblib.dump(pipe, config.MODELS_DIR / "transparency_model.pkl")
    print("  ✓ Saved models/transparency_model.pkl")
    
    # 4. Summary
    print(f"\n  Output files created:")
    print(f"    metrics/classification_metrics.csv")
    print(f"    metrics/transparency_coefficients.csv")
    print(f"    charts/transparency_confusion_matrix.png")
    print(f"    charts/transparency_drivers.png")
    print(f"    models/transparency_model.pkl")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ML PIPELINE: Salary Transparency Classification   ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    try:
        X_train, X_test, y_train, y_test, prep = step1_prepare()
        pipe = step2_train(X_train, X_test, y_train, y_test, prep)
        metrics = step3_evaluate(pipe, X_test, y_test)
        coef_df = step4_drivers(pipe)
        step5_save(pipe, metrics, coef_df)
        
        print("\n" + "="*60)
        print("✓ SUCCESS: ML Phase 3c (Classification) Complete.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
