"""
05_hot_job_model.py
Classification model to predict "Hot Jobs" (high engagement).
Target: is_hot_job = 1 if views >= 8 (75th percentile), else 0.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# ── Paths ─────────────────────────────────────────────────────
MODELS_DIR = config.MODELS_DIR
TABLES_DIR = config.TABLES_DIR
PRO_DIR    = config.DATA_PROCESSED_DIR

# ════════════════════════════════════════════════════════════
# 1. LOAD & LABEL
# ════════════════════════════════════════════════════════════
print("Loading jobs_master.csv …")
df = pd.read_csv(PRO_DIR / "jobs_master.csv", low_memory=False)

# Target: is_hot_job = 1 if views >= 8, else 0
# We drop rows where views is NaN to have a clean training set
df = df.dropna(subset=["views"])
df["is_hot_job"] = (df["views"] >= 8).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['is_hot_job'].value_counts(normalize=True)}")

# ════════════════════════════════════════════════════════════
# 2. FEATURE SELECTION & LEAKAGE PREVENTION
# ════════════════════════════════════════════════════════════
TARGET = "is_hot_job"

# LEAKAGE: Explicitly exclude columns derived from engagement
LEAKAGE_COLS = ["views", "applies", "is_high_views", "is_high_applies"]

# Features
TITLE_COL = "title"
CAT_COLS = [
    "primary_industry_name", 
    "formatted_work_type", 
    "formatted_experience_level", 
    "is_remote", 
    "is_sponsored"
]
NUM_COLS = [
    "skill_count", 
    "benefit_count", 
    "description_length", 
    "company_size", 
    "employee_count",
    "follower_count"
]

# Ensure cols exist
CAT_COLS = [c for c in CAT_COLS if c in df.columns]
NUM_COLS = [c for c in NUM_COLS if c in df.columns]

# Ensure categorical are strings for the imputer/encoder
for col in CAT_COLS:
    df[col] = df[col].astype(str).replace("nan", "Unknown").replace("None", "Unknown")

print(f"Categorical features: {CAT_COLS}")
print(f"Numerical features: {NUM_COLS}")

# Prepare X and y
X = df[[TITLE_COL] + CAT_COLS + NUM_COLS]
y = df[TARGET]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ════════════════════════════════════════════════════════════
# 3. PREPROCESSING PIPELINE
# ════════════════════════════════════════════════════════════

# Text: Title (TF-IDF)
tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=500, stop_words="english"))
])

# Categorical: OHE (NaNs already handled by string conversion)
cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Numerical: Impute + Scale
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("text", tfidf_pipe, TITLE_COL),
        ("cat", cat_pipe, CAT_COLS),
        ("num", num_pipe, NUM_COLS)
    ]
)

# ════════════════════════════════════════════════════════════
# 4. MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════
models = {
    "LogisticRegression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
}

metrics_list = []
best_f1 = 0
best_model = None
best_model_name = ""

for name, pipe in models.items():
    print(f"\nTraining {name}...")
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    
    metrics_list.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipe
        best_model_name = name

# Save metrics
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(TABLES_DIR / "hot_job_model_metrics.csv", index=False)

# Save best model
joblib.dump(best_model, MODELS_DIR / "hot_job_model.pkl")
print(f"\nBest model ({best_model_name}) saved to {MODELS_DIR / 'hot_job_model.pkl'}")

# ════════════════════════════════════════════════════════════
# 5. SAVE PREDICTIONS
# ════════════════════════════════════════════════════════════
# Full dataset predictions
df["hot_job_prob"] = best_model.predict_proba(X)[:, 1]
df["hot_job_pred"] = best_model.predict(X)

# Select key columns for export
export_cols = [
    "job_id", "title", "views", "is_hot_job", "hot_job_prob", "hot_job_pred", 
    "primary_industry_name", "formatted_experience_level"
]
df[export_cols].to_csv(PRO_DIR / "hot_job_predictions.csv", index=False)
print(f"Predictions saved to {PRO_DIR / 'hot_job_predictions.csv'}")

# ════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE (RandomForest)
# ════════════════════════════════════════════════════════════
if best_model_name == "RandomForest":
    print("\nExtracting feature importance from RandomForest...")
    rf = best_model.named_steps["model"]
    prep = best_model.named_steps["prep"]
    
    # Get feature names
    feature_names = prep.get_feature_names_out()
    importances = rf.feature_importances_
    
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
    print("\nTop 20 Features Importance:")
    print(feat_imp)
    
    # Save importance
    feat_imp.to_csv(TABLES_DIR / "hot_job_feature_importance.csv")

print("\nDone!")
