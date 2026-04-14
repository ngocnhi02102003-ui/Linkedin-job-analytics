"""
04_salary_model.py
Salary prediction model: Ridge Regression vs RandomForestRegressor
Input : data_processed/jobs_master.csv
Output: outputs/models/salary_model.pkl
        outputs/tables/salary_model_metrics.csv
        data_processed/salary_predictions.csv
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────
MODELS_DIR = config.MODELS_DIR
TABLES_DIR = config.TABLES_DIR
PRO_DIR    = config.DATA_PROCESSED_DIR

# ════════════════════════════════════════════════════════════
# 1. LOAD & FILTER
# ════════════════════════════════════════════════════════════
print("Loading jobs_master.csv …")
df = pd.read_csv(PRO_DIR / "jobs_master.csv", low_memory=False)
print(f"  Full dataset: {df.shape}")

# Keep only rows with a valid salary target
df_sal = df[df["normalized_salary"].notna() & (df["normalized_salary"] > 0)].copy()
print(f"  Rows with salary (raw): {len(df_sal):,}")

# Filter to realistic US annual salary range ($20k–$400k)
# Removes bad unit conversions (e.g. hourly rates stored as-is) and extreme outliers
SAL_MIN, SAL_MAX = 20_000, 400_000
df_sal = df_sal[(df_sal["normalized_salary"] >= SAL_MIN) &
                (df_sal["normalized_salary"] <= SAL_MAX)].copy()
print(f"  Rows after outlier filter ({SAL_MIN:,}–{SAL_MAX:,}): {len(df_sal):,}")

# ════════════════════════════════════════════════════════════
# 2. FEATURE SELECTION
# ════════════════════════════════════════════════════════════
TARGET = "normalized_salary"

TITLE_COL   = "title"          # TF-IDF text feature
CAT_COLS    = [
    "formatted_work_type",
    "formatted_experience_level",
    "primary_industry_name",
    "country",
    "is_remote",
    "is_sponsored",
    "salary_band",             # excluded from target leak check below
]
NUM_COLS    = [
    "benefit_count",
    "skill_count",
    "description_length",
    "employee_count",
    "follower_count",
]

# Drop salary_band — it's derived from the target (leakage)
CAT_COLS = [c for c in CAT_COLS if c != "salary_band"]

# Only keep columns that exist
CAT_COLS = [c for c in CAT_COLS if c in df_sal.columns]
NUM_COLS = [c for c in NUM_COLS if c in df_sal.columns]

print(f"\n  Text col   : {TITLE_COL}")
print(f"  Cat cols   : {CAT_COLS}")
print(f"  Num cols   : {NUM_COLS}")

# Fill missing text with empty string
df_sal[TITLE_COL] = df_sal[TITLE_COL].fillna("").astype(str)

# Encode booleans as int for sklearn
for c in CAT_COLS:
    if df_sal[c].dtype == bool:
        df_sal[c] = df_sal[c].astype(int)

# ════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════
X = df_sal[[TITLE_COL] + CAT_COLS + NUM_COLS]
y = df_sal[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ════════════════════════════════════════════════════════════
# 4. PREPROCESSING PIPELINES
# ════════════════════════════════════════════════════════════

# --- TF-IDF for job title ---
tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    ))
])

# --- Categorical ---
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# --- Numeric ---
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("text", tfidf_pipe, TITLE_COL),
        ("cat",  cat_pipe,   CAT_COLS),
        ("num",  num_pipe,   NUM_COLS),
    ],
    remainder="drop",
    sparse_threshold=0,
    n_jobs=-1,
)

# ════════════════════════════════════════════════════════════
# 5. BUILD & TRAIN MODELS
# ════════════════════════════════════════════════════════════
# For Ridge we work in log-salary space to handle right-skew
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

models = {
    "Ridge": Pipeline([
        ("prep",  preprocessor),
        ("model", Ridge(alpha=10.0)),
    ]),
    "RandomForest": Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=25,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    "HistGradientBoosting": Pipeline([
        ("prep",  preprocessor),
        ("model", HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=25,
            min_samples_leaf=2,
            random_state=42
        )),
    ]),
}

# Log-transform y for Ridge (fit on log, predict then expm1)
y_log = np.log1p(y_train)

metrics_rows = []
predictions  = {}

for name, pipe in models.items():
    print(f"\nTraining {name} …")

    if name == "Ridge":
        # Train on log-salary
        pipe.fit(X_train, y_log)
        y_pred_log = pipe.predict(X_test)
        y_pred = np.expm1(y_pred_log)   # back to dollar scale
    else:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE  : ${mae:>10,.0f}")
    print(f"  RMSE : ${rmse:>10,.0f}")
    print(f"  R²   : {r2:.4f}")

    metrics_rows.append({
        "model": name,
        "MAE":   round(mae, 2),
        "RMSE":  round(rmse, 2),
        "R2":    round(r2, 4),
    })
    predictions[name] = y_pred

# ════════════════════════════════════════════════════════════
# 6. PICK BEST MODEL & SAVE
# ════════════════════════════════════════════════════════════
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(TABLES_DIR / "salary_model_metrics.csv", index=False)
print(f"\n✓ Saved salary_model_metrics.csv")

# Best model by R²
best_name = metrics_df.loc[metrics_df["R2"].idxmax(), "model"]
best_pipe  = models[best_name]
print(f"\n  Best model: {best_name}  (R²={metrics_df.loc[metrics_df['model']==best_name,'R2'].values[0]:.4f})")

joblib.dump(best_pipe, MODELS_DIR / "salary_model.pkl")
print(f"✓ Saved salary_model.pkl  ({best_name})")

# ════════════════════════════════════════════════════════════
# 7. SALARY PREDICTIONS TABLE
# ════════════════════════════════════════════════════════════
# Re-predict on full salary-known set for export
ridge_log_pred = models["Ridge"].predict(X)
df_sal["pred_salary_ridge"] = np.expm1(ridge_log_pred)
df_sal["pred_salary_rf"]    = models["RandomForest"].predict(X)
df_sal["pred_salary_hgb"]   = models["HistGradientBoosting"].predict(X)
df_sal["pred_salary_best"]  = np.expm1(models[best_name].predict(X)) if best_name == "Ridge" else models[best_name].predict(X)
df_sal["residual_best"]     = df_sal[TARGET] - df_sal["pred_salary_best"]

export_cols = (
    ["job_id", TARGET, "pred_salary_ridge", "pred_salary_rf", "pred_salary_hgb",
     "pred_salary_best", "residual_best"]
    + [c for c in ["title", "formatted_experience_level",
                   "primary_industry_name", "is_remote"] if c in df_sal.columns]
)
df_export = df_sal[[c for c in export_cols if c in df_sal.columns]]
df_export.to_csv(PRO_DIR / "salary_predictions.csv", index=False)
print(f"✓ Saved salary_predictions.csv  shape={df_export.shape}")

# ════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE (RandomForest only)
# ════════════════════════════════════════════════════════════
print("\n=== TOP 20 FEATURE IMPORTANCES (RandomForest) ===")
rf_model  = models["RandomForest"].named_steps["model"]
prep_step = models["RandomForest"].named_steps["prep"]

# Collect feature names from ColumnTransformer
try:
    feat_names = prep_step.get_feature_names_out()
    fi = pd.Series(rf_model.feature_importances_, index=feat_names).sort_values(ascending=False)
    print(fi.head(20).to_string())

    fi_df = fi.reset_index()
    fi_df.columns = ["feature", "importance"]
    fi_df.to_csv(TABLES_DIR / "rf_feature_importance.csv", index=False)
    print("\n✓ Saved rf_feature_importance.csv")
except Exception as e:
    print(f"  (Could not extract feature names: {e})")

# ════════════════════════════════════════════════════════════
# 9. FINAL PRINT
# ════════════════════════════════════════════════════════════
print("\n=== MODEL COMPARISON ===")
print(metrics_df.to_string(index=False))
print(f"\nBest model saved: {best_name}")
