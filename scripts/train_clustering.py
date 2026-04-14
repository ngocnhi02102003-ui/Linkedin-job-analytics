"""
train_clustering.py
Phase 3b: ML — K-Means Clustering.
Verbose output: Feature Selection → Scaling → Elbow → Train → Profile.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

sys.path.append(str(Path(__file__).resolve().parent))
import config

# ════════════════════════════════════════════════════════════
# STEP 1: PREPARE DATA
# ════════════════════════════════════════════════════════════
def step1_prepare():
    print("\n" + "="*60)
    print("CLUSTER STEP 1: PREPARE DATA")
    print("="*60)
    
    df = pd.read_csv(config.DATA_PROCESSED_DIR / "jobs_master.csv")
    print(f"  Loaded: {len(df):,} rows")
    
    features = ["normalized_salary", "skill_count", "benefit_count",
                 "engagement_score", "is_remote", "is_sponsored"]
    
    print(f"\n  Clustering features: {features}")
    
    X = df[features].copy()
    
    # Impute missing (salary nulls get median)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    print(f"  After imputation: {X_imputed.shape}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"  After scaling: {X_scaled.shape}")
    
    # Stats
    print(f"\n  Feature stats (pre-scale):")
    print(f"    {'Feature':<25} {'Mean':>12} {'Std':>12}")
    print("    " + "-"*50)
    for i, f in enumerate(features):
        print(f"    {f:<25} {X_imputed[:, i].mean():>12,.2f} {X_imputed[:, i].std():>12,.2f}")
    
    return df, X_scaled, features

# ════════════════════════════════════════════════════════════
# STEP 2: FIND OPTIMAL K (Elbow + Silhouette)
# ════════════════════════════════════════════════════════════
def step2_find_k(X_scaled):
    print("\n" + "="*60)
    print("CLUSTER STEP 2: FIND OPTIMAL K")
    print("="*60)
    
    k_range = range(2, 8)
    inertias = []
    silhouettes = []
    
    print(f"\n  {'k':>5} {'Inertia':>15} {'Silhouette':>12}")
    print("  " + "-"*35)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        silhouettes.append(sil)
        print(f"  {k:>5} {km.inertia_:>15,.0f} {sil:>12.4f}")
    
    # Find best k by silhouette
    best_idx = np.argmax(silhouettes)
    best_k = list(k_range)[best_idx]
    print(f"\n  ★ Optimal k = {best_k} (Silhouette = {silhouettes[best_idx]:.4f})")
    
    # Elbow chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias, "bo-")
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.axvline(x=best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax1.legend()
    
    ax2.plot(list(k_range), silhouettes, "go-")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Score")
    ax2.axvline(x=best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax2.legend()
    
    fig.suptitle("K-Means: Optimal Cluster Selection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "elbow_curve.png", dpi=150)
    plt.close(fig)
    print("  ✓ Saved elbow_curve.png")
    
    return best_k

# ════════════════════════════════════════════════════════════
# STEP 3: TRAIN K-MEANS
# ════════════════════════════════════════════════════════════
def step3_train(df, X_scaled, best_k):
    print("\n" + "="*60)
    print(f"CLUSTER STEP 3: TRAIN K-MEANS (k={best_k})")
    print("="*60)
    
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, df["cluster"], sample_size=min(5000, len(X_scaled)))
    print(f"  Final Silhouette Score: {sil:.4f}")
    
    print(f"\n  Cluster Distribution:")
    print(f"    {'Cluster':>10} {'Count':>10} {'Pct':>8}")
    print("    " + "-"*30)
    for c, cnt in df["cluster"].value_counts().sort_index().items():
        pct = cnt / len(df) * 100
        print(f"    {c:>10} {cnt:>10,} {pct:>7.1f}%")
    
    # Save metrics
    metrics = pd.DataFrame([{"model": "K-Means", "k": best_k, "silhouette_score": sil}])
    metrics.to_csv(config.METRICS_DIR / "clustering_metrics.csv", index=False)
    print("  ✓ Saved metrics/clustering_metrics.csv")
    
    return df

# ════════════════════════════════════════════════════════════
# STEP 4: PROFILE CLUSTERS
# ════════════════════════════════════════════════════════════
def step4_profile(df, features):
    print("\n" + "="*60)
    print("CLUSTER STEP 4: PROFILE CLUSTERS")
    print("="*60)
    
    profile_cols = ["normalized_salary", "skill_count", "benefit_count", 
                    "engagement_score", "is_remote", "has_salary", "views", "applies"]
    available = [c for c in profile_cols if c in df.columns]
    
    profile = df.groupby("cluster")[available].mean()
    
    print(f"\n  {'Metric':<22}", end="")
    for c in profile.index:
        print(f"  {'Cluster '+str(c):>12}", end="")
    print()
    print("  " + "-" * (22 + 14 * len(profile.index)))
    
    for col in available:
        print(f"  {col:<22}", end="")
        for c in profile.index:
            val = profile.loc[c, col]
            if col == "normalized_salary":
                print(f"  ${val:>10,.0f}", end="")
            elif col in ["is_remote", "has_salary"]:
                print(f"  {val:>11.1%}", end="")
            else:
                print(f"  {val:>12.1f}", end="")
        print()
    
    # Cluster Name Suggestion
    cluster_names = {}
    for c in profile.index:
        sal = profile.loc[c, "normalized_salary"]
        skills = profile.loc[c, "skill_count"]
        if sal > 100000:
            cluster_names[c] = "Premium / Expert"
        elif sal > 50000:
            cluster_names[c] = "Mid-Market / Professional"
        else:
            cluster_names[c] = "Entry / Volume"
    
    print(f"\n  Suggested Cluster Names:")
    for c, name in cluster_names.items():
        print(f"    Cluster {c}: {name}")
    
    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.boxplot(data=df, x="cluster", y="normalized_salary", hue="cluster", palette="Set2", legend=False, ax=axes[0])
    axes[0].set_title("Salary by Cluster")
    
    sns.barplot(data=df, x="cluster", y="skill_count", hue="cluster", palette="Set2", legend=False, ax=axes[1])
    axes[1].set_title("Avg Skill Count by Cluster")
    
    sns.barplot(data=df, x="cluster", y="engagement_score", hue="cluster", palette="Set2", legend=False, ax=axes[2])
    axes[2].set_title("Avg Engagement by Cluster")
    
    fig.suptitle("Cluster Profiles", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / "cluster_profiles.png", dpi=150)
    plt.close(fig)
    print("  ✓ Saved cluster_profiles.png")

# ════════════════════════════════════════════════════════════
# STEP 5: SAVE OUTPUTS
# ════════════════════════════════════════════════════════════
def step5_save(df):
    print("\n" + "="*60)
    print("CLUSTER STEP 5: SAVE OUTPUTS")
    print("="*60)
    
    df[["job_id", "cluster"]].to_csv(config.PBI_DIR / "cluster_results.csv", index=False)
    print(f"  ✓ Saved powerbi/cluster_results.csv ({len(df):,} rows)")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ML PIPELINE: K-Means Market Segmentation          ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    try:
        df, X_scaled, features = step1_prepare()
        best_k = step2_find_k(X_scaled)
        df = step3_train(df, X_scaled, best_k)
        step4_profile(df, features)
        step5_save(df)
        
        print("\n" + "="*60)
        print("✓ SUCCESS: ML Phase 3b (Clustering) Complete.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
