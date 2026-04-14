"""
03_eda.py
Exploratory Data Analysis on jobs_master.csv
Saves charts to outputs/figures/ and tables to outputs/tables/
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ── Style Config (Bright High-Contrast Theme) ──────────────────
BG        = "#ffffff"
PANEL     = "#eff9ff"
ACCENT    = "#184683"
ACCENT2   = "#003859"
ACCENT3   = "#184683"
GRAY      = "#555555"
TEXT      = "#003859"
GRID      = "#c4d8e2"

PALETTE   = ["#184683", "#003859", "#1c5a91", "#337ab7", "#4d88ff",
             "#154360", "#21618c", "#2e86c1", "#5dade2", "#85c1e9"]

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     PANEL,
    "axes.edgecolor":     TEXT,
    "axes.labelcolor":    TEXT,
    "axes.titlecolor":    TEXT,
    "axes.titlesize":     15,
    "axes.labelsize":     12,
    "axes.titlepad":      15,
    "axes.grid":          True,
    "grid.color":         GRID,
    "grid.linewidth":     0.8,
    "xtick.color":        TEXT,
    "ytick.color":        TEXT,
    "text.color":         TEXT,
    "legend.facecolor":   BG,
    "legend.edgecolor":   TEXT,
    "legend.labelcolor":  TEXT,
    "font.family":        "DejaVu Sans",
    "figure.dpi":         150,
})

FIG   = config.FIGURES_DIR
TBL   = config.TABLES_DIR


def savefig(name):
    plt.tight_layout()
    path = FIG / name
    plt.savefig(path, bbox_inches="tight", facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  ✓ Saved {name}")
    return path


# ── Load data ─────────────────────────────────────────────────
print("Loading jobs_master.csv …")
df = pd.read_csv(config.DATA_PROCESSED_DIR / "jobs_master.csv", low_memory=False)
print(f"  Shape: {df.shape}")

# Salary subset (only known)
sal = df[df["normalized_salary"].notna() & (df["normalized_salary"] > 0)].copy()
print(f"  Rows with salary: {len(sal):,}")


# ══════════════════════════════════════════════════════════════
# 1. SALARY DISTRIBUTION
# ══════════════════════════════════════════════════════════════
print("\n[1] salary_distribution.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Salary Distribution (Annual, USD)", fontsize=16, color=TEXT, y=1.02)

# Histogram
ax = axes[0]
bins = np.linspace(sal["normalized_salary"].quantile(0.01),
                   sal["normalized_salary"].quantile(0.99), 50)
ax.hist(sal["normalized_salary"], bins=bins, color=ACCENT, alpha=0.85, edgecolor=BG, linewidth=0.4)
median_sal = sal["normalized_salary"].median()
ax.axvline(median_sal, color=ACCENT2, linewidth=3, linestyle="--",
           label=f"Median ${median_sal:,.0f}")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
ax.set_xlabel("Annual Salary")
ax.set_ylabel("Job Count")
ax.set_title("Salary Histogram")
ax.legend()

# Box by salary_band
ax2 = axes[1]
band_order = ["Low (<40k)", "Mid (40-80k)", "High (80-150k)", "Very High (>150k)"]
band_data = [sal[sal["salary_band"] == b]["normalized_salary"].dropna().values
             for b in band_order]
bp = ax2.boxplot(band_data, patch_artist=True, medianprops={"color": "#ffffff", "linewidth": 3},
                 whiskerprops={"color": GRAY}, capprops={"color": GRAY},
                 flierprops={"marker": ".", "color": GRAY, "alpha": 0.3, "markersize": 2})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels([b.split(" ")[0] for b in band_order], rotation=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
ax2.set_title("Salary Bands")
ax2.set_ylabel("Annual Salary")

savefig("salary_distribution.png")


# ══════════════════════════════════════════════════════════════
# 2. TOP INDUSTRIES
# ══════════════════════════════════════════════════════════════
print("[2] top_industries.png")
top_n = 15
ind_grp = (
    df[df["primary_industry_name"].notna()]
    .groupby("primary_industry_name")
    .agg(
        job_count=("job_id", "count"),
        avg_salary=("normalized_salary", "mean"),
        median_salary=("normalized_salary", "median"),
    )
    .sort_values("job_count", ascending=False)
    .head(top_n)
    .reset_index()
)
ind_grp.to_csv(TBL / "top_industries.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(ind_grp["primary_industry_name"][::-1],
               ind_grp["job_count"][::-1],
               color=PALETTE[:top_n][::-1], alpha=0.85, edgecolor=BG, linewidth=0.3)
for bar, val in zip(bars, ind_grp["job_count"][::-1]):
    ax.text(bar.get_width() + 120, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", color=TEXT, fontsize=9)
ax.set_xlabel("Number of Job Postings")
ax.set_title(f"Top {top_n} Industries by Job Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
savefig("top_industries.png")


# ══════════════════════════════════════════════════════════════
# 3. SALARY BY WORK TYPE
# ══════════════════════════════════════════════════════════════
print("[3] salary_by_worktype.png")
wt_grp = (
    sal[sal["formatted_work_type"].notna()]
    .groupby("formatted_work_type")
    .agg(
        job_count=("job_id", "count"),
        median_salary=("normalized_salary", "median"),
        mean_salary=("normalized_salary", "mean"),
        p25=("normalized_salary", lambda x: x.quantile(0.25)),
        p75=("normalized_salary", lambda x: x.quantile(0.75)),
    )
    .sort_values("median_salary", ascending=False)
    .reset_index()
)
wt_grp.to_csv(TBL / "salary_by_worktype.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(wt_grp))
bars = ax.bar(x, wt_grp["median_salary"], color=ACCENT, alpha=0.85,
              edgecolor=BG, linewidth=0.4, label="Median Salary")
ax.errorbar(x, wt_grp["median_salary"],
            yerr=[wt_grp["median_salary"] - wt_grp["p25"],
                  wt_grp["p75"] - wt_grp["median_salary"]],
            fmt="none", color=ACCENT2, linewidth=1.5, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(wt_grp["formatted_work_type"], rotation=20, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
ax.set_ylabel("Annual Salary (USD)")
ax.set_title("Median Salary by Work Type  (error bars = IQR)")
for bar, row in zip(bars, wt_grp.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f"n={row.job_count:,}", ha="center", va="bottom", fontsize=7.5, color=GRAY)
savefig("salary_by_worktype.png")


# ══════════════════════════════════════════════════════════════
# 4. SALARY BY EXPERIENCE LEVEL
# ══════════════════════════════════════════════════════════════
print("[4] salary_by_experience.png")
exp_col = "formatted_experience_level"
exp_order = ["Internship", "Entry Level", "Associate", "Mid-Senior Level",
             "Director", "Executive"]

exp_grp = (
    sal[sal[exp_col].notna()]
    .groupby(exp_col)
    .agg(
        job_count=("job_id", "count"),
        median_salary=("normalized_salary", "median"),
        mean_salary=("normalized_salary", "mean"),
        p25=("normalized_salary", lambda x: x.quantile(0.25)),
        p75=("normalized_salary", lambda x: x.quantile(0.75)),
    )
    .reindex([e.title() for e in exp_order if e.title() in
              sal[sal[exp_col].notna()][exp_col].unique()])
    .dropna(subset=["median_salary"])
    .reset_index()
)
# If reindex returns nothing, fall back to natural sort
if exp_grp.empty:
    exp_grp = (
        sal[sal[exp_col].notna()]
        .groupby(exp_col)
        .agg(
            job_count=("job_id", "count"),
            median_salary=("normalized_salary", "median"),
            mean_salary=("normalized_salary", "mean"),
            p25=("normalized_salary", lambda x: x.quantile(0.25)),
            p75=("normalized_salary", lambda x: x.quantile(0.75)),
        )
        .sort_values("median_salary")
        .reset_index()
    )

exp_grp.to_csv(TBL / "salary_by_experience.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 5))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(exp_grp))]
bars = ax.bar(exp_grp[exp_col], exp_grp["median_salary"],
              color=colors, alpha=0.85, edgecolor=BG, linewidth=0.4)
ax.errorbar(np.arange(len(exp_grp)), exp_grp["median_salary"],
            yerr=[exp_grp["median_salary"] - exp_grp["p25"],
                  exp_grp["p75"] - exp_grp["median_salary"]],
            fmt="none", color=ACCENT2, linewidth=1.5, capsize=5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
ax.set_ylabel("Annual Salary (USD)")
ax.set_title("Median Salary by Experience Level  (error bars = IQR)")
ax.set_xticks(np.arange(len(exp_grp)))
ax.set_xticklabels(exp_grp[exp_col], rotation=20, ha="right")
for bar, row in zip(bars, exp_grp.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"n={row.job_count:,}", ha="center", va="bottom", fontsize=7.5, color=GRAY)
savefig("salary_by_experience.png")


# ══════════════════════════════════════════════════════════════
# 5. TOP COMPANIES BY JOB POSTINGS
# ══════════════════════════════════════════════════════════════
print("[5] top_companies.png")
comp_col = "name" if "name" in df.columns else "company_name"
top_comp = (
    df[df[comp_col].notna()]
    .groupby(comp_col)
    .agg(
        job_count=("job_id", "count"),
        avg_salary=("normalized_salary", "mean"),
        median_salary=("normalized_salary", "median"),
    )
    .sort_values("job_count", ascending=False)
    .head(20)
    .reset_index()
)
top_comp.to_csv(TBL / "top_companies.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top_comp[comp_col][::-1], top_comp["job_count"][::-1],
               color=PALETTE[:20][::-1], alpha=0.85, edgecolor=BG, linewidth=0.3)
for bar, val in zip(bars, top_comp["job_count"][::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9, color=TEXT)
ax.set_xlabel("Number of Job Postings")
ax.set_title("Top 20 Companies by Job Count")
savefig("top_companies.png")


# ══════════════════════════════════════════════════════════════
# 6. SALARY BY REMOTE STATUS
# ══════════════════════════════════════════════════════════════
print("[6] salary_by_remote.png")
remote_sal = sal[sal["is_remote"].notna()].copy()
remote_sal["Remote"] = remote_sal["is_remote"].map({True: "Remote", False: "On-site"})

remote_grp = (
    remote_sal.groupby("Remote")
    .agg(
        job_count=("job_id", "count"),
        median_salary=("normalized_salary", "median"),
        mean_salary=("normalized_salary", "mean"),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Remote vs On-site Job Analysis", fontsize=15, color=TEXT)

# Left: job count pie
ax = axes[0]
wedge_colors = [ACCENT, ACCENT3]
wedges, texts, autotexts = ax.pie(
    remote_grp["job_count"], labels=remote_grp["Remote"],
    colors=wedge_colors, autopct="%1.1f%%", startangle=90,
    textprops={"color": TEXT}, pctdistance=0.7
)
for at in autotexts:
    at.set_color(BG)
    at.set_fontweight("bold")
ax.set_title("Job Count Split")

# Right: median salary bar
ax2 = axes[1]
bars = ax2.bar(remote_grp["Remote"], remote_grp["median_salary"],
               color=wedge_colors, alpha=0.85, edgecolor=BG, linewidth=0.4)
for bar, val in zip(bars, remote_grp["median_salary"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f"${val:,.0f}", ha="center", color=TEXT, fontsize=11, fontweight="bold")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
ax2.set_ylabel("Median Annual Salary")
ax2.set_title("Median Salary Comparison")

savefig("salary_by_remote.png")


# ══════════════════════════════════════════════════════════════
# 7. JOBS BY MONTH
# ══════════════════════════════════════════════════════════════
print("[7] jobs_by_month.png")
monthly = (
    df[df["listed_month"].notna() & (df["listed_month"] != "NaT")]
    .groupby("listed_month")
    .agg(job_count=("job_id", "count"))
    .reset_index()
    .sort_values("listed_month")
)
# Keep last 18 months max for readability
monthly = monthly.tail(18)

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(monthly))
ax.fill_between(x, monthly["job_count"], alpha=0.25, color=ACCENT)
ax.plot(x, monthly["job_count"], color=ACCENT, linewidth=2.5, marker="o",
        markersize=5, markerfacecolor=ACCENT2, markeredgecolor=BG)
ax.set_xticks(x)
ax.set_xticklabels(monthly["listed_month"], rotation=40, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_ylabel("Job Postings")
ax.set_title("Monthly Job Posting Trend")
# Annotate peak
peak_idx = monthly["job_count"].idxmax()
peak_row = monthly.loc[peak_idx]
xi = x[monthly.index.get_loc(peak_idx)]
ax.annotate(f"Peak\n{peak_row.job_count:,.0f}",
            xy=(xi, peak_row.job_count),
            xytext=(xi + 0.8, peak_row.job_count * 0.92),
            color=ACCENT2, fontsize=9, fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": ACCENT2})
savefig("jobs_by_month.png")


# ══════════════════════════════════════════════════════════════
# CORRELATION SUMMARY
# ══════════════════════════════════════════════════════════════
print("[+] corr_summary.csv")
num_cols = ["normalized_salary", "views", "applies", "benefit_count",
            "skill_count", "description_length", "employee_count"]
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr(numeric_only=True).round(3)
corr.to_csv(TBL / "corr_summary.csv")
print("  ✓ Saved corr_summary.csv")

# ── Final report ──────────────────────────────────────────────
print("\n=== EDA COMPLETE ===")
print(f"Charts saved to  : {FIG}")
print(f"Tables saved to  : {TBL}")
print("\nKey stats for insights:")
print(f"  Median salary (all with salary): ${sal['normalized_salary'].median():,.0f}")
print(f"  Remote vs On-site counts:\n{df['is_remote'].value_counts().to_string()}")
if "primary_industry_name" in df.columns:
    print(f"\n  Top 3 industries:\n{df['primary_industry_name'].value_counts().head(3).to_string()}")
if comp_col in df.columns:
    print(f"\n  Top 3 companies:\n{df[comp_col].value_counts().head(3).to_string()}")
if "formatted_experience_level" in df.columns:
    avg_sal_by_exp = sal.groupby("formatted_experience_level")["normalized_salary"].median().sort_values(ascending=False)
    print(f"\n  Median salary by experience:\n{avg_sal_by_exp.to_string()}")
