import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# SETUP
# ============================================================

INPUT_FILE = "final_comparison.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Ensure numeric
num_cols = [
    "LP_Radius", "charikar_radius", "gonzalez_radius",
    "SimplifiedDRG_radius", "OverclusterDRG_radius"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# RATIOS
# ============================================================

df["char_vs_lp"] = df["charikar_radius"] / df["LP_Radius"]
df["gonz_vs_lp"] = df["gonzalez_radius"] / df["LP_Radius"]
df["sdrg_vs_lp"] = df["SimplifiedDRG_radius"] / df["LP_Radius"]
df["ocdrg_vs_lp"] = df["OverclusterDRG_radius"] / df["LP_Radius"]

df["sdrg_vs_char"] = df["SimplifiedDRG_radius"] / df["charikar_radius"]
df["ocdrg_vs_char"] = df["OverclusterDRG_radius"] / df["charikar_radius"]

valid_lp = df.dropna(subset=["LP_Radius"])

# ============================================================
# 1️⃣ RADIUS VS N
# ============================================================

g = sns.FacetGrid(df, col="k", hue="outlier_percent", height=4, aspect=1.2)

g.map_dataframe(sns.lineplot, x="N", y="charikar_radius", marker="o")
g.map_dataframe(sns.lineplot, x="N", y="gonzalez_radius", linestyle="--", marker="s")
g.map_dataframe(sns.lineplot, x="N", y="SimplifiedDRG_radius", linestyle="-.", marker="^")
g.map_dataframe(sns.lineplot, x="N", y="OverclusterDRG_radius", linestyle=":", marker="D")

g.add_legend(title="Outlier %")
g.set_axis_labels("N", "Radius")
g.fig.suptitle("Radius vs N (All Algorithms)", y=1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "radius_with_drg.png"))
plt.close()

# ============================================================
# 2️⃣ APPROX RATIO VS LP
# ============================================================

g = sns.FacetGrid(valid_lp, col="k", hue="outlier_percent", height=4, aspect=1.2)

g.map_dataframe(sns.lineplot, x="N", y="char_vs_lp", marker="o")
g.map_dataframe(sns.lineplot, x="N", y="gonz_vs_lp", linestyle="--", marker="s")
g.map_dataframe(sns.lineplot, x="N", y="sdrg_vs_lp", linestyle="-.", marker="^")
g.map_dataframe(sns.lineplot, x="N", y="ocdrg_vs_lp", linestyle=":", marker="D")

g.add_legend(title="Outlier %")
g.set_axis_labels("N", "Approx Ratio vs LP")
g.fig.suptitle("Approximation Ratio vs LP (All Methods)", y=1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "approx_ratio_with_drg.png"))
plt.close()

# ============================================================
# 3️⃣ RUNTIME SCALING
# ============================================================

g = sns.FacetGrid(df, col="k", hue="outlier_percent", height=4, aspect=1.2)

g.map_dataframe(sns.lineplot, x="N", y="charikar_time", marker="o")
g.map_dataframe(sns.lineplot, x="N", y="gonzalez_time", marker="s")
g.map_dataframe(sns.lineplot, x="N", y="SimplifiedDRG_time", marker="^")
g.map_dataframe(sns.lineplot, x="N", y="OverclusterDRG_time", marker="D")

for ax in g.axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")

g.add_legend(title="Outlier %")
g.set_axis_labels("N (log)", "Runtime (log)")
g.fig.suptitle("Runtime Scaling (All Algorithms)", y=1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "runtime_with_drg.png"))
plt.close()

# ============================================================
# 4️⃣ TRADEOFF (BEST PLOT FOR PAPER)
# ============================================================

plt.figure(figsize=(7,6))

plt.scatter(df["charikar_time"], df["charikar_radius"], label="Charikar", alpha=0.7)
plt.scatter(df["gonzalez_time"], df["gonzalez_radius"], label="Gonzalez", alpha=0.5)
plt.scatter(df["SimplifiedDRG_time"], df["SimplifiedDRG_radius"], label="SimplifiedDRG", alpha=0.7)
plt.scatter(df["OverclusterDRG_time"], df["OverclusterDRG_radius"], label="OverclusterDRG", alpha=0.7)

plt.xscale("log")
plt.xlabel("Runtime (log)")
plt.ylabel("Radius")
plt.title("Quality vs Runtime Trade-off")

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tradeoff_all.png"))
plt.close()

print("✅ All DRG plots saved.")