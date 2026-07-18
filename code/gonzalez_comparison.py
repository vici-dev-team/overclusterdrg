import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
FILENAME = "gonzalez_kcenter_results_compared.csv"

try:
    df = pd.read_csv(FILENAME)
    print(f"Successfully loaded {len(df)} rows from {FILENAME}\n")
except FileNotFoundError:
    print(f"[!] Error: Could not find '{FILENAME}'. Make sure it is in the same directory.")
    exit()

# ------------------------------------------------------------
# 2. CALCULATE METRICS
# ------------------------------------------------------------
# How much smaller is the radius compared to the standard approach?
df['Det_Radius_Improvement_%'] = ((df['R_Standard'] - df['R_Deterministic']) / df['R_Standard']) * 100
df['Rand_Radius_Improvement_%'] = ((df['R_Standard'] - df['R_Randomized']) / df['R_Standard']) * 100

# Compare Randomized vs Deterministic 
# Ratio > 1 means Randomized is worse (larger radius). Ratio < 1 means Randomized is better.
df['Rand_vs_Det_Radius_Ratio'] = df['R_Randomized'] / df['R_Deterministic']
df['Rand_vs_Det_Time_Ratio'] = df['Time_Randomized'] / df['Time_Deterministic']

# ------------------------------------------------------------
# 3. PRINT TERMINAL REPORTS
# ------------------------------------------------------------
print("=== OVERALL AVERAGES ===")
print(f"Avg Deterministic Radius Improvement: {df['Det_Radius_Improvement_%'].mean():.2f}%")
print(f"Avg Randomized Radius Improvement:    {df['Rand_Radius_Improvement_%'].mean():.2f}%")
print(f"Avg Rand/Det Radius Ratio:            {df['Rand_vs_Det_Radius_Ratio'].mean():.3f} (1.0 = identical)")
print(f"Avg Rand/Det Time Penalty:            {df['Rand_vs_Det_Time_Ratio'].mean():.1f}x slower")
print("\n")

print("=== HOW RADIUS RATIO SCALES WITH K ===")
k_grouped = df.groupby('k')[['Rand_vs_Det_Radius_Ratio', 'Rand_vs_Det_Time_Ratio']].mean().reset_index()
print(k_grouped.to_string(index=False))
print("\n")

print("=== HOW ALGORITHMS SCALE WITH N (Time in seconds) ===")
n_grouped = df.groupby('N')[['Time_Standard', 'Time_Deterministic', 'Time_Randomized']].mean().reset_index()
print(n_grouped.to_string(index=False))

# ------------------------------------------------------------
# 4. GENERATE PUBLICATION CHARTS
# ------------------------------------------------------------
print("\nGenerating charts... close the chart windows to end the script.")

# Set up the figure for two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Chart 1: Radius Comparison by Dataset Size (for k=50, p=0.1) ---
# Filter data for a specific scenario to make the chart readable
subset_radius = df[(df['k'] == 50) & (df['Outlier_Percent'] == 0.1)]

bar_width = 0.25
x_indexes = np.arange(len(subset_radius['N']))

ax1.bar(x_indexes - bar_width, subset_radius['R_Standard'], width=bar_width, label='Standard Gonzalez', color='#ff9999', edgecolor='black')
ax1.bar(x_indexes, subset_radius['R_Deterministic'], width=bar_width, label='Deterministic (O+1)', color='#66b3ff', edgecolor='black')
ax1.bar(x_indexes + bar_width, subset_radius['R_Randomized'], width=bar_width, label='Randomized Pool', color='#99ff99', edgecolor='black')

ax1.set_title('Radius Comparison across Dataset Sizes\n(k=50, Outliers=10%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Dataset Size (N)', fontsize=11)
ax1.set_ylabel('Clustering Radius', fontsize=11)
ax1.set_xticks(x_indexes)
ax1.set_xticklabels(subset_radius['N'])
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# --- Chart 2: Time Scaling by Dataset Size (Averaged across k and p) ---
ax2.plot(n_grouped['N'], n_grouped['Time_Standard'], marker='o', linestyle='-', color='#ff9999', label='Standard Gonzalez', linewidth=2)
ax2.plot(n_grouped['N'], n_grouped['Time_Deterministic'], marker='s', linestyle='-', color='#66b3ff', label='Deterministic (O+1)', linewidth=2)
ax2.plot(n_grouped['N'], n_grouped['Time_Randomized'], marker='^', linestyle='-', color='#99ff99', label='Randomized Pool', linewidth=2)

ax2.set_title('Algorithm Runtime Scaling\n(Average across all k and outlier %)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Dataset Size (N)', fontsize=11)
ax2.set_ylabel('Time (Seconds)', fontsize=11)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
# Uncomment the line below if you want to automatically save the plot as an image
# plt.savefig('gonzalez_performance_charts.png', dpi=300)
plt.show()
