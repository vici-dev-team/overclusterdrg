"""Generate all paper plots."""
import sys,os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
sys.path.insert(0,'/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

plt.rcParams.update({'font.family':'serif','font.size':11,'axes.grid':True,
                     'grid.alpha':0.3,'figure.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
os.makedirs('paper/plots',exist_ok=True)

# ── PLOT 1: O ablation curve (the main finding) ──────────────────────────
df = pd.read_csv('explore2_results.csv')
O_cols = sorted([c for c in df.columns if c.startswith('r_O') and c[3:].isdigit()],
                key=lambda c: int(c[3:]))
O_vals = [int(c[3:]) for c in O_cols]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
colors = {'adult':'#2196F3','diabetes':'#4CAF50','covertype':'#FF9800'}
for ax, ds in zip(axes, ['adult','diabetes','covertype']):
    d = df[df.dataset==ds]
    # Average over all configs
    mean_r = [d[c].mean() for c in O_cols if c in d.columns]
    O_v    = [int(c[3:]) for c in O_cols if c in d.columns]
    # Normalize by O=z value (get per-config normalized then average)
    norms = []
    for _, row in d.iterrows():
        z = int(row.z)
        z_col = f'r_O{z}'
        if z_col not in row: continue
        r_at_z = row[z_col]
        vals = [row[c]/r_at_z if c in row and not pd.isna(row[c]) else np.nan
                for c in O_cols]
        norms.append(vals)
    norms = np.array(norms)
    mean_norm = np.nanmean(norms, axis=0)
    ax.plot(O_v, mean_norm, color=colors[ds], lw=2, marker='o', markersize=4)
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='O=z (reference)')
    ax.axvline(d['z'].mean(), color='gray', ls=':', lw=1.5, alpha=0.7, label=f'O=z (mean z={d.z.mean():.0f})')
    ax.set_title(ds.capitalize(), fontweight='bold')
    ax.set_xlabel('Pool overcount O')
    ax.set_ylabel('Radius / Radius(O=z)')
    ax.set_xscale('log')
    if ds == 'adult': ax.legend(fontsize=9)
fig.suptitle('Figure: Radius vs pool overcount O (normalized to O=z)\nO=z is near-optimal; O=20 is 15–50% worse',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('paper/plots/fig_O_ablation.png', bbox_inches='tight')
plt.close(); print("fig_O_ablation.png")

# ── PLOT 2: O/z ratio vs z/k (the predictor) ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for ds, marker, color in [('adult','o','#2196F3'),('diabetes','s','#4CAF50'),('covertype','^','#FF9800')]:
    d = df[df.dataset==ds].dropna(subset=['knee_over_z'])
    ax.scatter(d['z_over_k'], d['knee_over_z'], label=ds, marker=marker,
               color=color, alpha=0.7, s=60)
# Trend line
d_all = df.dropna(subset=['knee_over_z'])
z = np.polyfit(np.log(d_all.z_over_k+1), d_all.knee_over_z, 1)
x_fit = np.linspace(0, d_all.z_over_k.max(), 100)
ax.plot(x_fit, z[0]*np.log(x_fit+1)+z[1], 'k--', lw=1.5, label='log fit')
ax.axhline(1.0, color='red', ls=':', lw=1.5, label='O=z threshold')
ax.set_xlabel('z/k  (outliers per center)')
ax.set_ylabel('Knee O / z')
ax.set_title('Optimal pool size ratio vs z/k\nKnee at O ≈ z for most configs', fontweight='bold')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig('paper/plots/fig_O_vs_zk.png', bbox_inches='tight')
plt.close(); print("fig_O_vs_zk.png")

# ── PLOT 3: O=20 gap distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for ds, color in [('adult','#2196F3'),('diabetes','#4CAF50'),('covertype','#FF9800')]:
    d = df[df.dataset==ds].dropna(subset=['gap_O20_vs_opt'])
    ax.hist(d['gap_O20_vs_opt'], bins=12, alpha=0.6, label=ds, color=color, edgecolor='white')
ax.axvline(0, color='black', lw=1)
ax.set_xlabel('Gap: (radius at O=20 - optimal radius) / optimal  (%)')
ax.set_ylabel('Count')
ax.set_title('Distribution of O=20 sub-optimality gap\nMedian gap: 20-30%. Max: 64%.', fontweight='bold')
ax.legend()
fig.tight_layout()
fig.savefig('paper/plots/fig_O20_gap_distribution.png', bbox_inches='tight')
plt.close(); print("fig_O20_gap_distribution.png")

# ── PLOT 4: Phase transition visualization ────────────────────────────────
# Show specific curves for 3 representative configs
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
configs_to_show = [
    ('adult', 2000, 20, 0.10, 'Adult: N=2000,k=20,z=200'),
    ('diabetes', 5000, 20, 0.10, 'Diabetes: N=5000,k=20,z=500'),
    ('covertype', 5000, 50, 0.20, 'Covertype: N=5000,k=50,z=1000'),
]
for ax, (ds, N, k, op, title) in zip(axes, configs_to_show):
    row = df[(df.dataset==ds)&(df.N==N)&(df.k==k)&(np.isclose(df.outlier_percent,op))].iloc[0]
    z = int(row.z)
    O_v_plot = []; r_v_plot = []
    for c in O_cols:
        if c in row and not pd.isna(row[c]):
            O_v_plot.append(int(c[3:])); r_v_plot.append(row[c])
    ax.plot(O_v_plot, r_v_plot, 'b-o', lw=2, markersize=4)
    ax.axvline(z, color='red', ls='--', lw=2, label=f'O=z={z}')
    ax.axvline(20, color='orange', ls=':', lw=2, label='O=20 (old)')
    ax.scatter([20],[row.get('r_O20',np.nan)], color='orange', s=100, zorder=5)
    ax.scatter([z],[row.get(f'r_O{z}',np.nan)], color='red', s=100, zorder=5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('O'); ax.set_ylabel('Radius')
    ax.set_xscale('log')
    if ds=='adult': ax.legend(fontsize=9)
fig.suptitle('Pool overcount O vs radius: phase transition at O=z', fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('paper/plots/fig_phase_transition.png', bbox_inches='tight')
plt.close(); print("fig_phase_transition.png")

# ── PLOT 5: Synthetic - z/k effect ───────────────────────────────────────
syn = pd.read_csv('explore2_synthetic.csv')
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# z/k ratio
d = syn[syn.exp=='syn_vary_zk'].sort_values('z_over_k')
axes[0].scatter(d.z_over_k, d.knee_over_z, color='#2196F3', s=80)
axes[0].axhline(1.0, color='red', ls='--', lw=1.5, label='O=z')
axes[0].set_xlabel('z/k (outliers per center)'); axes[0].set_ylabel('Knee O / z')
axes[0].set_title('Effect of z/k ratio\n(synthetic, N=2000, d=6)', fontweight='bold')
axes[0].legend()

# dimensionality
d = syn[syn.exp=='syn_vary_d'].sort_values('d')
axes[1].plot(d.d, d.knee_over_z, 'g-o', lw=2)
axes[1].axhline(1.0, color='red', ls='--', lw=1.5)
axes[1].set_xlabel('Dimensionality d'); axes[1].set_ylabel('Knee O / z')
axes[1].set_title('Effect of dimensionality\n(N=2000,k=20,z=200)', fontweight='bold')

# cluster separation
d = syn[syn.exp=='syn_vary_sep']
axes[2].plot(d['sep'] if 'sep' in d.columns else d.get('separation', range(len(d))),
             d.knee_over_z, 'm-o', lw=2)
axes[2].axhline(1.0, color='red', ls='--', lw=1.5)
axes[2].set_xlabel('Cluster separation'); axes[2].set_ylabel('Knee O / z')
axes[2].set_title('Effect of cluster separation\n(N=2000,k=20,z=200,d=6)', fontweight='bold')

fig.suptitle('What determines optimal O?', fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig('paper/plots/fig_synthetic_analysis.png', bbox_inches='tight')
plt.close(); print("fig_synthetic_analysis.png")

print("\nAll plots saved to paper/plots/")
