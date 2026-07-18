import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, 'results')
PLOTS   = os.path.join(REPO, 'plots')
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
})

df  = pd.read_csv(os.path.join(RESULTS, 'explore2_results.csv'))
syn = pd.read_csv(os.path.join(RESULTS, 'explore2_synthetic.csv'))

O_cols = sorted([c for c in df.columns if c.startswith('r_O') and c[3:].isdigit()],
                key=lambda c: int(c[3:]))
O_vals = [int(c[3:]) for c in O_cols]

colors   = {'adult': '#2196F3', 'diabetes': '#4CAF50', 'covertype': '#FF9800'}
datasets = ['adult', 'diabetes', 'covertype']

# ── FIGURE 3.5: Radius vs O ablation — 3 panels vertical ─────────────────────
# Y-axis: radius normalised to O=z value; dashed lines at O=z and O=20
fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharey=False)
for ax, ds in zip(axes, datasets):
    d = df[df.dataset == ds]
    norms = []
    for _, row in d.iterrows():
        z = int(row['z'])
        z_col = f'r_O{z}'
        if z_col not in row.index or pd.isna(row[z_col]):
            continue
        r_at_z = row[z_col]
        vals = [row[c] / r_at_z if c in row.index and not pd.isna(row[c]) else np.nan
                for c in O_cols]
        norms.append(vals)
    if not norms:
        ax.set_title(ds.capitalize() + ' (no data)')
        continue
    norms = np.array(norms)
    mean_norm = np.nanmean(norms, axis=0)
    mean_z    = d['z'].mean()
    ax.plot(O_vals, mean_norm, color=colors[ds], lw=2, marker='o', markersize=3)
    ax.axhline(1.0,    color='red',    ls='--', lw=1.5, label=f'O=z  (mean z={mean_z:.0f})')
    ax.axvline(20,     color='orange', ls='--', lw=1.5, label='O=20 (old)')
    ax.axvline(mean_z, color='red',    ls=':',  lw=1,   alpha=0.6)
    ax.set_title(ds.capitalize(), fontweight='bold')
    ax.set_xlabel('Pool overcount O')
    ax.set_ylabel('Radius / Radius(O=z)')
    ax.set_xscale('log')
    ax.legend(fontsize=9)

fig.suptitle('Radius vs pool overcount O  (normalised to O=z)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig_O_ablation.png'), bbox_inches='tight')
plt.close()
print("fig_O_ablation.png")

# ── FIGURE 3.6: Knee O/z vs z/k ──────────────────────────────────────────────
markers = {'adult': 'o', 'diabetes': 's', 'covertype': '^'}
fig, ax = plt.subplots(figsize=(7, 5))
for ds in datasets:
    d = df[df.dataset == ds].dropna(subset=['knee_over_z'])
    ax.scatter(d['z_over_k'], d['knee_over_z'],
               label=ds, marker=markers[ds], color=colors[ds], alpha=0.8, s=70)
ax.axhline(1.0, color='red', ls=':', lw=1.5, label='O=z threshold')
ax.set_xlabel('z/k  (outliers per center)')
ax.set_ylabel('Knee O / z')
ax.set_title('What determines optimal O?\nStrongest predictor: z/k', fontweight='bold')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig_O_vs_zk.png'), bbox_inches='tight')
plt.close()
print("fig_O_vs_zk.png")

# ── FIGURE 3.4 (extra): O=20 gap distribution ────────────────────────────────
if 'gap_O20_vs_opt' in df.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    for ds in datasets:
        d = df[df.dataset == ds].dropna(subset=['gap_O20_vs_opt'])
        ax.hist(d['gap_O20_vs_opt'], bins=12, alpha=0.6, label=ds,
                color=colors[ds], edgecolor='white')
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Gap: (radius at O=20 − opt) / opt  (%)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of O=20 sub-optimality gap', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig_O20_gap_distribution.png'), bbox_inches='tight')
    plt.close()
    print("fig_O20_gap_distribution.png")

# ── FIGURE 3.7: Synthetic experiments — 3 panels vertical ────────────────────
fig, axes = plt.subplots(3, 1, figsize=(7, 12))

# Panel 1: vary z/k
d = syn[syn.exp == 'syn_vary_zk'].sort_values('z_over_k')
axes[0].scatter(d['z_over_k'], d['knee_over_z'], color='#2196F3', s=80)
axes[0].axhline(1.0, color='red', ls='--', lw=1.5, label='O=z')
axes[0].set_xlabel('z/k  (outliers per center)')
axes[0].set_ylabel('Knee O / z')
axes[0].set_title('Varying z/k  (N=2000, d=6, sep=3)', fontweight='bold')
axes[0].legend()

# Panel 2: vary dimensionality
d = syn[syn.exp == 'syn_vary_d'].sort_values('d')
axes[1].plot(d['d'], d['knee_over_z'], 'g-o', lw=2)
axes[1].axhline(1.0, color='red', ls='--', lw=1.5)
axes[1].set_xlabel('Dimensionality d')
axes[1].set_ylabel('Knee O / z')
axes[1].set_title('Varying dimensionality  (N=2000, k=20, z=200)', fontweight='bold')

# Panel 3: vary cluster separation
sep_col = 'sep' if 'sep' in syn.columns else 'separation'
d = syn[syn.exp == 'syn_vary_sep']
if sep_col in d.columns:
    d = d.sort_values(sep_col)
    x_sep = d[sep_col]
else:
    x_sep = range(len(d))
axes[2].plot(x_sep, d['knee_over_z'].values, 'm-o', lw=2)
axes[2].axhline(1.0, color='red', ls='--', lw=1.5)
axes[2].set_xlabel('Cluster separation')
axes[2].set_ylabel('Knee O / z')
axes[2].set_title('Varying cluster separation  (N=2000, k=20, z=200, d=6)',
                  fontweight='bold')

fig.suptitle('Synthetic experiments: z/k is strongest predictor; d matters for high-d',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig_synthetic.png'), bbox_inches='tight')
plt.close()
print("fig_synthetic.png")

# ── Phase transition traces ───────────────────────────────────────────────────
configs_to_show = [
    ('adult',    2000, 20, 0.10, 'Adult: N=2000, k=20, z=200'),
    ('diabetes', 5000, 20, 0.10, 'Diabetes: N=5000, k=20, z=500'),
    ('covertype',5000, 50, 0.20, 'Covertype: N=5000, k=50, z=1000'),
]
op_col = 'outlier_percent' if 'outlier_percent' in df.columns else 'z_over_N'

fig, axes = plt.subplots(3, 1, figsize=(7, 12))
for ax, (ds, N, k, op, title) in zip(axes, configs_to_show):
    rows = df[(df.dataset == ds) & (df.N == N) & (df.k == k) &
              (np.isclose(df[op_col], op))]
    if rows.empty:
        ax.set_title(title + ' (no data)', fontsize=10)
        continue
    row = rows.iloc[0]
    z   = int(row['z'])
    O_plot, r_plot = [], []
    for c in O_cols:
        if c in row.index and not pd.isna(row[c]):
            O_plot.append(int(c[3:]))
            r_plot.append(row[c])
    ax.plot(O_plot, r_plot, 'b-o', lw=2, markersize=4)
    ax.axvline(z,  color='red',    ls='--', lw=2,   label=f'O=z={z}')
    ax.axvline(20, color='orange', ls=':',  lw=2,   label='O=20 (old)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('O')
    ax.set_ylabel('Radius')
    ax.set_xscale('log')
    ax.legend(fontsize=9)

fig.suptitle('Pool overcount O vs radius: phase transition at O=z',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig_phase_transition.png'), bbox_inches='tight')
plt.close()
print("fig_phase_transition.png")

print(f"\nAll plots saved to {PLOTS}")
