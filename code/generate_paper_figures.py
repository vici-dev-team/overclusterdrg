import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')
PLOTS   = os.path.join(REPO, 'plots')
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Load adult data ──────────────────────────────────────────────────────────
fc = pd.read_csv(os.path.join(RESULTS, 'final_comparison.csv'))
fc['p'] = fc['p'].astype(float)
fc['OC_LP']   = fc['OverclusterDRG_radius'] / fc['LP_Radius']
fc['SDRG_LP'] = fc['SimplifiedDRG_radius']  / fc['LP_Radius']
fc['Char_LP'] = fc['charikar_radius']        / fc['LP_Radius']
fc['Gonz_LP'] = fc['gonzalez_radius']        / fc['LP_Radius']

ding = pd.read_csv(os.path.join(RESULTS, 'benchmark_ding_vs_drg.csv'))
ding['p'] = ding['outlier_pct'].astype(float)
fc = fc.merge(ding[['N', 'k', 'p', 'ding_radius', 'ding_time_s']],
              on=['N', 'k', 'p'], how='left')
fc['Ding_LP'] = fc['ding_radius'] / fc['LP_Radius']

# ── FIGURE 3.1: Quality–runtime tradeoff — 2 panels vertical ────────────────
# Top: 10% outliers, Bottom: 20% outliers
algs = [
    ('gonzalez_radius',       'gonzalez_time',       '#aaaaaa', 'o', 'Gonzalez'),
    ('charikar_radius',       'charikar_time',        '#ff7f0e', 's', 'Charikar'),
    ('SimplifiedDRG_radius',  'SimplifiedDRG_time',   '#17becf', 'D', 'OC-Forward'),
    ('OverclusterDRG_radius', 'OverclusterDRG_time',  '#2ca02c', 'D', 'OC-Local'),
]

fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharey=False)
for ax, op, label in zip(axes, [0.10, 0.20], ['Outlier rate 10%', 'Outlier rate 20%']):
    d = fc[np.isclose(fc['p'], op)].dropna(subset=['LP_Radius'])
    for rcol, tcol, color, marker, name in algs:
        sub = d.dropna(subset=[rcol, tcol])
        ax.scatter(sub[tcol], sub[rcol] / d.loc[sub.index, 'LP_Radius'],
                   label=name, color=color, marker=marker, s=60, alpha=0.8)
    ding_d = d.dropna(subset=['ding_radius', 'ding_time_s'])
    ax.scatter(ding_d['ding_time_s'],
               ding_d['ding_radius'] / ding_d['LP_Radius'],
               label='Ding', color='#9467bd', marker='^', s=60, alpha=0.8)
    lp_d = d.dropna(subset=['LP_Time'])
    ax.scatter(lp_d['LP_Time'], np.ones(len(lp_d)),
               label='LP', color='#d62728', marker='*', s=80, alpha=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('Runtime (s, log scale)')
    ax.set_ylabel('Radius / LP radius')
    ax.set_title(label, fontweight='bold')
    ax.set_ylim(bottom=0.98)
    ax.legend(fontsize=8, loc='upper right')

fig.suptitle('Quality–speed tradeoff  (lower-left = best)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig3_quality_speed.png'), bbox_inches='tight')
plt.close()
print("fig3_quality_speed.png")

# ── FIGURE 3.2: LP ratio vs N (Adult) — 2 panels vertical ───────────────────
fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharey=False)
for ax, op, label in zip(axes, [0.10, 0.20], ['Outlier rate 10%', 'Outlier rate 20%']):
    d = fc[np.isclose(fc['p'], op)].sort_values('N')
    mn = d.groupby('N')[['OC_LP', 'SDRG_LP', 'Char_LP', 'Ding_LP', 'Gonz_LP']].mean()

    ax.plot(mn.index, mn['Gonz_LP'],  color='#aaaaaa', marker='o', lw=1.5, ls='-.', label='Gonzalez')
    ax.plot(mn.index, mn['Char_LP'],  color='#ff7f0e', marker='s', lw=1.5, label='Charikar')
    ax.plot(mn.index, mn['SDRG_LP'],  color='#17becf', marker='D', lw=1.5, ls='--', label='OC-Forward')
    ax.plot(mn.index, mn['OC_LP'],    color='#2ca02c', marker='D', lw=2,   label='OC-Local')
    ax.plot(mn.index, mn['Ding_LP'],  color='#9467bd', marker='^', lw=1.5, ls=':', label='Ding')
    ax.axhline(1.0, color='#d62728', ls='--', lw=1.5, label='LP')
    ax.set_xlabel('N')
    ax.set_ylabel('Radius / LP radius')
    ax.set_title(label, fontweight='bold')
    ax.set_ylim(bottom=0.98)
    ax.legend(fontsize=8)

fig.suptitle('Approximation ratio vs dataset size  (averaged over k)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig1_ratio_vs_N.png'), bbox_inches='tight')
plt.close()
print("fig1_ratio_vs_N.png")

# ── FIGURE 3.3: LP ratio vs N — Diabetes + Covertype — 2 panels vertical ────
# Only LP, Ding, OC-Local, Gonzalez shown (Charikar too slow at these scales)
diab = pd.read_csv(os.path.join(RESULTS, 'multidata_diabetes_fixed.csv'))
cov  = pd.read_csv(os.path.join(RESULTS, 'multidata_covertype_fixed.csv'))

fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharey=False)
for ax, df_sec, title in [
    (axes[0], diab, 'Diabetes Indicators  (d=8)'),
    (axes[1], cov,  'UCI Covertype  (d=10)'),
]:
    df_sec['OC_LP']   = df_sec['OC_Local_radius']  / df_sec['LP_Radius']
    df_sec['Ding_LP'] = df_sec['DingRKC_radius']   / df_sec['LP_Radius']
    df_sec['Gonz_LP'] = df_sec['gonzalez_radius']  / df_sec['LP_Radius']
    mn = df_sec.groupby('N')[['OC_LP', 'Ding_LP', 'Gonz_LP']].mean()

    ax.axhline(1.0, color='#d62728', ls='--', lw=1.5, label='LP (oracle)')
    ax.plot(mn.index, mn['Ding_LP'],  color='#9467bd', marker='^', lw=1.5, label='Ding')
    ax.plot(mn.index, mn['OC_LP'],    color='#2ca02c', marker='D', lw=2,   label='OC-Local')
    ax.plot(mn.index, mn['Gonz_LP'],  color='#aaaaaa', marker='o', lw=1.5, ls='-.', label='Gonzalez')
    ax.set_xlabel('N')
    ax.set_ylabel('Approximation ratio  (avg. over k, z/N)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)

fig.suptitle('LP ratio vs N on secondary datasets', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig_ratio_both.png'), bbox_inches='tight')
plt.close()
print("fig_ratio_both.png")

# ── FIGURE 3.4: Local search vs exhaustive — 2 panels vertical ───────────────
ex = pd.read_csv(os.path.join(RESULTS, 'exhaustive_vs_local.csv'))
# gap = local_radius / exhaustive_radius (>1 means local is suboptimal)
if 'gap' in ex.columns:
    if ex['gap'].max() > 5:    # stored as percentage
        ex['ratio'] = 1 + ex['gap'] / 100
    else:
        ex['ratio'] = ex['gap']
else:
    ex['ratio'] = ex['local_radius'] / ex['opt_radius']

exact_pct = (ex['ratio'] < 1 + 1e-6).mean() * 100
nonzero   = ex[ex['ratio'] > 1 + 1e-6]
pool_size = ex['k'] + ex['z']

fig, axes = plt.subplots(2, 1, figsize=(7, 9))

# Top: gap vs pool size, coloured by k
for k_val, color, label in [(5, '#5bc8f5', 'k=5'), (10, '#4caf50', 'k=10')]:
    sub = ex[ex['k'] == k_val]
    axes[0].scatter(sub['k'] + sub['z'], sub.loc[sub.index, 'ratio'],
                    color=color, label=label, s=50, alpha=0.8)
axes[0].axhline(1.05, color='#e74c3c', ls='--', lw=1.5, label='gap=1.05')
axes[0].set_xlabel('Pool size  (k + z)')
axes[0].set_ylabel('Gap  (local / exhaustive)')
axes[0].set_title(f'Gap vs pool size\n(exact match in {exact_pct:.1f}% of instances)',
                  fontweight='bold')
axes[0].legend(fontsize=9)

# Bottom: distribution of non-exact gaps
axes[1].hist(nonzero['ratio'], bins=12, color='#9467bd', edgecolor='white', alpha=0.9)
mean_gap = nonzero['ratio'].mean() if len(nonzero) else 1.0
axes[1].axvline(mean_gap, color='black', ls='--', lw=1.5, label=f'mean={mean_gap:.4f}')
axes[1].set_xlabel('Gap value')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Distribution of non-exact gaps  (n={len(nonzero)}/400)',
                  fontweight='bold')
axes[1].legend(fontsize=9)

fig.suptitle('1-swap local search vs exhaustive search on pool Q',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, 'fig5_local_vs_exhaustive.png'), bbox_inches='tight')
plt.close()
print("fig5_local_vs_exhaustive.png")

print(f"\nAll paper figures saved to {PLOTS}")
