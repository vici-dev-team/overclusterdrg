"""
Revision experiments Part 2: A2 (rho), A3 (adaptive), A5/A6 (stats), A7 (OLS)
"""
import sys, os, ast, time, math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, pearsonr
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

BATCH = 1024

def robust_radius(X, centers_idx, O):
    n = len(X)
    min_d = np.full(n, np.inf, dtype=np.float64)
    for i in range(0, len(centers_idx), BATCH):
        batch = X[centers_idx[i:i+BATCH]]
        d = cdist(X, batch, metric='euclidean').min(axis=1)
        np.minimum(min_d, d, out=min_d)
    sd = np.sort(min_d)[::-1]
    return float(sd[O]) if O < n else 0.0

def gonzalez_with_distances(X, steps):
    n = len(X)
    mean = X.mean(axis=0, keepdims=True)
    c1 = int(cdist(X, mean, metric='euclidean').argmin())
    centers = [c1]
    min_d = cdist(X, X[[c1]], metric='euclidean').ravel()
    distances = []
    for _ in range(steps - 1):
        val = float(min_d.max())
        distances.append(val)
        nxt = int(min_d.argmax())
        centers.append(nxt)
        d_new = cdist(X, X[[nxt]], metric='euclidean').ravel()
        np.minimum(min_d, d_new, out=min_d)
    return centers, distances

def load_dataset(path):
    with open(path) as f:
        data = ast.literal_eval(f.read())
    return np.array(data, dtype=np.float64)

# Load data
print("Loading datasets...")
X_adult    = load_dataset('adult_final_dataset.py')
X_diabetes = load_dataset('diabetes_dataset.py')
X_cover    = load_dataset('covertype_dataset.py')
cert_df    = pd.read_csv('certified_ratios.csv')
explore_df = pd.read_csv('explore2_results.csv')
o_cols     = [c for c in explore_df.columns if c.startswith('r_O')]
o_vals     = sorted([int(c.split('_O')[1]) for c in o_cols])
print(f"  O values in ablation: {o_vals[:5]}...{o_vals[-3:]}")

# ============================================================
# A2 — ρ(O) per config
# ============================================================
print("\n" + "="*70)
print("A2: rho(O) OVERLAY — computing per config")
print("="*70)

rho_rows = []
# Cache gonzalez distances per (dataset, N, k, z)
cached_dists = {}

for _, row in explore_df.iterrows():
    N  = int(row['N']); k = int(row['k']); z = int(row['z'])
    ds = str(row['dataset']); knee_O = int(row['knee_O'])

    match = cert_df[(cert_df['dataset']==ds)&(cert_df['N']==N)&
                    (cert_df['k']==k)&(cert_df['z']==z)]
    if match.empty or pd.isna(match.iloc[0]['D_kz1']):
        continue
    D_kz1 = float(match.iloc[0]['D_kz1'])
    if D_kz1 <= 0: continue

    key = (ds, N, k, z)
    if key not in cached_dists:
        if ds == 'adult': X_full = X_adult
        elif ds == 'diabetes': X_full = X_diabetes
        else: X_full = X_cover
        X = X_full[:N]
        max_O_needed = max(o_vals)
        steps = min(k + max_O_needed + 2, N)
        _, dists = gonzalez_with_distances(X, steps)
        cached_dists[key] = dists
        print(f"  Cached {ds} N={N} k={k} z={z} ({len(dists)} distances)")

    dists = cached_dists[key]
    for O in o_vals:
        idx = k + O - 1   # dists[i] = D_{i+2}, D_{k+O+1} = dists[k+O-1]
        if idx < len(dists):
            D_kO1 = dists[idx]
            rho_O = D_kO1 / D_kz1
        else:
            D_kO1 = np.nan; rho_O = np.nan

        r_col = f'r_O{O}'
        r_val = float(row[r_col]) if r_col in row.index and not pd.isna(row.get(r_col, np.nan)) else np.nan
        rho_rows.append({'dataset':ds,'N':N,'k':k,'z':z,'knee_O':knee_O,
                         'O':O,'D_kO1':D_kO1,'rho_O':rho_O,'radius_at_O':r_val,
                         'radius_at_Oz': float(row['r_Oz']) if 'r_Oz' in row.index else np.nan})

rho_df = pd.DataFrame(rho_rows)
rho_df.to_csv('rho_O_curve.csv', index=False)
print(f"  Saved {len(rho_df)} rho rows")

# Find rho flattening: first O where consecutive difference in rho < 2% of initial range
flat_rows = []
for (ds,N,k,z), grp in rho_df.dropna(subset=['rho_O']).groupby(['dataset','N','k','z']):
    grp = grp.sort_values('O')
    rhos = grp['rho_O'].values
    Os   = grp['O'].values
    knee_O = grp['knee_O'].iloc[0]
    if len(rhos) < 3: continue
    rho_range = rhos[0] - rhos[-1]
    threshold = max(0.02 * rho_range, 0.005)
    flat_O = np.nan
    for i in range(len(rhos)-1):
        if abs(rhos[i+1] - rhos[i]) <= threshold:
            flat_O = Os[i]; break
    flat_rows.append({'dataset':ds,'N':N,'k':k,'z':z,'knee_O':knee_O,'flat_rho_O':flat_O})

flat_df = pd.DataFrame(flat_rows).dropna(subset=['flat_rho_O'])
print(f"\n  Flattening found for {len(flat_df)} / {len(flat_rows)} configs")
if len(flat_df) >= 3:
    r_val, p_val = pearsonr(flat_df['knee_O'].values, flat_df['flat_rho_O'].values)
    print(f"  Corr(O_knee, rho_flat_O): r={r_val:.3f}  p={p_val:.3e}")
else:
    print("  Not enough data for correlation")
flat_df.to_csv('rho_flattening.csv', index=False)

print("\n  rho(O_knee) for sample configs:")
rho_knee_vals = []
for r in flat_rows[:3]:
    ds = r['dataset']; N = r['N']; k = r['k']; z = r['z']
    kO = r['knee_O']
    sub = rho_df[(rho_df['dataset']==ds)&(rho_df['N']==N)&
                 (rho_df['k']==k)&(rho_df['z']==z)&(rho_df['O']==kO)]
    if not sub.empty:
        rv = sub.iloc[0]['rho_O']
        print(f"    {ds} N={N} k={k} z={z} O_knee={kO}: rho={rv:.3f}")
        rho_knee_vals.append(rv)

# ============================================================
# A3 — ADAPTIVE POOL RULE
# ============================================================
print("\n" + "="*70)
print("A3: ADAPTIVE POOL RULE VALIDATION")
print("="*70)

epsilons = [1.0, 0.5, 0.25, 0.1]
adaptive_rows = []

for _, row in explore_df.iterrows():
    N = int(row['N']); k = int(row['k']); z = int(row['z'])
    ds = str(row['dataset'])
    match = cert_df[(cert_df['dataset']==ds)&(cert_df['N']==N)&
                    (cert_df['k']==k)&(cert_df['z']==z)]
    if match.empty or pd.isna(match.iloc[0]['D_kz1']):
        continue
    D_kz1 = float(match.iloc[0]['D_kz1'])
    key = (ds, N, k, z)
    if key not in cached_dists:
        continue
    dists = cached_dists[key]

    for eps in epsilons:
        threshold = eps * D_kz1
        t_star = None
        # t is the pool overcount; we stop when D_{k+t+1} <= eps * D_{k+z+1}
        # D_{k+t+1} = dists[k+t-1]
        for t in range(1, len(dists) - k + 1):
            idx = k + t - 1
            if idx < len(dists) and dists[idx] <= threshold:
                t_star = t; break
        adaptive_rows.append({
            'dataset': ds, 'N': N, 'k': k, 'z': z,
            'epsilon': eps, 't_star': t_star,
            't_star_over_z': t_star/z if (t_star is not None and z > 0) else np.nan
        })

adaptive_df = pd.DataFrame(adaptive_rows)
adaptive_df.to_csv('adaptive_pool.csv', index=False)

print("\n  t*/z summary by epsilon:")
summary = adaptive_df.groupby('epsilon')['t_star_over_z'].agg(['mean','median','min','max','count'])
print(summary.round(3))

# ============================================================
# A5 — PAIRED COMPARISONS AT N<=2000 ADULT
# ============================================================
print("\n" + "="*70)
print("A5: PAIRED COMPARISONS AT N<=2000 (ADULT)")
print("="*70)

fc    = pd.read_csv('final_comparison.csv')
lp_df = pd.read_csv('LP_Results.csv').rename(columns={'outlier_percent':'p'})
fc['p'] = fc['p'].astype(float)
fc_lp = fc.merge(lp_df[['N','k','p','LP_Radius']], on=['N','k','p'], how='left')

for N_thresh, label in [(1000, "N<=1000 (12 configs)"), (2000, "N<=2000 (18 configs)")]:
    sub = fc_lp[fc_lp['N'] <= N_thresh].copy()
    print(f"\n  --- {label} ---")
    comparisons = [
        ('OverclusterDRG_radius', lambda r: r['charikar_radius']*3, 'OC-Local vs Charikar(×3)'),
        ('OverclusterDRG_radius', lambda r: r['gonzalez_radius'],   'OC-Local vs Gonzalez'),
        ('OverclusterDRG_radius', lambda r: r['SimplifiedDRG_radius'], 'OC-Local vs SimplifiedDRG'),
    ]
    for ca, cb_fn, name in comparisons:
        a = sub[ca].values
        b = np.array([cb_fn(r) for _, r in sub.iterrows()])
        wins  = (a < b).sum()
        ties  = (a == b).sum()
        try:
            stat, p = wilcoxon(a, b, alternative='less')
            print(f"  {name}: wins={wins}/{len(sub)} ties={ties}  Wilcoxon p={p:.3e}")
        except Exception as e:
            print(f"  {name}: wins={wins}/{len(sub)}  [{e}]")

# ============================================================
# A6 — WILCOXON ALL 36 ADULT CONFIGS
# ============================================================
print("\n" + "="*70)
print("A6: WILCOXON SIGNED-RANK ALL 36 ADULT CONFIGS")
print("="*70)

all36 = fc_lp.copy()
comparisons = [
    ('OverclusterDRG_radius', lambda r: r['charikar_radius']*3, 'OC-Local vs Charikar(×3)'),
    ('OverclusterDRG_radius', lambda r: r['gonzalez_radius'],   'OC-Local vs Gonzalez'),
    ('OverclusterDRG_radius', lambda r: r['SimplifiedDRG_radius'], 'OC-Local vs SimplifiedDRG'),
]
for ca, cb_fn, name in comparisons:
    a = all36[ca].values
    b = np.array([cb_fn(r) for _, r in all36.iterrows()])
    wins = (a < b).sum()
    try:
        stat, p = wilcoxon(a, b, alternative='less')
        print(f"  {name}: wins={wins}/{len(all36)}  Wilcoxon p={p:.3e}")
    except Exception as e:
        print(f"  {name}: wins={wins}/{len(all36)}  [{e}]")

# LP ratio comparison
print("\n  LP ratio stats (Adult configs with LP):")
lp_x = pd.read_csv('LP_Results.csv').rename(columns={'outlier_percent':'p'})
fc2 = pd.read_csv('final_comparison.csv')
fc2['p'] = fc2['p'].astype(float)
lp_x['p'] = lp_x['p'].astype(float)
all36_lp = fc2.merge(lp_x[['N','k','p','LP_Radius']], on=['N','k','p'], how='left')
all36_lp['OC_LP']   = all36_lp['OverclusterDRG_radius'] / all36_lp['LP_Radius']
all36_lp['Gonz_LP'] = all36_lp['gonzalez_radius'] / all36_lp['LP_Radius']
all36_lp['Char_LP'] = all36_lp['charikar_radius'] * 3 / all36_lp['LP_Radius']
all36_lp['DRG_LP']  = all36_lp['SimplifiedDRG_radius'] / all36_lp['LP_Radius']
for col, name in [('OC_LP','OC-Local'), ('Gonz_LP','Gonzalez'), ('Char_LP','Charikar(×3)'), ('DRG_LP','SimplifiedDRG')]:
    sub_valid = all36_lp[col].dropna()
    if len(sub_valid) > 0:
        print(f"  {name:18s}: mean={sub_valid.mean():.4f} std={sub_valid.std():.4f} "
              f"min={sub_valid.min():.4f} max={sub_valid.max():.4f} (n={len(sub_valid)})")

# ============================================================
# A7 — OLS REGRESSION FOR O_KNEE PREDICTORS
# ============================================================
print("\n" + "="*70)
print("A7: OLS / PARTIAL CORRELATIONS FOR O_KNEE/z")
print("="*70)

expl = explore_df.copy()
expl['knee_over_z'] = expl['knee_O'] / expl['z'].replace(0, np.nan)
features = ['z', 'z_over_k', 'N', 'sparsity_ratio', 'd']
target = 'knee_over_z'
sub_reg = expl[features + [target]].dropna()
print(f"  n={len(sub_reg)} configs")

print("\n  Pearson correlations:")
for f in features:
    r, p = pearsonr(sub_reg[f].values, sub_reg[target].values)
    print(f"    r({f:15s}, {target}) = {r:+.3f}  p={p:.3e}")

# OLS with numpy
Xmat = sub_reg[features].values
Xmat_std = (Xmat - Xmat.mean(0)) / Xmat.std(0)
Xmat_aug = np.column_stack([np.ones(len(Xmat_std)), Xmat_std])
yr = sub_reg[target].values
coeffs, res, _, _ = np.linalg.lstsq(Xmat_aug, yr, rcond=None)
ss_res = np.sum((yr - Xmat_aug @ coeffs)**2)
ss_tot = np.sum((yr - yr.mean())**2)
r2 = 1 - ss_res/ss_tot
print(f"\n  OLS (standardized features): R²={r2:.3f}")
for fname, coef in zip(['intercept']+features, coeffs):
    print(f"    {fname:15s}: {coef:+.3f}")

# ============================================================
# PRINT SUMMARY FOR PAPER
# ============================================================
print("\n" + "="*70)
print("NUMBERS FOR PAPER")
print("="*70)

print("\nA1 Certified ratio summary:")
for ds in ['adult','diabetes','covertype']:
    sub = cert_df[cert_df['dataset']==ds].dropna(subset=['cert_ratio'])
    print(f"  {ds}: mean={sub['cert_ratio'].mean():.2f} "
          f"(range {sub['cert_ratio'].min():.2f}–{sub['cert_ratio'].max():.2f})")

print("\nA1 cert_tightness (LB/R_LP) at N<=2000 Adult:")
cert_lp = pd.read_csv('certified_vs_lp.csv')
sub2k = cert_lp[cert_lp['N']<=2000]
print(f"  mean cert_tightness = {sub2k['cert_tightness'].mean():.3f} "
      f"(range {sub2k['cert_tightness'].min():.3f}–{sub2k['cert_tightness'].max():.3f})")
print(f"  The certificate captures {sub2k['cert_tightness'].mean()*100:.1f}% of R_LP on average")

print("\nA3 Adaptive pool stopping O=z:")
sub_z = adaptive_df[adaptive_df['epsilon']==1.0]
print(f"  epsilon=1.0 (O=z): t*/z mean={sub_z['t_star_over_z'].mean():.2f} "
      f"median={sub_z['t_star_over_z'].median():.2f}")
print(f"  This confirms: stopping at O=z IS the adaptive rule with epsilon=1")

print("\nAll CSVs saved.")
