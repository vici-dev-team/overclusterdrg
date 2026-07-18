"""
Revision experiments for overclusterdrg_alenex.tex
Covers A1-A9 from the master revision prompt.
"""

import sys, os, ast, time, math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, pearsonr
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

import drg as drg_mod
import drg1

BATCH = 1024

# ============================================================
# UTILITIES
# ============================================================

def robust_radius(X, centers_idx, O):
    C = X[centers_idx]
    n = len(X)
    min_d = np.full(n, np.inf, dtype=np.float64)
    for i in range(0, len(centers_idx), BATCH):
        batch = X[centers_idx[i:i+BATCH]]
        d = cdist(X, batch, metric='euclidean').min(axis=1)
        np.minimum(min_d, d, out=min_d)
    sd = np.sort(min_d)[::-1]
    return float(sd[O]) if O < n else 0.0

def gonzalez_with_distances(X, steps):
    """Run Gonzalez for `steps` steps, returning (centers, D) where D[t-1] is
    the selection distance when adding center t (for t=2..steps).
    D has length steps-1; D[i] = distance of center i+2 from its nearest predecessor.
    The standard LB formula uses D[k+z] (0-indexed) = D_{k+z+1} (1-indexed)."""
    n = len(X)
    mean = X.mean(axis=0, keepdims=True)
    c1 = int(cdist(X, mean, metric='euclidean').argmin())
    centers = [c1]
    min_d = cdist(X, X[[c1]], metric='euclidean').ravel()
    distances = []  # distances[i] = max(min_d) BEFORE adding center i+2
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

# ============================================================
# A1 — CERTIFIED RATIOS
# ============================================================
print("\n" + "="*70)
print("A1: CERTIFIED RATIOS")
print("="*70)

# Load existing OC-Local results
adult_df   = pd.read_csv('final_comparison.csv')
diab_df    = pd.read_csv('full_results_diabetes.csv')
cov_df     = pd.read_csv('full_results_covertype.csv')

# Rename OC-Local column for covertype/diabetes
adult_df['oc_r'] = adult_df['OverclusterDRG_radius']
diab_df['oc_r']  = diab_df['oc_correct_r']
cov_df['oc_r']   = cov_df['oc_correct_r']

# Load datasets
print("Loading datasets...")
X_adult    = load_dataset('adult_final_dataset.py')
X_diabetes = load_dataset('diabetes_dataset.py')
X_cover    = load_dataset('covertype_dataset.py')
print(f"  Adult: {X_adult.shape}, Diabetes: {X_diabetes.shape}, Covertype: {X_cover.shape}")

certified_rows = []

for (ds_name, df, X_full) in [
    ('adult',    adult_df,  X_adult),
    ('diabetes', diab_df,   X_diabetes),
    ('covertype',cov_df,    X_cover),
]:
    print(f"\n  {ds_name} ({len(df)} configs)...")
    for _, row in df.iterrows():
        N  = int(row['N'])
        k  = int(row['k'])
        op = float(row.get('outlier_pct', row.get('p', row.get('outlier_percent', np.nan))))
        if np.isnan(op):
            # fallback: try outlier_pct column
            try: op = float(row['outlier_pct'])
            except: op = float(row['p'])
        z  = int(op * N)
        oc_r = float(row['oc_r'])

        X = X_full[:N]
        steps_needed = k + z + 1  # to get D_{k+z+1}
        if steps_needed > N:
            print(f"    skip {ds_name} N={N} k={k} z={z}: steps > N")
            certified_rows.append({
                'dataset': ds_name, 'N': N, 'k': k, 'z': z, 'outlier_pct': op,
                'oc_r': oc_r, 'D_kz1': np.nan, 'LB': np.nan, 'cert_ratio': np.nan
            })
            continue

        t0 = time.perf_counter()
        _, dists = gonzalez_with_distances(X, steps_needed)
        elapsed = time.perf_counter() - t0

        # dists[i] = D_{i+2}, so D_{k+z+1} = dists[k+z-1] (0-indexed)
        # i.e. dists has length steps_needed-1 = k+z
        # dists[0]=D_2, dists[1]=D_3, ..., dists[k+z-1]=D_{k+z+1}
        D_kz1 = dists[k + z - 1]
        LB    = D_kz1 / 2.0
        cert_r = oc_r / LB if LB > 0 else np.nan

        certified_rows.append({
            'dataset': ds_name, 'N': N, 'k': k, 'z': z, 'outlier_pct': op,
            'oc_r': oc_r, 'D_kz1': D_kz1, 'LB': LB, 'cert_ratio': cert_r
        })
        print(f"    N={N:6d} k={k:2d} z={z:5d}: D_kz1={D_kz1:.4f} LB={LB:.4f} "
              f"oc_r={oc_r:.4f} cert={cert_r:.4f} ({elapsed:.2f}s)")

cert_df = pd.DataFrame(certified_rows)
cert_df.to_csv('certified_ratios.csv', index=False)
print("\nCertified ratio summary:")
print(cert_df.groupby('dataset')['cert_ratio'].describe().round(4))

# Compare with LP at N<=2000 for adult
adult_lp = pd.read_csv('LP_Results.csv')
adult_cert_lp = cert_df[cert_df['dataset']=='adult'].merge(
    adult_lp[['N','k','outlier_percent','LP_Radius']].rename(
        columns={'outlier_percent':'outlier_pct'}),
    on=['N','k','outlier_pct'], how='left')
adult_cert_lp = adult_cert_lp.dropna(subset=['LP_Radius'])
adult_cert_lp['lp_ratio'] = adult_cert_lp['oc_r'] / adult_cert_lp['LP_Radius']
adult_cert_lp['cert_tightness'] = adult_cert_lp['LB'] / adult_cert_lp['LP_Radius']
print("\nAdult cert vs LP (N<=2000):")
sub = adult_cert_lp[adult_cert_lp['N']<=2000][['N','k','outlier_pct','cert_ratio','lp_ratio','cert_tightness']]
print(sub.to_string(index=False))
adult_cert_lp.to_csv('certified_vs_lp.csv', index=False)

# ============================================================
# A2 — ρ(O) CURVE FOR 72 O-ABLATION CONFIGS
# ============================================================
print("\n" + "="*70)
print("A2: rho(O) CURVE OVERLAY")
print("="*70)

explore_df = pd.read_csv('explore2_results.csv')
print(f"  {len(explore_df)} ablation configs")

# O columns available in explore2_results.csv
o_cols = [c for c in explore_df.columns if c.startswith('r_O')]
o_vals = sorted([int(c.split('_O')[1]) for c in o_cols])

rho_rows = []
for _, row in explore_df.iterrows():
    N  = int(row['N']); k = int(row['k']); z = int(row['z'])
    ds = str(row['dataset']); knee_O = int(row['knee_O'])

    # D_kz1 is already computed from certified_ratios (match by N,k,z,ds)
    match = cert_df[(cert_df['dataset']==ds)&(cert_df['N']==N)&
                    (cert_df['k']==k)&(cert_df['z']==z)]
    if match.empty or pd.isna(match.iloc[0]['D_kz1']):
        continue
    D_kz1 = float(match.iloc[0]['D_kz1'])

    # Load dataset to run Gonzalez for larger pool
    if ds == 'adult':
        X_full = X_adult
    elif ds == 'diabetes':
        X_full = X_diabetes
    else:
        X_full = X_cover

    X = X_full[:N]
    max_O = max(o_vals)
    steps = k + max_O + 1
    if steps > N:
        steps = N

    _, dists = gonzalez_with_distances(X, steps)
    # dists[i] = D_{i+2}; D_{k+O+1} = dists[k+O-1]
    for O in o_vals:
        idx = k + O - 1
        if idx < len(dists):
            D_kO1 = dists[idx]
            rho_O = D_kO1 / D_kz1 if D_kz1 > 0 else np.nan
        else:
            D_kO1 = np.nan; rho_O = np.nan

        r_col = f'r_O{O}'
        r_val  = float(row[r_col]) if r_col in row.index and not pd.isna(row[r_col]) else np.nan
        rho_rows.append({'dataset':ds,'N':N,'k':k,'z':z,'knee_O':knee_O,
                         'O':O,'D_kO1':D_kO1,'rho_O':rho_O,'radius':r_val})

rho_df = pd.DataFrame(rho_rows)
rho_df.to_csv('rho_O_curve.csv', index=False)

# Find rho flattening point (where rho_O change < 0.01 of initial drop)
flat_rows = []
for (ds,N,k,z), grp in rho_df.dropna(subset=['rho_O']).groupby(['dataset','N','k','z']):
    grp = grp.sort_values('O')
    rhos = grp['rho_O'].values
    Os   = grp['O'].values
    # flattening: first O where |rho(O+step) - rho(O)| < 0.01
    flat_O = np.nan
    for i in range(len(rhos)-1):
        if abs(rhos[i+1] - rhos[i]) < 0.01:
            flat_O = Os[i]; break
    knee_O = grp['knee_O'].iloc[0]
    flat_rows.append({'dataset':ds,'N':N,'k':k,'z':z,'knee_O':knee_O,'flat_rho_O':flat_O})

flat_df = pd.DataFrame(flat_rows)
corr_val, corr_p = pearsonr(
    flat_df.dropna()['knee_O'].values,
    flat_df.dropna()['flat_rho_O'].values)
print(f"  Correlation O_knee vs rho-flat-O: r={corr_val:.3f} p={corr_p:.2e}")
flat_df.to_csv('rho_flattening.csv', index=False)

# ============================================================
# A3 — ADAPTIVE POOL RULE VALIDATION
# ============================================================
print("\n" + "="*70)
print("A3: ADAPTIVE POOL RULE")
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

    if ds == 'adult': X_full = X_adult
    elif ds == 'diabetes': X_full = X_diabetes
    else: X_full = X_cover
    X = X_full[:N]
    steps = k + max(o_vals) + 1
    if steps > N: steps = N
    _, dists = gonzalez_with_distances(X, steps)

    for eps in epsilons:
        threshold = eps * D_kz1
        t_star = None
        for t in range(1, len(dists) - k + 1):
            idx = k + t - 1
            if idx < len(dists) and dists[idx] <= threshold:
                t_star = t; break
        adaptive_rows.append({
            'dataset': ds, 'N': N, 'k': k, 'z': z,
            'epsilon': eps, 't_star': t_star,
            't_star_over_z': t_star/z if t_star is not None and z>0 else np.nan
        })

adaptive_df = pd.DataFrame(adaptive_rows)
adaptive_df.to_csv('adaptive_pool.csv', index=False)
print("\nAdaptive pool t*/z summary by epsilon:")
print(adaptive_df.groupby('epsilon')['t_star_over_z'].describe().round(3))

# ============================================================
# A5 — PAIRED COMPARISONS RESTRICTED TO N<=2000 (ADULT)
# ============================================================
print("\n" + "="*70)
print("A5: PAIRED COMPARISONS N<=2000 (ADULT)")
print("="*70)

fc = pd.read_csv('final_comparison.csv')
fc['p'] = fc['p'].astype(float)
lp = pd.read_csv('LP_Results.csv').rename(columns={'outlier_percent':'p'})
fc_lp = fc.merge(lp[['N','k','p','LP_Radius']], on=['N','k','p'], how='left')
fc_lp['OC_r'] = fc_lp['OverclusterDRG_radius']
fc_lp['Charikar_3r'] = fc_lp['charikar_radius'] * 3
fc_lp['Gonz_r'] = fc_lp['gonzalez_radius']
fc_lp['LP'] = fc_lp['LP_Radius']

sub12 = fc_lp[fc_lp['N']<=1000].copy()  # 12 configs (exact LP, fast solve)
sub18 = fc_lp[fc_lp['N']<=2000].copy()  # 18 configs

print(f"  Using N<=1000: {len(sub12)} configs  |  N<=2000: {len(sub18)} configs")

for label, sub in [("N<=1000 (12 exact-LP)", sub12), ("N<=2000 (18 exact-LP)", sub18)]:
    print(f"\n  {label}")
    comparisons = [
        ("OC_r", "Charikar_3r", "OC-Local vs Charikar(×3)"),
        ("OC_r", "Gonz_r",      "OC-Local vs Gonzalez"),
    ]
    for ca, cb, name in comparisons:
        a = sub[ca].values; b = sub[cb].values
        wins = (a < b).sum()
        stat, p = wilcoxon(a, b, alternative='less')
        print(f"    {name}: wins={wins}/{len(sub)}, Wilcoxon p={p:.3e}")

# A6 — WILCOXON FOR ALL 36 CONFIGS (ADULT)
print("\n" + "="*70)
print("A6: WILCOXON SIGNED-RANK (ALL 36 ADULT CONFIGS)")
print("="*70)

# We need OC-Local vs Charikar, Gonzalez, OC-Forward, SimplifiedDRG
# SimplifiedDRG is in final_comparison.csv; OC-Forward we'd need from existing data
# Let's check what columns are available
print("  Columns:", list(fc_lp.columns))

# Read OC-Forward from explore2_results (it's r_Oz = OC-Forward without local search)
# Actually OC-Forward isn't directly in final_comparison.csv; check other files
try:
    explore_adult = explore_df[explore_df['dataset']=='adult'].copy()
    explore_adult['r_Oz_forward'] = explore_adult['r_Oz']
    oc_fwd_map = {}
    for _, row in explore_adult.iterrows():
        oc_fwd_map[(int(row['N']), int(row['k']), int(row['z']))] = float(row['r_Oz'])
except:
    oc_fwd_map = {}

# Check drg_kcenter_results_compared.csv for OC-Forward
try:
    drg_comp = pd.read_csv('drg_kcenter_results_compared.csv')
    print("  drg_kcenter columns:", list(drg_comp.columns)[:10])
except: pass

# Wilcoxon on all 36 using available baselines
all_adult = fc_lp.copy()
print(f"\n  All 36 Adult configs:")
for ca, cb, name in [
    ('OC_r', 'Charikar_3r', 'OC-Local vs Charikar(×3)'),
    ('OC_r', 'Gonz_r',      'OC-Local vs Gonzalez'),
    ('OC_r', 'SimplifiedDRG_radius', 'OC-Local vs SimplifiedDRG'),
]:
    try:
        a = all_adult[ca].values; b = all_adult[cb].values
        wins = (a < b).sum()
        stat, p = wilcoxon(a, b, alternative='less')
        print(f"  {name}: wins={wins}/{len(all_adult)}, Wilcoxon p={p:.3e}")
    except Exception as e:
        print(f"  {name}: error {e}")

# ============================================================
# A7 — OLS / PARTIAL CORRELATIONS FOR O_KNEE PREDICTORS
# ============================================================
print("\n" + "="*70)
print("A7: OLS REGRESSION FOR O_KNEE/z PREDICTORS")
print("="*70)

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    have_sklearn = True
except:
    have_sklearn = False
    print("  sklearn not available; using numpy OLS")

expl = explore_df.copy()
expl['knee_over_z'] = expl['knee_O'] / expl['z'].replace(0, np.nan)

features = ['z', 'z_over_k', 'N', 'sparsity_ratio', 'd']
target = 'knee_over_z'
sub_reg = expl[features + [target]].dropna()

print(f"  n={len(sub_reg)} configs for regression")
for f in features:
    r, p = pearsonr(sub_reg[f].values, sub_reg[target].values)
    print(f"  Pearson r({f}, {target}) = {r:+.3f}  p={p:.2e}")

if have_sklearn:
    Xr = StandardScaler().fit_transform(sub_reg[features].values)
    yr = sub_reg[target].values
    reg = LinearRegression().fit(Xr, yr)
    print(f"\n  OLS R² = {reg.score(Xr, yr):.3f}")
    for fname, coef in zip(features, reg.coef_):
        print(f"    {fname:15s}: coef={coef:+.3f}")
else:
    Xr = sub_reg[features].values
    Xr = np.column_stack([np.ones(len(Xr)), Xr])
    yr = sub_reg[target].values
    coeffs = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    ss_res = np.sum((yr - Xr @ coeffs)**2)
    ss_tot = np.sum((yr - yr.mean())**2)
    print(f"\n  OLS R² = {1 - ss_res/ss_tot:.3f}")
    for fname, coef in zip(['intercept']+features, coeffs):
        print(f"    {fname:15s}: coef={coef:+.3f}")

# ============================================================
# A8 — DING TRIAL ABLATION (T in {10, 50, 200})
# ============================================================
print("\n" + "="*70)
print("A8: DING TRIAL ABLATION")
print("="*70)

try:
    import dingvOC as dingmod

    def ding_fixed_T(X, k, z, T):
        """Run Ding algorithm with fixed T trials."""
        best_r = np.inf
        for _ in range(T):
            # Sample z+1 random outlier-free starting set and apply Gonzalez-like
            idx = np.random.choice(len(X), z+1, replace=False)
            # Pick the point that minimizes current radius as starting center
            # Actually use dingmod's interface if available
            try:
                r = dingmod.ding_algorithm(X, k, z, max_trials=T)
                return r
            except:
                break
        return np.nan

    # Check dingmod interface
    print("  dingvOC functions:", [f for f in dir(dingmod) if not f.startswith('_')])

    # Use small adult subset for speed
    X_small = X_adult[:1000]
    k_test, z_test = 10, 100

    for T in [10, 50, 200]:
        t0 = time.perf_counter()
        try:
            r = dingmod.ding_kcenter(X_small, k_test, z_test, T=T)
            elapsed = time.perf_counter() - t0
            print(f"  T={T:4d}: radius={r:.4f} time={elapsed:.2f}s")
        except Exception as e:
            print(f"  T={T}: {e}")

except Exception as e:
    print(f"  Ding ablation: {e}")

# ============================================================
# A4 — SCALABILITY: N=50,000 AND N=99,000 (DIABETES)
# ============================================================
print("\n" + "="*70)
print("A4: SCALABILITY N=50000 and N=99000 (Diabetes k=20 z=10%)")
print("="*70)

k_s, op_s = 20, 0.10

for N_s in [50000, 99000]:
    z_s = int(op_s * N_s)
    X_s = X_diabetes[:N_s]
    print(f"\n  N={N_s:,} k={k_s} z={z_s}...")

    t0 = time.perf_counter()
    centers, Q = drg_mod.overcluster_drg(X_s, k_s, O=z_s)
    Q_list = list(Q) if not isinstance(Q, list) else Q
    centers_ls = drg1.local_search_fast(X_s, Q_list, k_s, z_s)
    r = robust_radius(X_s, centers_ls, z_s)
    elapsed = time.perf_counter() - t0
    print(f"  N={N_s:,}: radius={r:.4f} time={elapsed:.1f}s")

    # Also get D_{k+z+1} for certified LB
    steps = k_s + z_s + 1
    if steps <= N_s:
        t0 = time.perf_counter()
        _, dists = gonzalez_with_distances(X_s, steps)
        D_kz1 = dists[k_s + z_s - 1]
        LB = D_kz1 / 2.0
        cert_r = r / LB
        elapsed2 = time.perf_counter() - t0
        print(f"  N={N_s:,}: LB={LB:.4f} cert_ratio={cert_r:.4f} (gonzalez {elapsed2:.1f}s)")

# ============================================================
# A9 note — Covertype full at N=581,012 (too large for quick run)
# ============================================================
print("\n" + "="*70)
print("A9: Covertype full N note")
print("="*70)
print(f"  Covertype dataset has {len(X_cover):,} rows (max available).")
print("  Full covertype run requires the full 581,012-row dataset.")
print("  Current dataset: N_max =", len(X_cover))

# ============================================================
# SUMMARY TABLE FOR PAPER
# ============================================================
print("\n" + "="*70)
print("SUMMARY: CERTIFIED RATIOS BY DATASET")
print("="*70)

for ds in ['adult', 'diabetes', 'covertype']:
    sub = cert_df[cert_df['dataset']==ds].dropna(subset=['cert_ratio'])
    print(f"\n  {ds}: mean cert_ratio={sub['cert_ratio'].mean():.4f} "
          f"std={sub['cert_ratio'].std():.4f} "
          f"min={sub['cert_ratio'].min():.4f} max={sub['cert_ratio'].max():.4f}")

print("\n  Adult cert vs LP (all configs with LP):")
if len(adult_cert_lp):
    print(f"  mean cert_tightness (LB/R_LP) = {adult_cert_lp['cert_tightness'].mean():.4f}")
    print(f"  mean cert_ratio     (r/LB)    = {adult_cert_lp['cert_ratio'].mean():.4f}")
    print(f"  mean lp_ratio       (r/R_LP)  = {adult_cert_lp['lp_ratio'].mean():.4f}")

print("\nAll results saved.")
print("  certified_ratios.csv — A1 full table")
print("  certified_vs_lp.csv  — A1 LP comparison")
print("  rho_O_curve.csv      — A2 rho(O) per config")
print("  rho_flattening.csv   — A2 flattening-point correlation")
print("  adaptive_pool.csv    — A3 adaptive stopping")
