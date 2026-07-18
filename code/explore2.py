"""
Deep exploration: What determines optimal O?
- Theoretical: O vs z/k, z/N, k/N, dimensionality
- Dataset fingerprint: statistics that predict optimal O
- Fine-grained O sweep with curve fitting
- Synthetic datasets with controlled properties
"""
import sys, os, ast, time, itertools
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
sys.path.insert(0, '/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')
import drg as drg_mod, drg1

# ── core OC-Local ─────────────────────────────────────────────────────────
def oc_local(X, k, z, O):
    _, Q = drg_mod.overcluster_drg(X, k, O=O)
    c    = drg1.local_search_fast(X, Q, k, z)
    return drg_mod.robust_radius(X, c, z)

def gonzalez_pool(X, m, start=None):
    if start is None:
        start = int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())
    centers = [start]
    md = np.linalg.norm(X - X[start], axis=1).astype(np.float64)
    for _ in range(m-1):
        nxt = int(md.argmax())
        centers.append(nxt)
        np.minimum(md, np.linalg.norm(X - X[nxt], axis=1), out=md)
    return centers

# ── dataset statistics (fingerprint) ─────────────────────────────────────
def dataset_stats(X, k, z):
    N, d = X.shape
    # 1. Voronoi imbalance: run Gonzalez to k, get cell sizes
    pool_k = gonzalez_pool(X, k)
    D = np.column_stack([np.linalg.norm(X - X[c], axis=1) for c in pool_k])
    assignments = D.argmin(axis=1)
    cell_sizes = np.array([np.sum(assignments==i) for i in range(k)])
    voronoi_cv = cell_sizes.std() / cell_sizes.mean()  # coefficient of variation

    # 2. Outlier "pull" strength: how much do outliers distort Gonzalez?
    # Run Gonzalez to k+z, count how many steps are in sparse regions
    pool_kz = gonzalez_pool(X, k+z)
    # How many pool points are in very low-density regions?
    knn_dists = np.sort(cdist(X[pool_kz], X), axis=1)[:, 1:6].mean(axis=1)
    sparsity_ratio = knn_dists.max() / (knn_dists.mean() + 1e-10)

    # 3. Density heterogeneity: ratio of max to mean local density
    sample = np.random.default_rng(42).choice(len(X), size=min(500, len(X)), replace=False)
    knn_all = np.sort(cdist(X[sample], X), axis=1)[:, 1:11].mean(axis=1)
    density_cv = knn_all.std() / knn_all.mean()

    # 4. z/k ratio
    z_over_k = z / k

    # 5. Effective dimensionality (PCA explained variance)
    centered = X - X.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered[:min(500,len(X))], full_matrices=False)
    var_explained = (sv**2).cumsum() / (sv**2).sum()
    eff_dim = np.searchsorted(var_explained, 0.95) + 1

    # 6. Outlier fraction
    z_over_N = z / len(X)

    # 7. Cluster separation: ratio between-cluster to within-cluster dist
    pool_k2 = gonzalez_pool(X, k)
    D2 = np.column_stack([np.linalg.norm(X - X[c], axis=1) for c in pool_k2])
    within = D2.min(axis=1).mean()
    between_d = cdist(X[pool_k2], X[pool_k2])
    np.fill_diagonal(between_d, np.inf)
    between = between_d.min()
    separation = between / (within + 1e-10)

    return {
        'voronoi_cv': round(float(voronoi_cv), 4),
        'sparsity_ratio': round(float(sparsity_ratio), 4),
        'density_cv': round(float(density_cv), 4),
        'z_over_k': round(float(z_over_k), 4),
        'eff_dim': int(eff_dim),
        'z_over_N': round(float(z_over_N), 4),
        'separation': round(float(separation), 4),
        'd': int(d),
        'N': len(X), 'k': k, 'z': z
    }

# ── load datasets ─────────────────────────────────────────────────────────
DATASETS = {}
for name, fname in [('adult','adult_final_dataset.py'),
                     ('diabetes','diabetes_dataset.py'),
                     ('covertype','covertype_dataset.py')]:
    with open(fname) as f:
        DATASETS[name] = np.asarray(ast.literal_eval(f.read()), dtype=np.float32)
    print(f"Loaded {name}: {DATASETS[name].shape}")

# O values to sweep (finer near z)
O_SWEEP = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]

CONFIGS = [
    (500,  10, 0.10), (500,  10, 0.20), (500,  20, 0.10), (500,  20, 0.20),
    (500,  50, 0.10), (500,  50, 0.20),
    (1000, 10, 0.10), (1000, 10, 0.20), (1000, 20, 0.10), (1000, 20, 0.20),
    (1000, 50, 0.10), (1000, 50, 0.20),
    (2000, 10, 0.10), (2000, 10, 0.20), (2000, 20, 0.10), (2000, 20, 0.20),
    (2000, 50, 0.10), (2000, 50, 0.20),
    (5000, 10, 0.10), (5000, 10, 0.20), (5000, 20, 0.10), (5000, 20, 0.20),
    (5000, 50, 0.10), (5000, 50, 0.20),
]

rows = []

# ═══════════════════════════════════════════════════════════════════════════
# MAIN SWEEP: fine-grained O + dataset statistics per config
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("MAIN SWEEP: O ablation + dataset fingerprint")
print("="*70)

for ds in ['adult', 'diabetes', 'covertype']:
    X_full = DATASETS[ds]
    for (N, k, op) in CONFIGS:
        if N > len(X_full): continue
        X = X_full[:N]
        z = int(op * N)
        print(f"\n  {ds} N={N} k={k} z={z}", flush=True)

        # Dataset stats
        stats = dataset_stats(X, k, z)

        # O sweep
        radii = {}
        for O_val in [v for v in O_SWEEP if 1 <= v <= max(z*3, 100)]:
            r = oc_local(X, k, z, O_val)
            radii[O_val] = r

        # Find optimal O (min radius)
        opt_O = min(radii, key=radii.get)
        opt_r = radii[opt_O]
        r_at_20  = radii.get(20, None)
        r_at_z   = radii.get(z, radii.get(min(radii.keys(), key=lambda x: abs(x-z)), None))

        # Knee: smallest O within 2% of optimal
        knee_O = None
        for O_val in sorted(radii.keys()):
            if radii[O_val] <= opt_r * 1.02:
                knee_O = O_val; break

        print(f"    opt_O={opt_O}  opt_r={opt_r:.4f}  knee_O={knee_O}  "
              f"r@20={r_at_20:.4f}  r@z={r_at_z:.4f}  gap={100*(r_at_20-opt_r)/opt_r:.1f}%", flush=True)

        row = {**stats,
               'dataset': ds,
               'opt_O': opt_O, 'opt_r': round(opt_r,6),
               'knee_O': knee_O, 'opt_over_z': round(opt_O/z,3),
               'knee_over_z': round(knee_O/z,3) if knee_O else None,
               'r_at_O20': round(r_at_20,6) if r_at_20 else None,
               'r_at_Oz':  round(r_at_z,6)  if r_at_z  else None,
               'gap_O20_vs_opt': round(100*(r_at_20-opt_r)/opt_r,2) if r_at_20 else None,
        }
        # Store full O curve
        for O_val, r in radii.items():
            row[f'r_O{O_val}'] = round(r,6)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('explore2_results.csv', index=False)
print(f"\n[Saved {len(df)} rows → explore2_results.csv]")

# ═══════════════════════════════════════════════════════════════════════════
# CORRELATION: which stats predict optimal O?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CORRELATION ANALYSIS: what predicts optimal O?")
print("="*70)

stat_cols = ['voronoi_cv','sparsity_ratio','density_cv','z_over_k',
             'eff_dim','z_over_N','separation','d','z','k','N']
target = 'knee_over_z'  # optimal O / z ratio

print(f"\n  Predicting: {target}")
print(f"  {'Feature':<18}  {'Pearson r':>10}  {'p-value':>10}")
for col in stat_cols:
    valid = df[[col, target]].dropna()
    if len(valid) < 5: continue
    r, p = pearsonr(valid[col], valid[target])
    print(f"  {col:<18}  {r:>10.4f}  {p:>10.4f}{'  *' if p<0.05 else ''}")

# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC: controlled experiments to isolate factors
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SYNTHETIC: vary z/k ratio, dimensionality, cluster separation")
print("="*70)

def make_clustered(N, k, d, z, separation=3.0, seed=0):
    """N points, k balanced clusters in d dims, z outliers injected uniformly"""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)) * separation
    n_inliers = N - z
    pts_per = n_inliers // k
    pts = []
    for c in centers:
        pts.append(c + rng.standard_normal((pts_per, d)) * 0.5)
    pts = np.vstack(pts)
    # pad if needed
    if len(pts) < n_inliers:
        pts = np.vstack([pts, centers[0] + rng.standard_normal((n_inliers-len(pts), d))*0.5])
    outliers = rng.standard_normal((z, d)) * separation * 3
    X = np.vstack([pts[:n_inliers], outliers]).astype(np.float32)
    return X

syn_rows = []
print("\n  Varying z/k ratio (N=2000, d=6, separation=3):")
N, d = 2000, 6
for k in [5, 10, 20, 50]:
    for zr in [0.05, 0.10, 0.20, 0.30]:
        z = int(zr * N)
        X = make_clustered(N, k, d, z, separation=3.0)
        best_r, best_O, knee_O = np.inf, None, None
        radii = {}
        for O_val in [v for v in O_SWEEP if v <= z*3+10]:
            r = oc_local(X, k, z, O_val)
            radii[O_val] = r
            if r < best_r: best_r, best_O = r, O_val
        for O_val in sorted(radii.keys()):
            if radii[O_val] <= best_r*1.02: knee_O = O_val; break
        print(f"  k={k:2d} z/N={zr:.0%} z={z:3d}  opt_O={best_O:4d}  knee_O={knee_O:4d}  "
              f"opt/z={best_O/z:.2f}  knee/z={knee_O/z:.2f}  z/k={z/k:.1f}", flush=True)
        syn_rows.append({'exp':'syn_vary_zk','N':N,'d':d,'k':k,'z':z,'zr':zr,
                         'z_over_k':z/k,'opt_O':best_O,'knee_O':knee_O,
                         'opt_over_z':best_O/z,'knee_over_z':knee_O/z})

print("\n  Varying dimensionality (N=2000, k=20, z=200, separation=3):")
k, z = 20, 200
for d in [2, 4, 6, 8, 12, 16, 20]:
    X = make_clustered(2000, k, d, z, separation=3.0)
    best_r, best_O, knee_O = np.inf, None, None
    radii = {}
    for O_val in [v for v in O_SWEEP if v <= z*3]:
        r = oc_local(X, k, z, O_val)
        radii[O_val] = r
        if r < best_r: best_r, best_O = r, O_val
    for O_val in sorted(radii.keys()):
        if radii[O_val] <= best_r*1.02: knee_O = O_val; break
    print(f"  d={d:2d}  opt_O={best_O:4d}  knee_O={knee_O:4d}  opt/z={best_O/z:.2f}  knee/z={knee_O/z:.2f}", flush=True)
    syn_rows.append({'exp':'syn_vary_d','N':2000,'d':d,'k':k,'z':z,
                     'opt_O':best_O,'knee_O':knee_O,'opt_over_z':best_O/z,'knee_over_z':knee_O/z})

print("\n  Varying cluster separation (N=2000, k=20, z=200, d=6):")
k, z, d = 20, 200, 6
for sep in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
    X = make_clustered(2000, k, d, z, separation=sep)
    best_r, best_O, knee_O = np.inf, None, None
    radii = {}
    for O_val in [v for v in O_SWEEP if v <= z*3]:
        r = oc_local(X, k, z, O_val)
        radii[O_val] = r
        if r < best_r: best_r, best_O = r, O_val
    for O_val in sorted(radii.keys()):
        if radii[O_val] <= best_r*1.02: knee_O = O_val; break
    print(f"  sep={sep:.1f}  opt_O={best_O:4d}  knee_O={knee_O:4d}  opt/z={best_O/z:.2f}  knee/z={knee_O/z:.2f}", flush=True)
    syn_rows.append({'exp':'syn_vary_sep','N':2000,'d':d,'k':k,'z':z,'sep':sep,
                     'opt_O':best_O,'knee_O':knee_O,'opt_over_z':best_O/z,'knee_over_z':knee_O/z})

pd.DataFrame(syn_rows).to_csv('explore2_synthetic.csv', index=False)
print(f"\n[Saved → explore2_synthetic.csv]")
print("\nAll done.")
