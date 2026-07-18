"""
Comprehensive exploration on N≤5000.
Runs many experiments, prints results inline, saves to explore_results.csv.
"""
import sys, os, ast, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

import drg as drg_mod, drg1
from gonzalez import gonzalez_standard
from rkc import charikar_kcenter_outliers

# ── helpers ──────────────────────────────────────────────────────────────────
def robust_r(X, centers, z):
    D = np.column_stack([np.linalg.norm(X - X[c], axis=1) for c in centers])
    return float(np.partition(D.min(axis=1), -(z+1))[-(z+1)])

def gonzalez_pool(X, m, start_idx):
    centers = [start_idx]
    md = np.linalg.norm(X - X[start_idx], axis=1).astype(np.float64)
    for _ in range(m-1):
        nxt = int(md.argmax())
        centers.append(nxt)
        np.minimum(md, np.linalg.norm(X - X[nxt], axis=1), out=md)
    return centers

def centroid_start(X):
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())

def farthest_start(X):
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmax())

def random_start(X, seed):
    return int(np.random.default_rng(seed).integers(len(X)))

def kmeans_pp_start(X, seed):
    rng = np.random.default_rng(seed)
    return int(rng.choice(len(X)))  # just random first for now

def oc_local_full(X, k, z, O, start_idx=None):
    if start_idx is None: start_idx = centroid_start(X)
    pool = gonzalez_pool(X, k+O, start_idx)
    centers, Q = pool, pool  # Q = pool
    # forward greedy from drg.py (densest Voronoi)
    t0 = time.perf_counter()
    c_fwd, Q2 = drg_mod.overcluster_drg(X, k, O=O)
    c_ls = drg1.local_search_fast(X, Q2, k, z)
    r = drg_mod.robust_radius(X, c_ls, z)
    return r, time.perf_counter()-t0

def multi_start_oc(X, k, z, O, T=3):
    best_r = np.inf
    starts = [centroid_start(X), farthest_start(X)] + [random_start(X, s) for s in range(T-2)]
    for s in starts[:T]:
        pool = gonzalez_pool(X, k+O, s)
        _, Q = drg_mod.overcluster_drg(X, k, O=O)  # use drg's gonzalez
        c_ls = drg1.local_search_fast(X, Q, k, z)
        r = drg_mod.robust_radius(X, c_ls, z)
        if r < best_r: best_r = r
    return best_r

# Load datasets
DATASETS = {}
for name, fname in [('adult','adult_final_dataset.py'),
                     ('diabetes','diabetes_dataset.py'),
                     ('covertype','covertype_dataset.py')]:
    with open(fname) as f:
        DATASETS[name] = np.asarray(ast.literal_eval(f.read()), dtype=np.float32)
    print(f"Loaded {name}: {DATASETS[name].shape}")

SIZES  = [500, 1000, 2000, 5000]
KS     = [10, 20, 50]
OPS    = [0.10, 0.20]
rows   = []

# ═══════════════════════════════════════════════════════════════════════════
# EXP 1: O ablation — fine-grained O sweep
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 1: O ablation (pool size = k+O)")
print("="*70)

O_VALS = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000]

for ds in ['adult','diabetes','covertype']:
    X_full = DATASETS[ds]
    for N in [500, 1000, 2000, 5000]:
        X = X_full[:N]
        for k in [10, 20]:
            for op in [0.10, 0.20]:
                z = int(op*N)
                ref_r, _ = oc_local_full(X, k, z, O=z)  # reference O=z
                print(f"\n  {ds} N={N} k={k} z={z} ({int(op*100)}%)  ref(O=z={z}): r={ref_r:.5f}")
                for O_val in [v for v in O_VALS if v <= max(5, z*3)]:
                    r, t = oc_local_full(X, k, z, O=O_val)
                    pct = 100*(r-ref_r)/ref_r
                    rows.append({'exp':'O_ablation','dataset':ds,'N':N,'k':k,
                                 'z':z,'outlier_pct':op,'O':O_val,'pool':k+O_val,
                                 'radius':round(r,6),'time_s':round(t,4),
                                 'vs_ref_pct':round(pct,3)})
                    marker = " ← O=z" if O_val==z else (" ← O=20" if O_val==20 else "")
                    print(f"    O={O_val:>5} pool={k+O_val:>5}  r={r:.5f}  {pct:+.1f}%{marker}")

pd.DataFrame(rows).to_csv('explore_results.csv', index=False)
print(f"\n[saved {len(rows)} rows → explore_results.csv]")

# ═══════════════════════════════════════════════════════════════════════════
# EXP 2: Starting point comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 2: Starting point for Gonzalez Phase 1")
print("="*70)

rows2 = []
for ds in ['adult','diabetes','covertype']:
    X_full = DATASETS[ds]
    for N in [500, 1000, 5000]:
        X = X_full[:N]
        for k in [10, 20]:
            for op in [0.10, 0.20]:
                z = int(op*N)
                O = z
                print(f"\n  {ds} N={N} k={k} z={z}")
                for start_name, s_idx in [
                    ('centroid', centroid_start(X)),
                    ('farthest', farthest_start(X)),
                    ('random0',  random_start(X,0)),
                    ('random1',  random_start(X,1)),
                    ('random2',  random_start(X,2)),
                ]:
                    pool = gonzalez_pool(X, k+O, s_idx)
                    # use pool with drg1 local search
                    c_ls = drg1.local_search_fast(X, pool, k, z)
                    r = drg_mod.robust_radius(X, c_ls, z)
                    print(f"    {start_name:<12} r={r:.5f}")
                    rows2.append({'exp':'start_point','dataset':ds,'N':N,'k':k,
                                  'z':z,'op':op,'start':start_name,'radius':round(r,6)})

pd.DataFrame(rows + rows2).to_csv('explore_results.csv', index=False)
print(f"\n[saved → explore_results.csv]")

# ═══════════════════════════════════════════════════════════════════════════
# EXP 3: Multi-start (T=3 independent pools, take best)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 3: Multi-start vs single-start")
print("="*70)

rows3 = []
for ds in ['adult','diabetes','covertype']:
    X_full = DATASETS[ds]
    for N in [500, 1000, 5000]:
        X = X_full[:N]
        for k in [10, 20]:
            for op in [0.10, 0.20]:
                z = int(op*N); O = z
                r_single, t1 = oc_local_full(X, k, z, O)
                t0 = time.perf_counter()
                r_multi  = multi_start_oc(X, k, z, O, T=3)
                t3 = time.perf_counter()-t0
                gain = 100*(r_single-r_multi)/r_single
                print(f"  {ds} N={N:5d} k={k} z={z:5d}  "
                      f"single={r_single:.5f}  multi3={r_multi:.5f}  gain={gain:+.2f}%  t={t3:.2f}s")
                rows3.append({'exp':'multi_start','dataset':ds,'N':N,'k':k,'z':z,'op':op,
                               'r_single':round(r_single,6),'r_multi3':round(r_multi,6),
                               'gain_pct':round(gain,3),'time_s':round(t3,4)})

pd.DataFrame(rows + rows2 + rows3).to_csv('explore_results.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# EXP 4: O=z curve fit — knee point formula
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP 4: O/z ratio — does optimal O = α·z for some α?")
print("="*70)

rows4 = []
for ds in ['adult']:  # just adult for speed
    X = DATASETS[ds][:2000]
    k, op = 20, 0.10; z = int(op*2000)
    best_r = np.inf; best_ratio = None
    for alpha in [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        O_val = max(5, int(alpha*z))
        r, _ = oc_local_full(X, k, z, O_val)
        if r < best_r: best_r = r; best_ratio = alpha
        print(f"  α={alpha:.2f}  O={O_val}  r={r:.5f}{'  ← best' if r==best_r else ''}")
        rows4.append({'exp':'alpha_sweep','dataset':ds,'N':2000,'k':k,'z':z,
                      'alpha':alpha,'O':O_val,'radius':round(r,6)})
print(f"  Best α = {best_ratio}")

all_rows = rows + rows2 + rows3 + rows4
pd.DataFrame(all_rows).to_csv('explore_results.csv', index=False)
print(f"\n[Final: {len(all_rows)} rows → explore_results.csv]")
print("\nAll experiments complete.")
