"""
Full benchmark: Gonzalez, Charikar (×3), OC-Local (drg.py, pool=k+z), Ding
Saves after every config. Two datasets run sequentially.
"""
import sys, os, ast, time, math
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

import drg as drg_mod
import drg1
from rkc import charikar_kcenter_outliers, charikar_feasible
from gonzalez import gonzalez_standard
import dingvOC as dingmod   # for Ding

TARGET_SIZES     = [500, 1000, 5000, 10000, 15000, 20000]
K_VALUES         = [10, 20, 50]
OUTLIER_PERCENTS = [0.10, 0.20]

def num_trials(z, prob=0.99):
    p = 1.0 / (z + 1)
    return math.ceil(math.log(1.0 - prob) / math.log(1.0 - p))

def oc_local(X, k, z):
    """drg.py correct implementation: pool = k+z, O=z as outlier budget."""
    centers, Q = drg_mod.overcluster_drg(X, k, O=z)
    centers_ls  = drg1.local_search_fast(X, Q, k, z)
    r = drg_mod.robust_radius(X, centers_ls, z)
    return r

def run_dataset(ds_file, out_csv):
    print(f"\n{'='*65}\n  {ds_file}\n{'='*65}")
    with open(ds_file) as f:
        X_full = np.asarray(ast.literal_eval(f.read()), dtype=np.float32)
    print(f"  Loaded {X_full.shape[0]:,} x {X_full.shape[1]}")

    try:
        existing = pd.read_csv(out_csv)
        rows     = existing.to_dict("records")
        done     = set(zip(existing.N, existing.k, existing.outlier_pct))
        print(f"  Resuming: {len(rows)} rows saved")
    except FileNotFoundError:
        rows, done = [], set()

    sizes = [n for n in TARGET_SIZES if n <= len(X_full)]
    total = len(sizes) * len(K_VALUES) * len(OUTLIER_PERCENTS); idx = 0

    for N in sizes:
        X = X_full[:N]
        for k in K_VALUES:
            for op in OUTLIER_PERCENTS:
                idx += 1
                if (N, k, op) in done:
                    print(f"  [{idx:3d}/{total}] skip"); continue
                z = int(op * N)
                T = num_trials(z)
                print(f"\n  [{idx:3d}/{total}] N={N:6d} k={k:2d} z={z:5d} ({int(op*100)}%)", flush=True)

                # ── Gonzalez ──────────────────────────────────────────
                t0 = time.perf_counter()
                g_r = gonzalez_standard(X, k, z, seed=42)
                g_t = time.perf_counter() - t0
                print(f"    Gonzalez : r={g_r:.6f}  t={g_t:.4f}s", flush=True)

                # ── Charikar (return r*, certified = 3r*) ────────────
                t0 = time.perf_counter()
                c_rstar = charikar_kcenter_outliers(X, k, z)
                c_t = time.perf_counter() - t0
                c_r3 = 3.0 * c_rstar if c_rstar else float('nan')
                print(f"    Charikar : r*={c_rstar:.6f}  3r*={c_r3:.6f}  t={c_t:.3f}s", flush=True)

                # ── OC-Local (drg.py, correct) ────────────────────────
                t0 = time.perf_counter()
                oc_r = oc_local(X, k, z)
                oc_t = time.perf_counter() - t0
                print(f"    OC-Local : r={oc_r:.6f}  t={oc_t:.3f}s", flush=True)

                # ── Ding (T trials, 99% guarantee) ────────────────────
                t0 = time.perf_counter()
                d_r, d_t, _, _ = dingmod.ding_et_al(X, k, z, seed=N+k+z)
                print(f"    Ding     : r={d_r:.6f}  t={d_t:.1f}s  T={T}", flush=True)

                # ── Summary ───────────────────────────────────────────
                best = min(g_r, c_r3, oc_r, d_r)
                print(f"    Winner   : {'Gonz' if best==g_r else 'Char3r*' if best==c_r3 else 'OC-Local' if best==oc_r else 'Ding'}")

                rows.append({
                    "N":N, "k":k, "outlier_pct":op, "z":z,
                    "gonz_r":    round(g_r,8),    "gonz_t":   round(g_t,4),
                    "char_rstar":round(c_rstar,8), "char_3r":  round(c_r3,8), "char_t":round(c_t,4),
                    "oc_r":      round(oc_r,8),    "oc_t":     round(oc_t,4),
                    "ding_r":    round(d_r,8),     "ding_t":   round(d_t,4), "ding_T": T,
                })
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\n  ── {ds_file} ──")
    print(f"  Mean Gonzalez:   {df.gonz_r.mean():.4f}")
    print(f"  Mean Char 3r*:   {df.char_3r.mean():.4f}")
    print(f"  Mean OC-Local:   {df.oc_r.mean():.4f}")
    print(f"  Mean Ding:       {df.ding_r.mean():.4f}")
    wins = (df.oc_r < df.char_3r).sum()
    print(f"  OC < Char(3r*):  {wins}/{len(df)} configs")
    wins = (df.oc_r < df.ding_r).sum()
    print(f"  OC < Ding:       {wins}/{len(df)} configs")
    print(f"  Saved → {out_csv}")

if __name__ == "__main__":
    run_dataset("diabetes_dataset.py",  "full_benchmark_diabetes.csv")
    run_dataset("covertype_dataset.py", "full_benchmark_covertype.csv")
    print("\nAll done.")
