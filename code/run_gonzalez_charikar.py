"""Gonzalez + Charikar on diabetes and covertype. Saves after every config."""
import sys, os
sys.path.insert(0, '/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

import numpy as np, pandas as pd, ast, time

# Import from existing scripts
from gonzalez import gonzalez_standard, squared_euclidean_to_center
from rkc import charikar_kcenter_outliers, charikar_feasible

TARGET_SIZES     = [500, 1000, 5000, 10000, 15000, 20000]
K_VALUES         = [10, 20, 50]
OUTLIER_PERCENTS = [0.10, 0.20]

def robust_r(X, centers, z):
    dists = np.array([squared_euclidean_to_center(X, X[c]) for c in centers])
    min_d = np.sqrt(dists.min(axis=0))
    return float(np.partition(min_d, -(z+1))[-(z+1)])

DATASETS = [
    ("diabetes_dataset.py",  "benchmark_gonz_char_diabetes.csv"),
    ("covertype_dataset.py", "benchmark_gonz_char_covertype.csv"),
]

for ds_file, out_csv in DATASETS:
    print(f"\n{'='*65}\n  {ds_file}\n{'='*65}")
    with open(ds_file) as f:
        X_full = np.asarray(ast.literal_eval(f.read()), dtype=np.float32)
    print(f"  Loaded {X_full.shape}")

    try:
        existing = pd.read_csv(out_csv)
        rows = existing.to_dict("records")
        done_keys = set(zip(existing.N, existing.k, existing.outlier_pct))
        print(f"  Resuming: {len(rows)} rows")
    except FileNotFoundError:
        rows, done_keys = [], set()

    sizes = [n for n in TARGET_SIZES if n <= len(X_full)]
    total = len(sizes)*len(K_VALUES)*len(OUTLIER_PERCENTS); idx=0

    for N in sizes:
        X = X_full[:N]
        for k in K_VALUES:
            for op in OUTLIER_PERCENTS:
                idx += 1
                key = (N, k, op)
                if key in done_keys:
                    print(f"  [{idx:3d}/{total}] skip"); continue
                z = int(op * N)
                print(f"  [{idx:3d}/{total}] N={N:6d} k={k:2d} z={z:5d} ({op:.0%})", flush=True)

                # Gonzalez
                t0 = time.perf_counter()
                g_r = gonzalez_standard(X, k, z, seed=42)
                g_t = time.perf_counter() - t0
                print(f"    Gonzalez: r={g_r:.6f}  t={g_t:.4f}s", flush=True)

                # Charikar
                t0 = time.perf_counter()
                c_r = charikar_kcenter_outliers(X, k, z)
                c_t = time.perf_counter() - t0
                if c_r is None: c_r = float('nan')
                print(f"    Charikar: r={c_r:.6f}  t={c_t:.3f}s", flush=True)

                rows.append({
                    "N": N, "k": k, "outlier_pct": op, "z": z,
                    "gonz_radius": round(g_r,8), "gonz_time_s": round(g_t,6),
                    "char_radius": round(float(c_r),8), "char_time_s": round(c_t,4),
                })
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\n  Gonzalez mean radius: {df.gonz_radius.mean():.4f}")
    print(f"  Charikar mean radius: {df.char_radius.mean():.4f}")
    print(f"  Saved → {out_csv}")

print("\nDone.")
