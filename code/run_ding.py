"""Ding et al. (T trials, 99% guarantee) — runs in background."""
import numpy as np, ast, time, csv, math, os

TARGET_SIZES  = [500, 1000, 5000, 10000, 15000, 20000]
K_VALUES      = [10, 20, 50]
OUTLIER_PCTS  = [0.10, 0.20]

DATASETS = [
    ("diabetes_dataset.py",  "ding_results_diabetes.csv"),
    ("covertype_dataset.py", "ding_results_covertype.csv"),
]

def num_trials(z, prob=0.99):
    p = 1.0 / (z + 1)
    return math.ceil(math.log(1.0 - prob) / math.log(1.0 - p))

def robust_r(md, z):
    return float(np.partition(md, -(z+1))[-(z+1)])

def centroid_start(X):
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())

def gonzalez_ding(X, k, z, rng):
    N = len(X)
    start = centroid_start(X)
    centers = [start]
    md = np.linalg.norm(X - X[start], axis=1).astype(np.float64)
    for _ in range(k - 1):
        n_cand = min(z + 1, N - len(centers))
        top_idx = np.argpartition(md, -n_cand)[-n_cand:]
        nxt = int(rng.choice(top_idx))
        centers.append(nxt)
        np.minimum(md, np.linalg.norm(X - X[nxt], axis=1), out=md)
    return centers, md

def ding_et_al(X, k, z, seed=0):
    T = num_trials(z)
    rng = np.random.default_rng(seed)
    best_r, trial_radii = np.inf, []
    t0 = time.perf_counter()
    for _ in range(T):
        _, md = gonzalez_ding(X, k, z, rng)
        r = robust_r(md, z)
        trial_radii.append(r)
        if r < best_r: best_r = r
    return best_r, time.perf_counter() - t0, T

os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

for ds_file, out_csv in DATASETS:
    print(f"\n{'='*60}\n  {ds_file}\n{'='*60}")
    X_full = np.array(ast.literal_eval(open(ds_file).read()), dtype=np.float32)
    sizes = [n for n in TARGET_SIZES if n <= len(X_full)]
    total = len(sizes) * len(K_VALUES) * len(OUTLIER_PCTS); done = 0

    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['N','k','outlier_pct','z','ding_radius','ding_time_s','trials'])

    for N in sizes:
        X = X_full[:N]
        for k in K_VALUES:
            for op in OUTLIER_PCTS:
                done += 1; z = int(op * N)
                r, t, T = ding_et_al(X, k, z, seed=N+k+z)
                print(f"  [{done:3d}/{total}] N={N:6d} k={k:2d} z={z:5d} ({int(op*100)}%)  "
                      f"Ding r={r:.5f}  t={t:.1f}s  T={T}", flush=True)
                with open(out_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([N,k,op,z,round(r,8),round(t,4),T])

    print(f"\n  Done → {out_csv}")

print("\nDing complete.")
