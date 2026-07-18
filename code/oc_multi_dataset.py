"""
OverclusterDRG + 1-Swap vs Ding et al. — multi-dataset benchmark.
Runs the same experiment as dingvOC.py across all available datasets.
One output CSV per dataset; resumes safely if interrupted.
"""
import numpy as np
import pandas as pd
import ast
import math
import time
import os

DRG_O       = 20
TARGET_PROB = 0.99

target_sizes     = [500, 1000, 5000, 10000, 15000, 20000]
k_values         = [10, 20, 50]
outlier_percents = [0.10, 0.20]

DATASETS = [
    ("bank_dataset.py",      "benchmark_bank.csv"),
    ("diabetes_dataset.py",  "benchmark_diabetes.csv"),
    ("shuttle_dataset.py",   "benchmark_shuttle.csv"),
    ("covertype_dataset.py", "benchmark_covertype.csv"),
]

# ── utilities (identical to dingvOC.py) ──────────────────────────────────────

def _robust_r(min_d, z):
    if z == 0:
        return float(min_d.max())
    return float(np.partition(min_d, -(z + 1))[-(z + 1)])

def _centroid_start(X):
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())

def _pool_dists(X, pool):
    return np.column_stack([np.linalg.norm(X - X[c], axis=1) for c in pool])

def num_trials(z, prob=TARGET_PROB):
    p = 1.0 / (z + 1)
    return math.ceil(math.log(1.0 - prob) / math.log(1.0 - p))

# ── Ding et al. ───────────────────────────────────────────────────────────────

def _gonzalez_ding(X, k, z, rng):
    N = len(X)
    start = _centroid_start(X)
    centers = [start]
    min_d = np.linalg.norm(X - X[start], axis=1).astype(np.float64)
    for _ in range(k - 1):
        n_cand = min(z + 1, N - len(centers))
        top_idx = np.argpartition(min_d, -n_cand)[-n_cand:]
        nxt = int(rng.choice(top_idx))
        centers.append(nxt)
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1), out=min_d)
    return centers, min_d

def ding_et_al(X, k, z, seed=0):
    T   = num_trials(z)
    rng = np.random.default_rng(seed)
    best_r, trial_radii = np.inf, []
    t0 = time.perf_counter()
    for _ in range(T):
        centers, min_d = _gonzalez_ding(X, k, z, rng)
        r = _robust_r(min_d, z)
        trial_radii.append(r)
        if r < best_r:
            best_r = r
    return best_r, time.perf_counter() - t0, T, trial_radii

# ── OverclusterDRG + 1-Swap ───────────────────────────────────────────────────

def _gonzalez_det(X, k, start):
    centers = [start]
    min_d = np.linalg.norm(X - X[start], axis=1).astype(np.float64)
    for _ in range(k - 1):
        nxt = int(min_d.argmax())
        centers.append(nxt)
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1), out=min_d)
    return centers, min_d

def _greedy_init(D, k, z):
    selected  = [0]
    min_d     = D[:, 0].copy()
    remaining = list(range(1, D.shape[1]))
    for _ in range(k - 1):
        best_r, best_qi = np.inf, None
        for qi in remaining:
            r = _robust_r(np.minimum(min_d, D[:, qi]), z)
            if r < best_r:
                best_r, best_qi = r, qi
        selected.append(best_qi)
        np.minimum(min_d, D[:, best_qi], out=min_d)
        remaining.remove(best_qi)
    return selected

def _local_search(D, sel, k, z, max_iters=200):
    sel     = list(sel)
    sel_set = set(sel)
    best_r  = _robust_r(D[:, sel].min(axis=1), z)
    for _ in range(max_iters):
        improved = False
        for i in range(k):
            ci     = sel[i]
            others = [sel[j] for j in range(k) if j != i]
            base_d = D[:, others].min(axis=1) if others else np.full(len(D), np.inf)
            for qi in range(D.shape[1]):
                if qi in sel_set:
                    continue
                r = _robust_r(np.minimum(base_d, D[:, qi]), z)
                if r < best_r - 1e-10:
                    sel_set.discard(ci); sel_set.add(qi)
                    sel[i] = qi
                    best_r = r
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return sel, best_r

def overcluster_drg(X, k, z, O=DRG_O):
    t0 = time.perf_counter()
    pool, _ = _gonzalez_det(X, k + O, _centroid_start(X))
    D       = _pool_dists(X, pool)
    sel     = _greedy_init(D, k, z)
    _, r    = _local_search(D, sel, k, z)
    return r, time.perf_counter() - t0

# ── per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(dataset_file, output_csv):
    if not os.path.exists(dataset_file):
        print(f"\n[SKIP] {dataset_file} not found — run prepare_datasets.py first\n")
        return

    print(f"\n{'='*70}")
    print(f"  Dataset : {dataset_file}")
    print(f"  Output  : {output_csv}")
    print(f"{'='*70}")

    with open(dataset_file) as f:
        X_full = np.asarray(ast.literal_eval(f.read()), dtype=np.float32)
    print(f"  Loaded {X_full.shape[0]:,} rows x {X_full.shape[1]} cols")

    max_N = max(n for n in target_sizes if n <= len(X_full))
    sizes = [n for n in target_sizes if n <= len(X_full)]
    if not sizes:
        print(f"  [SKIP] dataset too small (< {min(target_sizes)} rows)")
        return

    try:
        existing = pd.read_csv(output_csv)
        rows = existing.to_dict("records")
        done_keys = set(zip(existing.N, existing.k, existing.outlier_pct))
        print(f"  Resuming: {len(rows)} rows already in {output_csv}")
    except FileNotFoundError:
        rows, done_keys = [], set()

    total = len(sizes) * len(k_values) * len(outlier_percents)
    done  = 0

    for N in sizes:
        X = X_full[:N]
        for k in k_values:
            for op in outlier_percents:
                done += 1
                key = (N, k, op)
                if key in done_keys:
                    print(f"  [{done:3d}/{total}] N={N:6d} k={k:2d} z={int(op*N):5d} — skip")
                    continue

                z = int(op * N)
                T = num_trials(z)
                print(f"  [{done:3d}/{total}] N={N:6d} k={k:2d} z={z:5d} ({op:.0%})  T={T:6d}",
                      flush=True)

                d_r, d_t, d_T, d_rs = ding_et_al(X, k, z, seed=N + k + z)
                print(f"           Ding: r={d_r:.6f}  t={d_t:.3f}s", flush=True)

                g_r, g_t = overcluster_drg(X, k, z)
                print(f"           DRG : r={g_r:.6f}  t={g_t:.3f}s  pool={k+DRG_O}",
                      flush=True)

                gap     = g_r / d_r if d_r > 1e-12 else 1.0
                speedup = d_t / g_t if g_t > 1e-12 else float("inf")
                print(f"           gap={gap:.5f}  speedup={speedup:.1f}x\n", flush=True)

                rows.append({
                    "dataset":         dataset_file,
                    "N":               N,
                    "k":               k,
                    "z":               z,
                    "outlier_pct":     op,
                    "drg_O":           DRG_O,
                    "ding_trials":     d_T,
                    "ding_radius":     round(d_r, 8),
                    "ding_time_s":     round(d_t, 6),
                    "ding_trial_min":  round(min(d_rs), 8),
                    "ding_trial_max":  round(max(d_rs), 8),
                    "ding_trial_mean": round(float(np.mean(d_rs)), 8),
                    "drg_radius":      round(g_r, 8),
                    "drg_time_s":      round(g_t, 6),
                    "gap_drg_vs_ding": round(gap, 6),
                    "speedup_drg_x":   round(speedup, 3),
                })
                pd.DataFrame(rows).to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"  Saved {len(df)} rows → {output_csv}")
    if len(df):
        print(f"  DRG faster   : {(df.speedup_drg_x > 1).sum()}/{len(df)}")
        print(f"  DRG <= Ding r: {(df.gap_drg_vs_ding <= 1 + 1e-6).sum()}/{len(df)}")
        print(f"  Mean gap     : {df.gap_drg_vs_ding.mean():.5f}")

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for ds_file, out_csv in DATASETS:
        run_dataset(ds_file, out_csv)
    print("\nAll datasets done.")
