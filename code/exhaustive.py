import os
import sys
import ast
import csv
import math
import time
import itertools
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(REPO, 'datasets')
RESULTS  = os.path.join(REPO, 'results')

from drg import gonzalez_pool, forward_greedy, local_search

SIZES    = list(range(50, 1050, 50))   # 50, 100, ..., 1000
K_VALUES = [10, 20, 50]
Z_FRACS  = [0.10, 0.20]

# Skip exhaustive when C(pool_size, k) exceeds this threshold
COMBO_LIMIT = 500_000


def exhaustive_phase2(X, pool, k, z):
    """
    Exhaustive optimal selection of k centers from pool.
    Tests all C(len(pool), k) subsets and returns (centers, radius).
    """
    pool_arr = np.array(pool)
    best_r = np.inf
    best_centers = None
    for subset in itertools.combinations(range(len(pool)), k):
        cidx = pool_arr[list(subset)]
        dists = cdist(X, X[cidx]).min(axis=1)
        r = float(np.partition(dists, -(z + 1))[-(z + 1)])
        if r < best_r:
            best_r = r
            best_centers = list(cidx)
    return best_centers, best_r


def _ncr(n, r):
    """C(n, r), returns early if result exceeds COMBO_LIMIT."""
    if r > n:
        return 0
    r = min(r, n - r)
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
        if result > COMBO_LIMIT:
            return result
    return result


if __name__ == '__main__':
    path = os.path.join(DATASETS, 'adult_final_dataset.py')
    X_full = np.array(ast.literal_eval(open(path).read()), dtype=np.float32)
    print(f"Loaded adult: {X_full.shape}")

    os.makedirs(RESULTS, exist_ok=True)
    out_csv = os.path.join(RESULTS, 'exhaustive_vs_local.csv')

    fieldnames = ['N', 'k', 'outlier_pct', 'z',
                  'local_radius', 'local_time', 'opt_radius', 'opt_time', 'gap']

    with open(out_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    sizes = [n for n in SIZES if n <= len(X_full)]
    total = len(sizes) * len(K_VALUES) * len(Z_FRACS)
    done = 0

    for N in sizes:
        X = X_full[:N]
        for k in K_VALUES:
            for op in Z_FRACS:
                done += 1
                z = int(op * N)

                # Shared pool for a fair comparison between local and exhaustive
                pool = gonzalez_pool(X, k + z)
                D = cdist(X, X[pool])

                t0 = time.perf_counter()
                sel = forward_greedy(D, k, z)
                sel, local_r = local_search(D, sel, k, z)
                local_t = time.perf_counter() - t0

                combos = _ncr(len(pool), k)
                if combos <= COMBO_LIMIT:
                    t0 = time.perf_counter()
                    _, opt_r = exhaustive_phase2(X, pool, k, z)
                    opt_t = time.perf_counter() - t0
                    gap = local_r / opt_r if opt_r > 0 else 1.0
                    status = f"gap={gap:.4f}"
                else:
                    opt_r = opt_t = float('nan')
                    gap = float('nan')
                    status = f"exhaustive skipped (C={combos:,})"

                print(f"[{done}/{total}] N={N} k={k} z={z}: local={local_r:.4f} {status}")

                row = {
                    'N': N, 'k': k, 'outlier_pct': op, 'z': z,
                    'local_radius': round(local_r, 8),
                    'local_time':   round(local_t, 4),
                    'opt_radius':   '' if math.isnan(opt_r) else round(opt_r, 8),
                    'opt_time':     '' if math.isnan(opt_t) else round(opt_t, 4),
                    'gap':          '' if math.isnan(gap) else round(gap, 6),
                }
                with open(out_csv, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
