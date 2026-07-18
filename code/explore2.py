import os
import sys
import ast
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(REPO, 'datasets')
RESULTS  = os.path.join(REPO, 'results')

from drg import gonzalez_pool, forward_greedy, local_search

DATASET_FILES = {
    'adult':     'adult_final_dataset.py',
    'diabetes':  'diabetes_dataset.py',
    'covertype': 'covertype_dataset.py',
}

# 24 base configs (4 N x 3 k x 2 outlier%) x 3 datasets = 72 total
CONFIGS = [
    (N, k, op)
    for N  in [500, 1000, 2000, 5000]
    for k  in [10, 20, 50]
    for op in [0.10, 0.20]
]


def _oc_local_overcount(X, k, z, O):
    """OC-Local with explicit pool overcount O (pool size = k+O, outlier budget = z)."""
    pool = gonzalez_pool(X, k + O)
    D = cdist(X, X[pool])
    sel = forward_greedy(D, k, z)
    sel, r = local_search(D, sel, k, z)
    return r


def _o_values(z):
    """Practical O sweep from 1 to 3*z (logarithmically spaced for large z)."""
    pts = list(range(1, min(21, z + 1)))
    for v in [25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]:
        if v <= 3 * z:
            pts.append(v)
    return sorted(set(pts))


def _dataset_stats(X, k, z):
    """Lightweight dataset fingerprint for the given (k, z) configuration."""
    N, d = X.shape

    # sparsity_ratio: how unevenly spaced are the gonzalez pool points?
    pool = gonzalez_pool(X, k + z)
    knn_dists = np.sort(cdist(X[pool], X), axis=1)[:, 1:6].mean(axis=1)
    sparsity_ratio = float(knn_dists.max() / (knn_dists.mean() + 1e-10))

    # separation: ratio of min between-center distance to mean within-cluster distance
    pool_k = pool[:k]
    D_kc = cdist(X, X[pool_k])
    within = D_kc.min(axis=1).mean()
    D_cc = cdist(X[pool_k], X[pool_k])
    np.fill_diagonal(D_cc, np.inf)
    between = float(D_cc.min())
    separation = between / (within + 1e-10)

    return {
        'N': N, 'd': int(d),
        'z_over_k': round(z / k, 4),
        'sparsity_ratio': round(sparsity_ratio, 4),
        'separation': round(separation, 4),
    }


if __name__ == '__main__':
    os.makedirs(RESULTS, exist_ok=True)

    datasets = {}
    for name, fname in DATASET_FILES.items():
        path = os.path.join(DATASETS, fname)
        datasets[name] = np.array(ast.literal_eval(open(path).read()), dtype=np.float32)
        print(f"Loaded {name}: {datasets[name].shape}")

    rows = []
    total = sum(1 for ds in datasets for (N, k, op) in CONFIGS if N <= len(datasets[ds]))
    done = 0

    for ds_name, X_full in datasets.items():
        for N, k, op in CONFIGS:
            if N > len(X_full):
                continue
            done += 1
            X = X_full[:N]
            z = int(op * N)

            stats = _dataset_stats(X, k, z)
            o_vals = _o_values(z)

            radii = {}
            for O_val in o_vals:
                radii[O_val] = _oc_local_overcount(X, k, z, O_val)

            best_r = min(radii.values())
            opt_O  = min(radii, key=radii.get)

            # knee: smallest O within 2% of best achievable radius
            knee_O = next((v for v in sorted(radii) if radii[v] <= best_r * 1.02), opt_O)

            print(f"[{done}/{total}] {ds_name} N={N} k={k} z={z}: "
                  f"opt_O={opt_O} knee_O={knee_O} best_r={best_r:.4f}")

            row = {
                'dataset': ds_name, 'N': N, 'k': k, 'outlier_pct': op, 'z': z,
                **stats,
                'opt_O': opt_O, 'opt_r': round(best_r, 6),
                'knee_O': knee_O,
                'opt_over_z': round(opt_O / z, 3) if z > 0 else None,
                'knee_over_z': round(knee_O / z, 3) if z > 0 else None,
            }
            for O_val, r in radii.items():
                row[f'r_O{O_val}'] = round(r, 6)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, 'explore2_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")
