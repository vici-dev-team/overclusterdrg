import os
import ast
import csv
import math
import time
import numpy as np

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(REPO, 'datasets')
RESULTS  = os.path.join(REPO, 'results')

SIZES    = [500, 1000, 2000, 5000, 10000, 20000]
K_VALUES = [10, 20, 50]
Z_FRACS  = [0.10, 0.20]

DATASET_FILES = {
    'adult':     'adult_final_dataset.py',
    'diabetes':  'diabetes_dataset.py',
    'covertype': 'covertype_dataset.py',
}


def robust_r(min_d, z):
    return float(np.partition(min_d, -(z + 1))[-(z + 1)])


def centroid_start(X):
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())


def gonzalez_ding(X, k, z, rng):
    """Gonzalez-style with random tie-breaking among the z+1 farthest candidates."""
    start = centroid_start(X)
    centers = [start]
    min_d = np.linalg.norm(X - X[start], axis=1).astype(np.float64)
    for _ in range(k - 1):
        n_cand = min(z + 1, len(X) - len(centers))
        top = np.argpartition(min_d, -n_cand)[-n_cand:]
        nxt = int(rng.choice(top))
        centers.append(nxt)
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1), out=min_d)
    return centers, min_d


def ding_et_al(X, k, z, seed=0):
    """Run T trials; return (best_r, elapsed_s, T). T = ceil((z+1)*ln(100))."""
    T = math.ceil((z + 1) * math.log(100))
    rng = np.random.default_rng(seed)
    best_r = np.inf
    t0 = time.perf_counter()
    for _ in range(T):
        _, min_d = gonzalez_ding(X, k, z, rng)
        r = robust_r(min_d, z)
        if r < best_r:
            best_r = r
    return best_r, time.perf_counter() - t0, T


if __name__ == '__main__':
    os.makedirs(RESULTS, exist_ok=True)

    for name, fname in DATASET_FILES.items():
        path = os.path.join(DATASETS, fname)
        X_full = np.array(ast.literal_eval(open(path).read()), dtype=np.float32)
        out_csv = os.path.join(RESULTS, f'ding_results_{name}.csv')

        with open(out_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['N', 'k', 'outlier_pct', 'z', 'ding_radius', 'ding_time_s', 'trials'])

        sizes = [n for n in SIZES if n <= len(X_full)]
        total = len(sizes) * len(K_VALUES) * len(Z_FRACS)
        done = 0

        for N in sizes:
            X = X_full[:N]
            for k in K_VALUES:
                for op in Z_FRACS:
                    done += 1
                    z = int(op * N)
                    r, t, T = ding_et_al(X, k, z, seed=N + k + z)
                    print(f"[{done}/{total}] {name} N={N} k={k} z={z}: r={r:.4f} ({t:.1f}s) T={T}")
                    with open(out_csv, 'a', newline='') as f:
                        csv.writer(f).writerow([N, k, op, z, round(r, 8), round(t, 4), T])
