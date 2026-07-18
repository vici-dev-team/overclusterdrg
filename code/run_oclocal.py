import os
import sys
import ast
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(REPO, 'datasets')
RESULTS  = os.path.join(REPO, 'results')

from drg import oc_local

SIZES    = [500, 1000, 2000, 5000, 10000, 20000]
K_VALUES = [10, 20, 50]
Z_FRACS  = [0.10, 0.20]

DATASET_FILES = {
    'diabetes':  'diabetes_dataset.py',
    'covertype': 'covertype_dataset.py',
}

if __name__ == '__main__':
    os.makedirs(RESULTS, exist_ok=True)

    for name, fname in DATASET_FILES.items():
        path = os.path.join(DATASETS, fname)
        X_full = np.array(ast.literal_eval(open(path).read()), dtype=np.float32)
        print(f"Loaded {name}: {X_full.shape}")

        out_csv = os.path.join(RESULTS, f'oc_results_{name}.csv')
        with open(out_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['N', 'k', 'outlier_pct', 'z', 'oc_radius', 'oc_time_s'])

        sizes = [n for n in SIZES if n <= len(X_full)]
        total = len(sizes) * len(K_VALUES) * len(Z_FRACS)
        done = 0

        for N in sizes:
            X = X_full[:N]
            for k in K_VALUES:
                for op in Z_FRACS:
                    done += 1
                    z = int(op * N)
                    t0 = time.perf_counter()
                    _, r = oc_local(X, k, z)
                    t = time.perf_counter() - t0
                    print(f"[{done}/{total}] {name} N={N} k={k} z={z}: r={r:.4f} ({t:.1f}s)")
                    with open(out_csv, 'a', newline='') as f:
                        csv.writer(f).writerow([N, k, op, z, round(r, 8), round(t, 4)])
