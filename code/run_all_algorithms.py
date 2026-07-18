import os
import sys
import ast
import csv
import math
import time
import numpy as np

# allow running from repo root: python code/run_all_algorithms.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(REPO, 'datasets')
RESULTS  = os.path.join(REPO, 'results')

from drg import oc_greedy, oc_local
from gonzalez import rkc_gonzalez
from rkc import charikar_kcenter_outliers

try:
    from LP_relaxed import lp_lower_bound
    LP_AVAILABLE = True
except ImportError:
    LP_AVAILABLE = False

SIZES    = [500, 1000, 2000, 5000, 10000, 20000]
K_VALUES = [10, 20, 50]
Z_FRACS  = [0.10, 0.20]

DATASET_FILES = {
    'adult':     'adult_final_dataset.py',
    'diabetes':  'diabetes_dataset.py',
    'covertype': 'covertype_dataset.py',
}

# LP is expensive — only run for small N
LP_MAX_N = 500


def _fmt(v):
    return '' if isinstance(v, float) and math.isnan(v) else round(v, 8)


def run_config(X, k, z):
    row = {}

    t0 = time.perf_counter()
    row['gonz_r'] = rkc_gonzalez(X, k, z)
    row['gonz_t'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    row['char_r'] = charikar_kcenter_outliers(X, k, z)
    row['char_t'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, row['ocg_r'] = oc_greedy(X, k, z)
    row['ocg_t'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, row['ocl_r'] = oc_local(X, k, z)
    row['ocl_t'] = time.perf_counter() - t0

    if LP_AVAILABLE and len(X) <= LP_MAX_N:
        t0 = time.perf_counter()
        row['lp_r'] = lp_lower_bound(X, k, z)
        row['lp_t'] = time.perf_counter() - t0
    else:
        row['lp_r'] = float('nan')
        row['lp_t'] = float('nan')

    return row


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'adult'
    if name not in DATASET_FILES:
        raise SystemExit(f"Unknown dataset '{name}'. Choose from: {list(DATASET_FILES)}")

    path = os.path.join(DATASETS, DATASET_FILES[name])
    X_full = np.array(ast.literal_eval(open(path).read()), dtype=np.float32)
    print(f"Loaded {name}: {X_full.shape}")

    os.makedirs(RESULTS, exist_ok=True)
    out_csv = os.path.join(RESULTS, 'final_comparison.csv')

    fieldnames = ['dataset', 'N', 'k', 'outlier_pct', 'z',
                  'gonz_r', 'gonz_t', 'char_r', 'char_t',
                  'ocg_r', 'ocg_t', 'ocl_r', 'ocl_t', 'lp_r', 'lp_t']
    write_header = not os.path.exists(out_csv)

    sizes = [n for n in SIZES if n <= len(X_full)]
    total = len(sizes) * len(K_VALUES) * len(Z_FRACS)
    done = 0

    with open(out_csv, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for N in sizes:
            X = X_full[:N]
            for k in K_VALUES:
                for op in Z_FRACS:
                    done += 1
                    z = int(op * N)
                    t_start = time.perf_counter()
                    row = run_config(X, k, z)
                    elapsed = time.perf_counter() - t_start
                    print(f"[{done}/{total}] {name} N={N} k={k} z={z}: "
                          f"gonz={row['gonz_r']:.4f} char={row['char_r']:.4f} "
                          f"ocg={row['ocg_r']:.4f} ocl={row['ocl_r']:.4f} ({elapsed:.1f}s)")
                    writer.writerow({'dataset': name, 'N': N, 'k': k, 'outlier_pct': op, 'z': z,
                                     **{key: _fmt(row[key]) for key in row}})
                    fh.flush()
