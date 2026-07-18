# ============================================================
# GONZALEZ K-CENTER WITH OUTLIERS
# Benchmark: N -> k -> Outlier %
# Output: gonzalez_kcenter_results.csv
# ============================================================

import numpy as np
import ast
import time
import os

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------
DATASET_FILE = "adult_final_dataset.py"
OUTPUT_FILE = "gonzalez_kcenter_results_updated.csv"

target_sizes = [500, 1000, 5000, 10000, 15000, 20000]
k_values = [10, 20, 50]
outlier_percents = [0.10, 0.20]

# ------------------------------------------------------------
# 2. DATA LOADING
# ------------------------------------------------------------
def load_data():
    try:
        with open(DATASET_FILE, "r") as f:
            data = ast.literal_eval(f.read())
        X = np.asarray(data, dtype=np.float32)
        print(f"Loaded '{DATASET_FILE}' with N={X.shape[0]}, d={X.shape[1]}")
        return X
    except FileNotFoundError:
        print("[!] Dataset not found. Generating synthetic data (N=30000, d=10)...")
        return np.random.rand(30000, 10).astype(np.float32)

# ------------------------------------------------------------
# 3. DISTANCE HELPER
# ------------------------------------------------------------
def squared_euclidean_to_center(X, c):
    diff = X - c
    return np.einsum("ij,ij->i", diff, diff)


def gonzalez_kcenter_outliers(X, k, z, seed=None):

    if seed is not None:
        np.random.seed(seed)

    n = X.shape[0]

    # Choose first center randomly
    idx = np.random.randint(n)
    min_dist = squared_euclidean_to_center(X, X[idx])

    for _ in range(1, k):
        idx_k = np.argpartition(min_dist, -(z + 1))[-(z + 1)]
        new_dist = squared_euclidean_to_center(X, X[idx_k])
        min_dist = np.minimum(min_dist, new_dist)

    radius_sq = np.partition(min_dist, -(z + 1))[-(z + 1)]
    return radius_sq

# ------------------------------------------------------------
# 5. FILE APPEND
# ------------------------------------------------------------
def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")

# ------------------------------------------------------------
# 6. MAIN BENCHMARK
# ------------------------------------------------------------
if __name__ == "__main__":

    X_full = load_data()
    MAX_N = X_full.shape[0]

    # Initialize CSV file
    if not os.path.exists(OUTPUT_FILE):
        header = "N,k,Outlier_Percent,z,Radius,Time"
        with open(OUTPUT_FILE, "w") as f:
            f.write(header + "\n")
        print(f"Created output file: {OUTPUT_FILE}")
    else:
        print(f"Appending to existing file: {OUTPUT_FILE}")

    print("\nSTARTING GONZALEZ BENCHMARK")
    print("-" * 60)

    # Loop: Dataset size
    for size_req in target_sizes:

        if size_req == 'full':
            current_N = MAX_N
        else:
            current_N = int(size_req)

        if current_N > MAX_N:
            print(f"Skipping N={current_N} (Dataset limit {MAX_N})")
            continue

        X = X_full[:current_N]
        print(f"\n[ Dataset Size: {current_N} ]")

        # Loop: k
        for k in k_values:
            print(f"  > k = {k}")

            # Loop: Outlier %
            for p in outlier_percents:

                z = int(p * current_N)

                t0 = time.time()
                radius_sq = gonzalez_kcenter_outliers(X, k, z, seed=42)
                radius = np.sqrt(radius_sq)
                elapsed = time.time() - t0

                # Save
                csv_line = f"{current_N},{k},{p},{z},{radius:.6f},{elapsed:.6f}"
                append_to_file(OUTPUT_FILE, csv_line)

                print(f"    Outliers {int(p*100)}% (z={z}) "
                      f"| Radius = {radius:.5f} "
                      f"| Time = {elapsed:.4f}s")

    print("\nBenchmark completed.")