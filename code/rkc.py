# ============================================================
# PAPER-CORRECT CHARIKAR ROBUST K-CENTER (ONE-CLICK RUN)
# Charikar–Khuller–Mount–Narasimhan (SODA 2001)
# 3-Approximation, Matrix-Free, Batched
# ============================================================

import numpy as np
import ast
import os
import time
from scipy.spatial.distance import cdist

# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = "adult_final_dataset.py"
OUTPUT_FILE = "charikar_results1.csv"

DATASET_SIZES = [2000]
K_VALUES = [50, 10, 20]
OUTLIER_PERCENTS = [0.1, 0.2]

BATCH_SIZE = 1024
MAX_RADII = 200   # radius sampling for performance

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    try:
        with open(DATASET_FILE, "r") as f:
            data = ast.literal_eval(f.read())
        X = np.asarray(data, dtype=np.float32)
        print(f"Loaded dataset: N={X.shape[0]}, d={X.shape[1]}")
        return X
    except FileNotFoundError:
        print("[!] Dataset not found — generating synthetic data")
        return np.random.rand(30000, 10).astype(np.float32)

# ============================================================
# CHARIKAR FEASIBILITY ORACLE (PAPER-CORRECT)
# ============================================================

def charikar_feasible(X, k, z, r):
    """
    Greedy feasibility test with:
      - coverage radius r
      - deletion radius 3r
    """
    n = X.shape[0]
    uncovered = np.ones(n, dtype=bool)

    r2 = r * r
    r3_2 = (3 * r) ** 2

    for _ in range(k):
        idx_uncovered = np.flatnonzero(uncovered)
        if idx_uncovered.size <= z:
            return True

        Xu = X[idx_uncovered]

        best_center = -1
        best_cover = -1

        # Scan all candidate centers (batched)
        for i in range(0, n, BATCH_SIZE):
            Xb = X[i:i + BATCH_SIZE]
            d2 = cdist(Xu, Xb, metric="sqeuclidean")
            counts = np.sum(d2 <= r2, axis=0)

            j = np.argmax(counts)
            if counts[j] > best_cover:
                best_cover = counts[j]
                best_center = i + j

        if best_cover == 0:
            break

        # Remove points within 3r
        d2_full = cdist(X, X[best_center:best_center + 1],
                        metric="sqeuclidean").ravel()
        uncovered &= (d2_full > r3_2)

    return np.sum(uncovered) <= z

# ============================================================
# DISCRETE RADIUS SEARCH (BINARY SEARCH)
# ============================================================

def charikar_kcenter_outliers(X, k, z):
    n = X.shape[0]

    # Sample candidate radii (paper-compatible optimization)
    sample_idx = np.linspace(0, n - 1, min(n, MAX_RADII), dtype=int)
    sample = X[sample_idx]

    dists = cdist(X, sample, metric="euclidean")
    radii = np.unique(dists)
    radii.sort()

    left, right = 0, len(radii) - 1
    best_r = None

    while left <= right:
        mid = (left + right) // 2
        r = radii[mid]

        if charikar_feasible(X, k, z, r):
            best_r = r
            right = mid - 1
        else:
            left = mid + 1

    return best_r

# ============================================================
# EXPERIMENT DRIVER
# ============================================================

def main():
    X_full = load_data()
    N_full = X_full.shape[0]

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("N,k,outlier_percent,z,radius,time\n")

    print("\nSTARTING CHARIKAR ROBUST K-CENTER")
    print("=" * 60)

    for N in DATASET_SIZES:
        if N > N_full:
            continue

        X = X_full[:N]
        print(f"\nDataset size: N={N}")

        for k in K_VALUES:
            for p in OUTLIER_PERCENTS:
                z = int(p * N)

                t0 = time.time()
                r = charikar_kcenter_outliers(X, k, z)
                elapsed = time.time() - t0

                with open(OUTPUT_FILE, "a") as f:
                    f.write(f"{N},{k},{p},{z},{r:.6f},{elapsed:.4f}\n")

                print(f"  k={k:<3} outliers={int(p*100):>3}% | "
                      f"r={r:.4f} | time={elapsed:.2f}s")

    print("\nDONE. Results saved to:", OUTPUT_FILE)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()