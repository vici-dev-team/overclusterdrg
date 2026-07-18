# ============================================================
# GONZALEZ K-CENTER WITH OUTLIERS: ALGORITHM COMPARISON
# Benchmark: N -> k -> Outlier %
# Output: gonzalez_kcenter_results_compared.csv
# ============================================================

import numpy as np
import ast
import time
import os

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------
DATASET_FILE = "adult_final_dataset.py"
OUTPUT_FILE = "drg_kcenter_results_compared.csv"

target_sizes = [500, 1000, 5000, 10000, 15000, 20000]
k_values = [10, 20, 50]
outlier_percents = [0.10, 0.20]

# Number of trials for the randomized algorithm
RANDOM_TRIALS_T = 5  

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

# ------------------------------------------------------------
# 4. ALGORITHMS
# ------------------------------------------------------------

# Approach 1: Standard Gonzalez (Ignores outliers during center selection)
def gonzalez_standard(X, k, z, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n = X.shape[0]
    idx = np.random.randint(n)
    min_dist = squared_euclidean_to_center(X, X[idx])

    for _ in range(1, k):
        idx_k = np.argmax(min_dist)
        new_dist = squared_euclidean_to_center(X, X[idx_k])
        min_dist = np.minimum(min_dist, new_dist)

    radius_sq = np.partition(min_dist, -(z + 1))[-(z + 1)]
    return np.sqrt(radius_sq)

# Approach 2: Deterministic O+1 (Bare skip, no diagnostic)
def gonzalez_deterministic_o_plus_1(X, k, z, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    n = X.shape[0]
    idx = np.random.randint(n)
    min_dist = squared_euclidean_to_center(X, X[idx])

    for _ in range(1, k):
        idx_k = np.argpartition(min_dist, -(z + 1))[-(z + 1)]
        new_dist = squared_euclidean_to_center(X, X[idx_k])
        min_dist = np.minimum(min_dist, new_dist)

    radius_sq = np.partition(min_dist, -(z + 1))[-(z + 1)]
    return np.sqrt(radius_sq)

# Approach 3: Randomized Pool O+1
def gonzalez_randomized_pool(X, k, z, T=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    n = X.shape[0]
    best_radius_sq = np.inf
    
    for trial in range(T):
        idx = np.random.randint(n)
        min_dist = squared_euclidean_to_center(X, X[idx])
        
        for _ in range(1, k):
            top_pool_indices = np.argpartition(min_dist, -(z + 1))[-(z + 1):]
            idx_k = np.random.choice(top_pool_indices)
            new_dist = squared_euclidean_to_center(X, X[idx_k])
            min_dist = np.minimum(min_dist, new_dist)
            
        radius_sq = np.partition(min_dist, -(z + 1))[-(z + 1)]
        if radius_sq < best_radius_sq:
            best_radius_sq = radius_sq
            
    return np.sqrt(best_radius_sq)

# Approach 4: SimplifiedDRG — (O+1)-skip + Voronoi density diagnostic
#
# Selection: identical to Approach 2 (rank O+1, always).
# Diagnostic: after each selection, compute |V(x*)| and flag if <= O.
#
# Returns: (robust_radius, num_flags, flag_details)
#   - num_flags: how many steps had |V(x*)| <= z  (potential SOI deviations)
#   - flag_details: list of (step, voronoi_size) for flagged steps
#
def simplified_drg(X, k, z, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n = X.shape[0]
    tau = z + 1

    # Pick initial center
    idx = np.random.randint(n)
    centers = [idx]
    min_dist = squared_euclidean_to_center(X, X[idx])

    flags = []

    for step in range(1, k):
        # --- Selection: rank O+1, identical to deterministic ---
        idx_k = np.argpartition(min_dist, -(z + 1))[-(z + 1)]

        # --- Diagnostic: Voronoi cell size of the selected center ---
        # V(x*) = { p : d(p, x*) <= d(p, c) for all c in C_current }
        # min_dist already holds d(p, C_current)^2 for each p.
        # We just need d(p, x*)^2 and compare.
        dist_to_new = squared_euclidean_to_center(X, X[idx_k])
        voronoi_size = int(np.sum(dist_to_new <= min_dist))

        if voronoi_size <= z:
            flags.append((step, voronoi_size))

        # --- Update distances ---
        min_dist = np.minimum(min_dist, dist_to_new)

        centers.append(idx_k)

    # Robust radius: remove z farthest
    radius_sq = np.partition(min_dist, -(z + 1))[-(z + 1)]
    return np.sqrt(radius_sq), len(flags), flags

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

    if not os.path.exists(OUTPUT_FILE):
        header = ("N,k,Outlier_Percent,z,"
                  "R_Standard,Time_Standard,"
                  "R_Deterministic,Time_Deterministic,"
                  "R_Randomized,Time_Randomized,"
                  "R_DRG,Time_DRG,DRG_Flags")
        with open(OUTPUT_FILE, "w") as f:
            f.write(header + "\n")
        print(f"Created output file: {OUTPUT_FILE}")
    else:
        print(f"Appending to existing file: {OUTPUT_FILE}")

    print("\nSTARTING ALGORITHM COMPARISON BENCHMARK")
    print("-" * 100)

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

        for k in k_values:
            print(f"  > k = {k}")

            for p in outlier_percents:
                z = int(p * current_N)
                base_seed = 42
                
                # 1. Standard Gonzalez
                t0 = time.time()
                r_std = gonzalez_standard(X, k, z, seed=base_seed)
                t_std = time.time() - t0
                
                # 2. Deterministic O+1
                t0 = time.time()
                r_det = gonzalez_deterministic_o_plus_1(X, k, z, seed=base_seed)
                t_det = time.time() - t0
                
                # 3. Randomized Pool
                t0 = time.time()
                r_rand = gonzalez_randomized_pool(X, k, z, T=RANDOM_TRIALS_T, seed=base_seed)
                t_rand = time.time() - t0

                # 4. SimplifiedDRG
                t0 = time.time()
                r_drg, n_flags, flag_details = simplified_drg(X, k, z, seed=base_seed)
                t_drg = time.time() - t0

                csv_line = (f"{current_N},{k},{p},{z},"
                            f"{r_std:.6f},{t_std:.6f},"
                            f"{r_det:.6f},{t_det:.6f},"
                            f"{r_rand:.6f},{t_rand:.6f},"
                            f"{r_drg:.6f},{t_drg:.6f},{n_flags}")
                
                append_to_file(OUTPUT_FILE, csv_line)

                flag_str = f" flags={flag_details}" if n_flags > 0 else ""
                print(f"    p={int(p*100)}% (z={z:4d}) | "
                      f"Std={r_std:.4f} ({t_std:.3f}s) | "
                      f"Det={r_det:.4f} ({t_det:.3f}s) | "
                      f"Rand={r_rand:.4f} ({t_rand:.3f}s) | "
                      f"DRG={r_drg:.4f} ({t_drg:.3f}s) [{n_flags} flags]{flag_str}")

    print("\nBenchmark completed. Check", OUTPUT_FILE)