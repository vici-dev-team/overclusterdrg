# ============================================================
# DRG vs LP / Charikar / Ding RKC Benchmark (UPDATED)
# Overclustering replaced with Forward + Local Search
# ============================================================

import numpy as np
import pandas as pd
import ast
import time
from scipy.spatial.distance import cdist

# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = "adult_final_dataset.py"
LP_FILE      = "LP_Results.csv"
OUTPUT_FILE  = "final_comparison1.csv"

DATASET_SIZES    = [500, 1000, 2000, 5000, 10000, 20000]
K_VALUES         = [10, 20, 50]
OUTLIER_PERCENTS = [0.1, 0.2]

BATCH_SIZE = 1024
EPS = 1.0

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    try:
        with open(DATASET_FILE, "r") as f:
            data = ast.literal_eval(f.read())
        return np.asarray(data, dtype=np.float32)
    except:
        print("[!] Using synthetic data")
        return np.random.rand(30000, 10).astype(np.float32)

# ============================================================
# UTILITIES
# ============================================================

def robust_radius(X, centers_idx, O):
    n = len(X)
    min_d = np.full(n, np.inf, dtype=np.float32)

    for i in range(0, len(centers_idx), BATCH_SIZE):
        batch = X[centers_idx[i:i+BATCH_SIZE]]
        d = cdist(X, batch).min(axis=1)
        np.minimum(min_d, d, out=min_d)

    return float(np.partition(min_d, -O-1)[-O-1])


def gonzalez(X, k):
    mean = X.mean(axis=0, keepdims=True)
    c1 = int(cdist(X, mean).argmin())

    centers = [c1]
    min_d = cdist(X, X[[c1]]).ravel()

    for _ in range(k - 1):
        nxt = int(min_d.argmax())
        centers.append(nxt)
        d_new = cdist(X, X[[nxt]]).ravel()
        np.minimum(min_d, d_new, out=min_d)

    return centers

# ============================================================
# DRG ALGORITHMS
# ============================================================

def simplified_drg(X, k, O):
    mean = X.mean(axis=0, keepdims=True)
    c1 = int(cdist(X, mean).argmin())

    centers = [c1]
    min_d = cdist(X, X[[c1]]).ravel()

    for _ in range(k - 1):
        ranked = np.argsort(min_d)[::-1]
        nxt = int(ranked[O])
        centers.append(nxt)

        d_new = cdist(X, X[[nxt]]).ravel()
        np.minimum(min_d, d_new, out=min_d)

    return centers

# ============================================================
# OVERCLUSTERING (UPDATED)
# ============================================================

def forward_greedy_fast(X, Q, k, O):
    D_Q = cdist(X, X[Q])

    selected = [0]
    min_d = D_Q[:, 0].copy()
    remaining = list(range(1, len(Q)))

    for _ in range(k - 1):
        best_r = np.inf
        best_qi = None

        for qi in remaining:
            trial = np.minimum(min_d, D_Q[:, qi])
            r = float(np.partition(trial, -O-1)[-O-1])

            if r < best_r:
                best_r = r
                best_qi = qi

        selected.append(best_qi)
        np.minimum(min_d, D_Q[:, best_qi], out=min_d)
        remaining.remove(best_qi)

    return [Q[i] for i in selected]


def local_search_fast(X, Q, k, O, iters=5):
    D_Q = cdist(X, X[Q])

    selected = forward_greedy_fast(X, Q, k, O)
    selected_q = [Q.index(c) for c in selected]

    min_d = D_Q[:, selected_q].min(axis=1)
    best_r = float(np.partition(min_d, -O-1)[-O-1])

    for _ in range(iters):
        improved = False

        for i in range(k):
            other = [q for j, q in enumerate(selected_q) if j != i]
            base_d = D_Q[:, other].min(axis=1)

            for qi in range(len(Q)):
                if qi in selected_q:
                    continue

                trial_d = np.minimum(base_d, D_Q[:, qi])
                r = float(np.partition(trial_d, -O-1)[-O-1])

                if r < best_r:
                    selected_q[i] = qi
                    min_d = trial_d
                    best_r = r
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return [Q[i] for i in selected_q]


def overcluster_forward(X, k, O):
    Q = gonzalez(X, k + O)
    return forward_greedy_fast(X, Q, k, O)


def overcluster_local(X, k, O):
    Q = gonzalez(X, k + O)
    return local_search_fast(X, Q, k, O)

# ============================================================
# DING ET AL.
# ============================================================

def ding_rkc_multi(X, k, O, eps=EPS, runs=20):
    best_C = None
    best_r = np.inf

    for _ in range(runs):
        # --- one run ---
        n = len(X)
        c1 = np.random.randint(n)

        centers = [c1]
        min_d = cdist(X, X[[c1]]).ravel()

        for _ in range(k - 1):
            Q_size = int((1 + eps) * O)
            ranked = np.argsort(min_d)[::-1]
            Q = ranked[:Q_size]

            nxt = int(np.random.choice(Q))
            centers.append(nxt)

            d_new = cdist(X, X[[nxt]]).ravel()
            np.minimum(min_d, d_new, out=min_d)

        # --- evaluate ---
        r = float(np.partition(min_d, -O-1)[-O-1])

        if r < best_r:
            best_r = r
            best_C = centers

    return best_C

# ============================================================
# RUN GRID
# ============================================================

def run_drg_grid(X_full):
    rows = []

    for N in DATASET_SIZES:
        X = X_full[:N]

        for k in K_VALUES:
            for p in OUTLIER_PERCENTS:
                O = int(p * N)

                print(f"Running: N={N}, k={k}, O={O}")

                # SimplifiedDRG
                t0 = time.time()
                C1 = simplified_drg(X, k, O)
                t1 = time.time()
                r1 = robust_radius(X, C1, O)

                # Overcluster Forward
                t2 = time.time()
                C2 = overcluster_forward(X, k, O)
                t3 = time.time()
                r2 = robust_radius(X, C2, O)

                # Overcluster Local Search
                t4 = time.time()
                C3 = overcluster_local(X, k, O)
                t5 = time.time()
                r3 = robust_radius(X, C3, O)

                # Ding RKC
                t6 = time.time()
                C4 = ding_rkc_multi(X, k, O, runs = 20)
                t7 = time.time()
                r4 = robust_radius(X, C4, O)

                rows.append({
                    "N": N,
                    "k": k,
                    "z": O,

                    "SimplifiedDRG_radius": r1,
                    "SimplifiedDRG_time": t1 - t0,

                    "OC_Forward_radius": r2,
                    "OC_Forward_time": t3 - t2,

                    "OC_Local_radius": r3,
                    "OC_Local_time": t5 - t4,

                    "DingRKC_radius": r4,
                    "DingRKC_time": t7 - t6,
                })

    return pd.DataFrame(rows)

# ============================================================
# MERGE
# ============================================================

def merge_results(drg_df):
    lp_df = pd.read_csv(LP_FILE)

    if "z" not in lp_df.columns:
        lp_df["z"] = (lp_df["outlier_percent"] * lp_df["N"]).astype(int)

    merged = pd.merge(
        lp_df,
        drg_df,
        on=["N", "k", "z"],
        how="inner"
    )

    return merged

# ============================================================
# MAIN
# ============================================================

def main():
    X = load_data()

    drg_df = run_drg_grid(X)
    final_df = merge_results(drg_df)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Done!")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()