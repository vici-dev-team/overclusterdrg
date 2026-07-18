# ============================================================
# OVERCLUSTERING BENCHMARK (FINAL)
# Optimized Forward Greedy + Fast Local Search
# ============================================================

import numpy as np
import ast
import time
from scipy.spatial.distance import cdist

# ============================================================
# CONFIG
# ============================================================

DATASET_FILE  = "adult_final_dataset.py"

DATASET_SIZES    = [2000]
K_VALUES         = [10, 20]
OUTLIER_PERCENTS = [0.1, 0.2]

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    try:
        with open(DATASET_FILE, "r") as f:
            data = ast.literal_eval(f.read())
        return np.asarray(data, dtype=np.float32)
    except:
        print("[!] Dataset not found — using synthetic")
        return np.random.rand(30000, 10).astype(np.float32)

# ============================================================
# UTILITIES
# ============================================================

def robust_radius_fast(min_d, O):
    return float(np.partition(min_d, -O-1)[-O-1])

def compute_min_d(X, centers):
    return cdist(X, X[centers]).min(axis=1)

# ============================================================
# PHASE 1 — GONZALEZ
# ============================================================

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
# PHASE 2 — OPTIMIZED FORWARD GREEDY
# ============================================================

def forward_greedy_fast(X, Q, k, O):
    D_Q = cdist(X, X[Q])  # cache distances

    selected = [0]
    min_d = D_Q[:, 0].copy()
    remaining = list(range(1, len(Q)))

    for _ in range(k - 1):
        best_r = np.inf
        best_qi = None

        for qi in remaining:
            trial = np.minimum(min_d, D_Q[:, qi])
            r = robust_radius_fast(trial, O)

            if r < best_r:
                best_r = r
                best_qi = qi

        selected.append(best_qi)
        np.minimum(min_d, D_Q[:, best_qi], out=min_d)
        remaining.remove(best_qi)

    return [Q[i] for i in selected]

# ============================================================
# PHASE 2 — FAST LOCAL SEARCH
# ============================================================

def local_search_fast(X, Q, k, O, iters=5):
    D_Q = cdist(X, X[Q])

    # start from forward greedy
    selected = forward_greedy_fast(X, Q, k, O)
    selected_q = [Q.index(c) for c in selected]

    min_d = D_Q[:, selected_q].min(axis=1)
    best_r = robust_radius_fast(min_d, O)

    for _ in range(iters):
        improved = False

        for i in range(k):
            old_qi = selected_q[i]

            # distances without center i
            other = [q for j, q in enumerate(selected_q) if j != i]
            base_d = D_Q[:, other].min(axis=1)

            for qi in range(len(Q)):
                if qi in selected_q:
                    continue

                trial_d = np.minimum(base_d, D_Q[:, qi])
                r = robust_radius_fast(trial_d, O)

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

# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(X, k, O):
    results = {}

    # Phase 1
    Q = gonzalez(X, k + O)

    # Forward Greedy
    t0 = time.time()
    C_fwd = forward_greedy_fast(X, Q, k, O)
    t1 = time.time()
    r_fwd = robust_radius_fast(compute_min_d(X, C_fwd), O)
    results["Forward"] = (r_fwd, t1 - t0)

    # Local Search
    t0 = time.time()
    C_loc = local_search_fast(X, Q, k, O)
    t1 = time.time()
    r_loc = robust_radius_fast(compute_min_d(X, C_loc), O)
    results["LocalSearch"] = (r_loc, t1 - t0)

    return results

# ============================================================
# MAIN
# ============================================================

def main():
    X_full = load_data()
    N_full = len(X_full)

    print("\nOVERCLUSTERING (FORWARD vs LOCAL SEARCH)")
    print("=" * 80)
    print(f"{'N':>6} {'k':>4} {'O%':>5} {'z':>5} │ "
          f"{'Method':<12} {'Radius':>10} {'Time(s)':>9}")
    print("-" * 80)

    for N in DATASET_SIZES:
        if N > N_full:
            continue

        X = X_full[:N]

        for k in K_VALUES:
            for p in OUTLIER_PERCENTS:
                O = int(p * N)

                res = run_experiment(X, k, O)

                for name, (r, t) in res.items():
                    print(f"{N:>6} {k:>4} {int(p*100):>4}% {O:>5} │ "
                          f"{name:<12} {r:>10.4f} {t:>9.4f}")

                print("-" * 80)

    print("\nDone.")

# ============================================================

if __name__ == "__main__":
    main()