# ============================================================
# OVERCLUSTER PHASE 2: EXHAUSTIVE vs GREEDY + LOCAL SEARCH
# ============================================================

import numpy as np
import pandas as pd
import ast
import time
import itertools
from scipy.spatial.distance import cdist

# ============================================================
# CONFIG
# ============================================================

DATASET_FILE = "adult_final_dataset.py"
OUTPUT_FILE  = "exhaustive_vs_local.csv"

DATASET_SIZES    = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
K_VALUES         = [10, 20, 50]
OUTLIER_PERCENTS = [0.1, 0.2]

MAX_O_EXACT = 20   # only run exhaustive if O <= this

# ============================================================
# DATA
# ============================================================

def load_data():
    try:
        with open(DATASET_FILE, "r") as f:
            data = ast.literal_eval(f.read())
        return np.asarray(data, dtype=np.float32)
    except:
        print("[!] Using synthetic data")
        return np.random.rand(5000, 10).astype(np.float32)

# ============================================================
# UTILITIES
# ============================================================

def robust_radius_fast(min_d, O):
    return float(np.partition(min_d, -O-1)[-O-1])

def compute_min_d(X, centers):
    return cdist(X, X[centers]).min(axis=1)

# ============================================================
# GONZALEZ (Phase 1)
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
# FORWARD GREEDY
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
            r = robust_radius_fast(trial, O)

            if r < best_r:
                best_r = r
                best_qi = qi

        selected.append(best_qi)
        np.minimum(min_d, D_Q[:, best_qi], out=min_d)
        remaining.remove(best_qi)

    return [Q[i] for i in selected]

# ============================================================
# LOCAL SEARCH (1-SWAP)
# ============================================================

def local_search_fast(X, Q, k, O, iters=5):
    D_Q = cdist(X, X[Q])

    selected = forward_greedy_fast(X, Q, k, O)
    selected_q = [Q.index(c) for c in selected]

    min_d = D_Q[:, selected_q].min(axis=1)
    best_r = robust_radius_fast(min_d, O)

    for _ in range(iters):
        improved = False

        for i in range(k):
            other = [q for j,q in enumerate(selected_q) if j != i]
            base_d = D_Q[:, other].min(axis=1)

            for qi in range(len(Q)):
                if qi in selected_q:
                    continue

                trial = np.minimum(base_d, D_Q[:, qi])
                r = robust_radius_fast(trial, O)

                if r < best_r:
                    selected_q[i] = qi
                    min_d = trial
                    best_r = r
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return [Q[i] for i in selected_q]

# ============================================================
# EXHAUSTIVE PHASE 2
# ============================================================

def exhaustive_phase2(X, Q, k, O):
    best_r = np.inf
    best_C = None

    for subset in itertools.combinations(range(len(Q)), k):
        C = [Q[i] for i in subset]
        min_d = compute_min_d(X, C)
        r = robust_radius_fast(min_d, O)

        if r < best_r:
            best_r = r
            best_C = C

    return best_C, best_r

# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(X, k, O):
    Q = gonzalez(X, k + O)

    results = {}

    # ---------- Local Search ----------
    t0 = time.time()
    C_loc = local_search_fast(X, Q, k, O)
    t1 = time.time()

    r_loc = robust_radius_fast(compute_min_d(X, C_loc), O)

    results["LocalSearch"] = (r_loc, t1 - t0)

    # ---------- Exhaustive ----------
    if O <= MAX_O_EXACT:
        t0 = time.time()
        C_opt, r_opt = exhaustive_phase2(X, Q, k, O)
        t1 = time.time()

        results["Exhaustive"] = (r_opt, t1 - t0)

        gap = r_loc / r_opt if r_opt > 0 else 1.0
        results["Gap"] = gap
    else:
        results["Exhaustive"] = (np.nan, np.nan)
        results["Gap"] = np.nan

    return results

# ============================================================
# MAIN
# ============================================================

def main():
    X_full = load_data()
    rows = []

    print("\nEXHAUSTIVE vs LOCAL SEARCH\n")
    print("=" * 80)

    for N in DATASET_SIZES:
        X = X_full[:N]

        for k in K_VALUES:
            for p in OUTLIER_PERCENTS:
                O = int(p * N)

                print(f"N={N}, k={k}, O={O}")

                res = run_experiment(X, k, O)

                r_loc, t_loc = res["LocalSearch"]
                r_opt, t_opt = res["Exhaustive"]
                gap = res["Gap"]

                print(f"  Local:      r={r_loc:.4f}  t={t_loc:.4f}")
                if not np.isnan(r_opt):
                    print(f"  Exhaustive: r={r_opt:.4f}  t={t_opt:.4f}")
                    print(f"  Gap:        {gap:.4f}")
                else:
                    print("  Exhaustive: skipped")

                rows.append({
                    "N": N,
                    "k": k,
                    "z": O,
                    "local_radius": r_loc,
                    "local_time": t_loc,
                    "opt_radius": r_opt,
                    "opt_time": t_opt,
                    "gap": gap
                })

                print("-" * 60)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved results to {OUTPUT_FILE}")

# ============================================================

if __name__ == "__main__":
    main()