# ============================================================
# DRG ALGORITHMS BENCHMARK
# SimplifiedDRG + OverclusterDRG (Forward Greedy Phase 2)
# Matches Charikar script interface for direct comparison
# ============================================================

import numpy as np
import ast
import os
import time
from scipy.spatial.distance import cdist

# ============================================================
# CONFIG  (mirror Charikar script exactly)
# ============================================================

DATASET_FILE  = "adult_final_dataset.py"
OUTPUT_FILE   = "drg_results.csv"

DATASET_SIZES    = [2000]
K_VALUES         = [50, 10, 20]
OUTLIER_PERCENTS = [0.1, 0.2]

BATCH_SIZE = 1024   # for large-n distance batching

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
        print("[!] Dataset not found — generating synthetic data (N=30000, d=10)")
        return np.random.rand(30000, 10).astype(np.float32)

# ============================================================
# SHARED UTILITIES
# ============================================================

def robust_radius(X, centers_idx, O):
    """r_O(C, X): (O+1)-th largest distance from any point to nearest center."""
    C = X[centers_idx]
    # batched to avoid O(n*k) memory spike for large n
    n = len(X)
    min_d = np.full(n, np.inf, dtype=np.float32)
    for i in range(0, len(centers_idx), BATCH_SIZE):
        batch = X[centers_idx[i:i+BATCH_SIZE]]
        d = cdist(X, batch, metric="euclidean").min(axis=1).astype(np.float32)
        np.minimum(min_d, d, out=min_d)
    sorted_d = np.sort(min_d)[::-1]
    return float(sorted_d[O]) if O < n else 0.0

def gonzalez(X, k, c1=None):
    """Standard Gonzalez greedy k-center. Returns list of k center indices."""
    n = len(X)
    if c1 is None:
        # medoid-approximation: point nearest to dataset mean
        mean = X.mean(axis=0, keepdims=True)
        c1 = int(cdist(X, mean, metric="euclidean").argmin())
    centers = [c1]
    min_d = cdist(X, X[[c1]], metric="euclidean").ravel()
    for _ in range(k - 1):
        nxt = int(min_d.argmax())
        centers.append(nxt)
        d_new = cdist(X, X[[nxt]], metric="euclidean").ravel()
        np.minimum(min_d, d_new, out=min_d)
    return centers

def voronoi_sizes(X, centers_idx):
    """Return Voronoi cell sizes for each center."""
    C = X[centers_idx]
    assignments = cdist(X, C, metric="euclidean").argmin(axis=1)
    return np.bincount(assignments, minlength=len(centers_idx))

# ============================================================
# ALGORITHM 1 — SimplifiedDRG
# Deterministic O(nk), 2-approx under SOI
# ============================================================

def simplified_drg(X, k, O):
    """
    At each step select the (O+1)-th farthest point from current centers.
    No density filtering — selection loop is identical to Gonzalez except
    for the rank offset.
    """
    n = len(X)
    mean = X.mean(axis=0, keepdims=True)
    c1   = int(cdist(X, mean, metric="euclidean").argmin())
    centers = [c1]
    min_d = cdist(X, X[[c1]], metric="euclidean").ravel()

    for _ in range(k - 1):
        # rank all points descending; pick rank O+1 (0-indexed: index O)
        ranked = np.argsort(min_d)[::-1]
        nxt    = int(ranked[O])
        centers.append(nxt)
        d_new  = cdist(X, X[[nxt]], metric="euclidean").ravel()
        np.minimum(min_d, d_new, out=min_d)

    return centers

# ============================================================
# ALGORITHM 2 — OverclusterDRG (Forward Greedy Phase 2)
# O(nk(k+O)), unconditional ~3·OPT
# ============================================================

def overcluster_drg(X, k, O):
    """
    Phase 1: Gonzalez for k+O steps → candidate pool Q.
    Phase 2: Forward greedy selection of k centers from Q,
             at each step picking the candidate minimising r_O.

    Note: Claim 4 proof gap exists (see paper audit). Algorithm
    is empirically correct; proof holds for exhaustive Phase 2
    (see overcluster_drg_exact for the provably correct variant).
    """
    # ── Phase 1 ──────────────────────────────────────────────
    Q = gonzalez(X, k + O)          # k+O candidates
    X_Q = X[Q]                      # shape (k+O, d)

    # ── Phase 2: densest-first initialisation ────────────────
    vsizes = voronoi_sizes(X, Q)
    s0_pos = int(vsizes.argmax())   # index within Q
    selected    = [Q[s0_pos]]       # global indices
    remaining_Q = [i for i in range(len(Q)) if i != s0_pos]  # Q-local indices

    min_d = cdist(X, X[selected], metric="euclidean").ravel()

    # ── Phase 2: forward greedy ──────────────────────────────
    for _ in range(k - 1):
        best_r, best_qi = np.inf, None

        for qi in remaining_Q:
            q_global = Q[qi]
            trial_d  = np.minimum(min_d,
                                  cdist(X, X[[q_global]],
                                        metric="euclidean").ravel())
            r = float(np.sort(trial_d)[::-1][O])
            if r < best_r:
                best_r  = r
                best_qi = qi

        best_global = Q[best_qi]
        selected.append(best_global)
        remaining_Q.remove(best_qi)
        d_new = cdist(X, X[[best_global]], metric="euclidean").ravel()
        np.minimum(min_d, d_new, out=min_d)

    return selected, Q

# ============================================================
# ALGORITHM 3 — OverclusterDRG Exact (exhaustive Phase 2)
# Provably 3·OPT for fixed O; O(nk^O) complexity
# ============================================================

def overcluster_drg_exact(X, k, O):
    """
    Phase 1: Gonzalez for k+O steps.
    Phase 2: Exhaustive best k-subset of Q.
    Provably correct: Claims 1-3 directly give r_O ≤ 3·OPT.
    Practical for small O (O=1: O(nk), O=2: O(nk²)).
    """
    from itertools import combinations

    Q = gonzalez(X, k + O)

    best_r, best_C = np.inf, None
    for subset in combinations(range(len(Q)), k):
        cidx = [Q[i] for i in subset]
        r    = robust_radius(X, cidx, O)
        if r < best_r:
            best_r  = r
            best_C  = cidx

    return best_C, Q

# ============================================================
# SOI DIAGNOSTIC
# ============================================================

def soi_ratio(X, centers_idx, O):
    """
    Estimate SOI ratio = gap(Z_hat, X\\Z_hat) / diam(X\\Z_hat).
    Uses the output outlier set as proxy for Z*.
    SOI holds when ratio > 1.
    """
    min_d = cdist(X, X[centers_idx], metric="euclidean").min(axis=1)
    out_idx = np.argsort(min_d)[::-1][:O].tolist()
    inl_idx = [i for i in range(len(X)) if i not in set(out_idx)]
    if not out_idx or not inl_idx:
        return float("inf")
    gap  = cdist(X[out_idx], X[inl_idx], metric="euclidean").min()
    diam = cdist(X[inl_idx], X[inl_idx], metric="euclidean").max()
    return float(gap / diam) if diam > 1e-12 else float("inf")

# ============================================================
# EXPERIMENT DRIVER
# ============================================================

def run_experiment(X, k, O, use_exact=False):
    """Run both DRG variants and return timing + radius."""
    results = {}

    # ── SimplifiedDRG ─────────────────────────────────────
    t0 = time.time()
    C_sdrg = simplified_drg(X, k, O)
    t1 = time.time()
    r_sdrg = robust_radius(X, C_sdrg, O)
    soi    = soi_ratio(X, C_sdrg, O)
    results["SimplifiedDRG"] = {
        "radius": r_sdrg,
        "time":   t1 - t0,
        "soi":    soi,
        "centers": C_sdrg,
    }

    # ── OverclusterDRG (forward greedy) ───────────────────
    t0 = time.time()
    C_ocdrg, Q_ocdrg = overcluster_drg(X, k, O)
    t1 = time.time()
    r_ocdrg = robust_radius(X, C_ocdrg, O)
    results["OverclusterDRG"] = {
        "radius":  r_ocdrg,
        "time":    t1 - t0,
        "centers": C_ocdrg,
        "pool_size": len(Q_ocdrg),
    }

    # ── OverclusterDRG Exact (exhaustive, only for small k+O) ─
    if use_exact and k + O <= 25:
        t0 = time.time()
        C_exact, _ = overcluster_drg_exact(X, k, O)
        t1 = time.time()
        r_exact = robust_radius(X, C_exact, O)
        results["OverclusterDRG_Exact"] = {
            "radius": r_exact,
            "time":   t1 - t0,
            "centers": C_exact,
        }

    return results

def main():
    X_full = load_data()
    N_full = X_full.shape[0]

    # CSV header
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("N,k,outlier_percent,z,algorithm,radius,time,soi_ratio\n")

    print("\nDRG ALGORITHMS BENCHMARK")
    print("=" * 70)
    print(f"{'N':>6} {'k':>4} {'O%':>5} {'z':>5} │ "
          f"{'Algorithm':<22} {'Radius':>10} {'Time(s)':>9} {'SOI':>7}")
    print("─" * 70)

    for N in DATASET_SIZES:
        if N > N_full:
            print(f"[!] N={N} exceeds dataset size {N_full}, skipping")
            continue

        X = X_full[:N]

        for k in K_VALUES:
            for p in OUTLIER_PERCENTS:
                O = int(p * N)

                # use exhaustive only when k+O is small enough to be fast
                use_exact = (k + O) <= 20

                res = run_experiment(X, k, O, use_exact=use_exact)

                for alg, info in res.items():
                    soi_val = info.get("soi", float("nan"))
                    soi_str = f"{soi_val:.3f}" if not np.isinf(soi_val) else "inf"

                    print(f"{N:>6} {k:>4} {int(p*100):>4}% {O:>5} │ "
                          f"{alg:<22} {info['radius']:>10.4f} "
                          f"{info['time']:>9.4f} {soi_str:>7}")

                    with open(OUTPUT_FILE, "a") as f:
                        f.write(
                            f"{N},{k},{p},{O},{alg},"
                            f"{info['radius']:.6f},"
                            f"{info['time']:.6f},"
                            f"{soi_val:.4f}\n"
                        )

                print("─" * 70)

    print(f"\nDone. Results saved to: {OUTPUT_FILE}")
    print("\nTo compare with Charikar:")
    print("  join drg_results.csv + charikar_results1.csv on (N, k, z)")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()