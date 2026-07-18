# ================================================================
# BENCHMARK: Ding et al. [ESA 2019] vs OverclusterDRG + 1-Swap
# ================================================================
#
# Dataset: adult_final_dataset.py
#
# Ding et al. (corrected):
#   At each of k Gonzalez steps, sample uniformly from the z+1
#   farthest unassigned points instead of always taking the single
#   farthest (which is likely an outlier). Among z+1 farthest, at
#   most z are outliers so at least 1 is a genuine inlier center.
#   Per-step success probability >= 1/(z+1).
#
#   Trial count for 99% guarantee:
#     T = ceil(log(0.01) / log(1 - 1/(z+1)))  ~  (z+1) * ln(100)
#   This depends on z only, not k or N.
#   For large z this may be huge — stop the script if needed.
#   CSV flushes after every trial so partial results are always saved.
#
# OverclusterDRG + 1-Swap:
#   Single deterministic run. Centroid start → Gonzalez to k+DRG_O
#   → greedy init + 1-swap local search on pool.
#
# Both evaluated with z = floor(outlier_percent * N) outliers.
# ================================================================

import numpy as np
import pandas as pd
import ast
import math
import time

DATASET_FILE = "adult_final_dataset.py"
OUTPUT_CSV   = "benchmark_ding_vs_drg.csv"

DRG_O       = 20
TARGET_PROB = 0.99

target_sizes     = [500, 1000, 5000, 10000, 15000, 20000]
k_values         = [10, 20, 50]
outlier_percents = [0.10, 0.20]

# ================================================================
# DATA
# ================================================================

def load_data():
    try:
        with open(DATASET_FILE, "r") as f:
            data = ast.literal_eval(f.read())
        X = np.asarray(data, dtype=np.float32)
        print(f"[data] Loaded {DATASET_FILE}: {X.shape}")
        return X
    except Exception as e:
        raise RuntimeError(f"Could not load {DATASET_FILE}: {e}")

# ================================================================
# UTILITIES
# ================================================================

def _robust_r(min_d, z):
    if z == 0:
        return float(min_d.max())
    return float(np.partition(min_d, -(z + 1))[-(z + 1)])


def _centroid_start(X):
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())


def _pool_dists(X, pool):
    return np.column_stack([np.linalg.norm(X - X[c], axis=1) for c in pool])


def num_trials(z, prob=TARGET_PROB):
    """
    T = ceil(log(1-prob) / log(1 - 1/(z+1)))
    Per-trial success prob = 1/(z+1): one critical Gonzalez step
    must pick the inlier from the z+1 farthest candidates.
    Depends on z only, not k or N.
    """
    p = 1.0 / (z + 1)
    return math.ceil(math.log(1.0 - prob) / math.log(1.0 - p))

# ================================================================
# DING ET AL. — corrected Gonzalez step
# ================================================================

def _gonzalez_ding(X, k, z, rng):
    """
    Gonzalez where each step samples uniformly from the z+1 farthest
    unassigned points (not deterministically the single farthest).
    """
    N = len(X)
    start = _centroid_start(X)          # fixed start, randomness is in steps
    centers = [start]
    min_d = np.linalg.norm(X - X[start], axis=1).astype(np.float64)

    for _ in range(k - 1):
        # z+1 farthest candidates (indices into X)
        n_cand = min(z + 1, N - len(centers))
        top_idx = np.argpartition(min_d, -n_cand)[-n_cand:]
        nxt = int(rng.choice(top_idx))
        centers.append(nxt)
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1), out=min_d)

    return centers, min_d


def ding_et_al(X, k, z, seed=0):
    T   = num_trials(z)
    rng = np.random.default_rng(seed)

    best_r      = np.inf
    trial_radii = []

    t0 = time.perf_counter()
    for trial_idx in range(T):
        centers, min_d = _gonzalez_ding(X, k, z, rng)
        r = _robust_r(min_d, z)
        trial_radii.append(r)
        if r < best_r:
            best_r = r
    wall = time.perf_counter() - t0

    return best_r, wall, T, trial_radii

# ================================================================
# OVERCLUSTERDRG + 1-SWAP
# ================================================================

def _gonzalez_det(X, k, start):
    centers = [start]
    min_d = np.linalg.norm(X - X[start], axis=1).astype(np.float64)
    for _ in range(k - 1):
        nxt = int(min_d.argmax())
        centers.append(nxt)
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1), out=min_d)
    return centers, min_d


def _greedy_init(D, k, z):
    selected  = [0]
    min_d     = D[:, 0].copy()
    remaining = list(range(1, D.shape[1]))
    for _ in range(k - 1):
        best_r, best_qi = np.inf, None
        for qi in remaining:
            r = _robust_r(np.minimum(min_d, D[:, qi]), z)
            if r < best_r:
                best_r, best_qi = r, qi
        selected.append(best_qi)
        np.minimum(min_d, D[:, best_qi], out=min_d)
        remaining.remove(best_qi)
    return selected


def _local_search(D, sel, k, z, max_iters=200):
    sel     = list(sel)
    sel_set = set(sel)
    min_d   = D[:, sel].min(axis=1)
    best_r  = _robust_r(min_d, z)

    for _ in range(max_iters):
        improved = False
        for i in range(k):
            ci     = sel[i]
            others = [sel[j] for j in range(k) if j != i]
            base_d = D[:, others].min(axis=1) if others else np.full(len(D), np.inf)
            for qi in range(D.shape[1]):
                if qi in sel_set:
                    continue
                r = _robust_r(np.minimum(base_d, D[:, qi]), z)
                if r < best_r - 1e-10:
                    sel_set.discard(ci)
                    sel_set.add(qi)
                    sel[i]  = qi
                    min_d   = np.minimum(base_d, D[:, qi])
                    best_r  = r
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return sel, best_r


def overcluster_drg(X, k, z, O=DRG_O):
    t0 = time.perf_counter()
    pool, _ = _gonzalez_det(X, k + O, _centroid_start(X))
    D       = _pool_dists(X, pool)
    sel     = _greedy_init(D, k, z)
    _, r    = _local_search(D, sel, k, z)
    return r, time.perf_counter() - t0

# ================================================================
# MAIN
# ================================================================

def main():
    X_full = load_data()
    assert len(X_full) >= max(target_sizes), (
        f"Dataset too small: {len(X_full)} rows, need {max(target_sizes)}"
    )

    # load existing rows so we can resume after a stop
    try:
        existing = pd.read_csv(OUTPUT_CSV)
        rows = existing.to_dict("records")
        done_keys = set(zip(existing.N, existing.k, existing.outlier_pct))
        print(f"[resume] Found {len(rows)} existing rows in {OUTPUT_CSV}")
    except FileNotFoundError:
        rows = []
        done_keys = set()

    total = len(target_sizes) * len(k_values) * len(outlier_percents)
    done  = 0

    print(f"\n{'='*70}")
    print(f"  Ding et al. (corrected) vs OverclusterDRG + 1-Swap")
    print(f"  DRG_O={DRG_O}  |  Target prob={TARGET_PROB:.0%}")
    print(f"  NOTE: T = (z+1)*ln(100) — large z => many trials => slow")
    print(f"  Stop anytime; CSV saves after every config.")
    print(f"{'='*70}\n")

    for N in target_sizes:
        X = X_full[:N]

        for k in k_values:
            for op in outlier_percents:
                done += 1
                key = (N, k, op)
                if key in done_keys:
                    print(f"[{done:3d}/{total}]  N={N:6d} k={k:2d} z={int(op*N):5d} — skipped (already done)")
                    continue

                z = int(op * N)
                T = num_trials(z)

                print(f"[{done:3d}/{total}]  N={N:6d}  k={k:2d}  "
                      f"z={z:5d} ({op:.0%})  T={T:6d}", flush=True)

                # --- Ding et al. ---
                d_r, d_t, d_T, d_rs = ding_et_al(X, k, z, seed=N + k + z)
                print(f"         Ding : r={d_r:.6f}  t={d_t:.3f}s  "
                      f"[min={min(d_rs):.6f}  max={max(d_rs):.6f}]", flush=True)

                # --- OverclusterDRG ---
                g_r, g_t = overcluster_drg(X, k, z)
                print(f"         DRG  : r={g_r:.6f}  t={g_t:.3f}s  "
                      f"pool={k + DRG_O}", flush=True)

                gap     = g_r / d_r if d_r > 1e-12 else 1.0
                speedup = d_t / g_t if g_t > 1e-12 else float("inf")
                print(f"         gap={gap:.5f}  speedup={speedup:.1f}x\n", flush=True)

                rows.append({
                    "N":               N,
                    "k":               k,
                    "z":               z,
                    "outlier_pct":     op,
                    "drg_O":           DRG_O,
                    "ding_trials":     d_T,
                    "ding_radius":     round(d_r, 8),
                    "ding_time_s":     round(d_t, 6),
                    "ding_trial_min":  round(min(d_rs), 8),
                    "ding_trial_max":  round(max(d_rs), 8),
                    "ding_trial_mean": round(float(np.mean(d_rs)), 8),
                    "drg_radius":      round(g_r, 8),
                    "drg_time_s":      round(g_t, 6),
                    "gap_drg_vs_ding": round(gap, 6),
                    "speedup_drg_x":   round(speedup, 3),
                })

                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"  Saved {len(df)} rows → {OUTPUT_CSV}")
    if len(df):
        print(f"  DRG faster in  {(df.speedup_drg_x > 1).sum()}/{len(df)} configs")
        print(f"  DRG <= Ding r  {(df.gap_drg_vs_ding <= 1 + 1e-6).sum()}/{len(df)} configs")
        print(f"  Mean gap:       {df.gap_drg_vs_ding.mean():.5f}")
        print(f"  Max  gap:       {df.gap_drg_vs_ding.max():.5f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()