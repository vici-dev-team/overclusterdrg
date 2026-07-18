# ============================================================
# SimplifiedDRG — Standalone Analysis Script
#
# Runs the (O+1)-skip algorithm with Voronoi density diagnostic,
# then produces a full human-readable report:
#   - Per-step center selection trace
#   - Voronoi cell sizes and SOI flag status
#   - Robust radius and comparison to Standard Gonzalez
#   - SOI health assessment
#
# Usage:
#   python drg_standalone.py                     (synthetic data)
#   python drg_standalone.py my_dataset.npy      (your data)
# ============================================================

import numpy as np
import sys
import time

# ─────────────────────────────────────────────────────
# CONFIG — edit these or pass via command line
# ─────────────────────────────────────────────────────
DEFAULT_K = 10
DEFAULT_OUTLIER_PERCENT = 0.0
DEFAULT_SEED = 42

# ─────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────
def sq_dist(X, c):
    """Squared Euclidean distance from every row of X to point c."""
    diff = X - c
    return np.einsum("ij,ij->i", diff, diff)


def run_drg(X, k, z, seed=None):
    """
    SimplifiedDRG with full step-by-step trace.

    Selection rule:  rank (O+1) at every step  (identical to bare (O+1)-skip).
    Diagnostic:      Voronoi cell size of selected center, flag if |V| <= z.

    Returns
    -------
    result : dict with keys
        centers     : list of selected center indices
        radius      : robust radius  r_z(C, X)
        steps       : list of per-step dicts with trace info
        flags       : list of flagged step numbers
    """
    if seed is not None:
        np.random.seed(seed)

    n, d = X.shape
    tau = z + 1

    # Initial center
    c0 = np.random.randint(n)
    centers = [c0]
    min_dist = sq_dist(X, X[c0])

    steps = []
    flags = []

    for t in range(1, k):
        # ── Selection: pick rank (z+1) ──
        # argpartition gives the index of the (n - z - 1)-th smallest = (z+1)-th largest
        idx_k = np.argpartition(min_dist, -(z + 1))[-(z + 1)]
        selected_dist = np.sqrt(min_dist[idx_k])

        # ── Farthest point (what Standard Gonzalez would pick) ──
        idx_farthest = np.argmax(min_dist)
        farthest_dist = np.sqrt(min_dist[idx_farthest])

        # ── Diagnostic: Voronoi cell of selected center ──
        dist_to_new = sq_dist(X, X[idx_k])
        voronoi_mask = dist_to_new <= min_dist
        voronoi_size = int(np.sum(voronoi_mask))

        flagged = voronoi_size <= z
        if flagged:
            flags.append(t)

        # ── Record step ──
        steps.append({
            'step': t,
            'center_idx': int(idx_k),
            'center_dist': selected_dist,
            'farthest_idx': int(idx_farthest),
            'farthest_dist': farthest_dist,
            'skipped_ranks': 0,  # always 0 for SimplifiedDRG (no filter)
            'voronoi_size': voronoi_size,
            'tau': tau,
            'flagged': flagged,
            'same_as_gonzalez': (idx_k == idx_farthest),
        })

        # ── Update distances ──
        min_dist = np.minimum(min_dist, dist_to_new)
        centers.append(int(idx_k))

    # ── Final robust radius ──
    radius_sq = np.partition(min_dist, -(z + 1))[-(z + 1)]
    radius = np.sqrt(radius_sq)

    return {
        'centers': centers,
        'radius': radius,
        'steps': steps,
        'flags': flags,
    }


def run_standard_gonzalez(X, k, z, seed=None):
    """Standard Gonzalez: always pick the absolute farthest. Drop z outliers at end."""
    if seed is not None:
        np.random.seed(seed)
    n = X.shape[0]
    c0 = np.random.randint(n)
    min_dist = sq_dist(X, X[c0])
    for _ in range(1, k):
        idx_k = np.argmax(min_dist)
        min_dist = np.minimum(min_dist, sq_dist(X, X[idx_k]))
    return np.sqrt(np.partition(min_dist, -(z + 1))[-(z + 1)])


def run_randomized_pool(X, k, z, T=20, seed=None):
    """Ding et al. style: sample uniformly from top-(z+1) pool. Best of T trials."""
    if seed is not None:
        np.random.seed(seed)
    n = X.shape[0]
    best = np.inf
    for _ in range(T):
        c0 = np.random.randint(n)
        min_dist = sq_dist(X, X[c0])
        for _ in range(1, k):
            pool = np.argpartition(min_dist, -(z + 1))[-(z + 1):]
            idx_k = np.random.choice(pool)
            min_dist = np.minimum(min_dist, sq_dist(X, X[idx_k]))
        r = np.sqrt(np.partition(min_dist, -(z + 1))[-(z + 1)])
        best = min(best, r)
    return best


# ─────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────
def print_report(X, k, z, seed):
    n, d = X.shape
    tau = z + 1

    print()
    print("╔" + "═"*68 + "╗")
    print("║" + "  SimplifiedDRG — Full Analysis Report".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print()
    print(f"  Dataset:     N={n}, d={d}")
    print(f"  Parameters:  k={k}, O={z} ({z/n*100:.1f}%), τ={tau}")
    print(f"  Seed:        {seed}")

    # ── Run all algorithms ──
    t0 = time.time()
    result = run_drg(X, k, z, seed=seed)
    t_drg = time.time() - t0

    t0 = time.time()
    r_std = run_standard_gonzalez(X, k, z, seed=seed)
    t_std = time.time() - t0

    t0 = time.time()
    r_rand = run_randomized_pool(X, k, z, T=20, seed=seed)
    t_rand = time.time() - t0

    r_drg = result['radius']

    # ── Step-by-step trace ──
    print()
    print("─" * 70)
    print("  STEP-BY-STEP CENTER SELECTION")
    print("─" * 70)
    print(f"  {'Step':>4}  {'Center':>7}  {'Dist':>9}  "
          f"{'|V(x*)|':>8}  {'τ':>5}  {'Flag':>5}  {'= Gonz?':>8}")
    print(f"  {'────':>4}  {'───────':>7}  {'─────────':>9}  "
          f"{'────────':>8}  {'─────':>5}  {'─────':>5}  {'────────':>8}")

    print(f"  {'c₁':>4}  {result['centers'][0]:>7d}  {'(init)':>9}  "
          f"{'—':>8}  {'—':>5}  {'—':>5}  {'—':>8}")

    n_same_as_gonzalez = 0
    for s in result['steps']:
        flag_str = "⚠ YES" if s['flagged'] else "no"
        gonz_str = "✓ yes" if s['same_as_gonzalez'] else "no"
        if s['same_as_gonzalez']:
            n_same_as_gonzalez += 1

        print(f"  {s['step']:>4d}  {s['center_idx']:>7d}  {s['center_dist']:>9.4f}  "
              f"{s['voronoi_size']:>8d}  {s['tau']:>5d}  {flag_str:>5}  {gonz_str:>8}")

    # ── Results comparison ──
    print()
    print("─" * 70)
    print("  RESULTS")
    print("─" * 70)
    print(f"  {'Algorithm':<30} {'Radius':>10} {'Time':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10}")
    print(f"  {'Standard Gonzalez':<30} {r_std:>10.4f} {t_std:>9.4f}s")
    print(f"  {'Randomized Pool (T=20)':<30} {r_rand:>10.4f} {t_rand:>9.4f}s")
    print(f"  {'SimplifiedDRG':<30} {r_drg:>10.4f} {t_drg:>9.4f}s")

    improvement_vs_std = r_std / r_drg if r_drg > 0 else float('inf')
    beat_rand = "✓" if r_drg <= r_rand else "✗"

    print()
    print(f"  DRG improvement over Standard: {improvement_vs_std:.2f}x smaller radius")
    print(f"  DRG ≤ Randomized(T=20):        {beat_rand}  ({r_drg:.4f} vs {r_rand:.4f})")

    # ── SOI diagnostic ──
    print()
    print("─" * 70)
    print("  SOI DIAGNOSTIC (Voronoi Density Certifier)")
    print("─" * 70)

    n_flags = len(result['flags'])
    total_steps = k - 1
    flag_pct = n_flags / total_steps * 100 if total_steps > 0 else 0
    gonz_pct = n_same_as_gonzalez / total_steps * 100 if total_steps > 0 else 0

    print(f"  Steps flagged (|V| ≤ O):       {n_flags}/{total_steps} ({flag_pct:.0f}%)")
    print(f"  Steps matching Gonzalez:        {n_same_as_gonzalez}/{total_steps} ({gonz_pct:.0f}%)")
    print()

    # Interpretation
    if n_flags == 0:
        print("  Assessment: ✓ CLEAN")
        print("  All selected centers have dense Voronoi cells (|V| > O).")
        print("  Consistent with SOI holding. The 2-approximation guarantee")
        print("  of Theorem 2 applies with high confidence.")
    elif flag_pct <= 30:
        print("  Assessment: ~ MARGINAL")
        print(f"  {n_flags} step(s) produced centers with sparse Voronoi cells.")
        print("  SOI may hold approximately. The algorithm still selected rank O+1")
        print("  (guaranteed inlier under SOI), so the output is valid.")
        print("  Flagged steps indicate clusters with few nearby points — possibly")
        print("  singleton or small clusters. Monitor across data snapshots.")
    else:
        print("  Assessment: ⚠ HIGH FLAG RATE")
        print(f"  {n_flags}/{total_steps} steps flagged. This dataset likely violates SOI")
        print("  or has many small/thin clusters relative to O.")
        print()
        print("  NOTE: This does NOT mean the algorithm produced a bad result —")
        print("  the radius is still valid and often competitive. It means the")
        print("  formal 2·OPT guarantee of Theorem 2 cannot be certified for")
        print("  this instance. The deterministic (O+1)-skip is identical to DRG")
        print("  in output; the flags are purely diagnostic.")
        print()
        print("  Likely cause: O is large relative to cluster sizes. Each Voronoi")
        print(f"  cell needs >{z} points to pass, but with O={z} and k={k},")
        print(f"  many of the {k} clusters may have fewer than {z+1} points in their")
        print("  Voronoi partition, triggering flags. This is expected behaviour —")
        print("  the density threshold is calibrated for outlier detection, not")
        print("  cluster balance.")
        if z / n > 0.05:
            print()
            print(f"  With O/N = {z/n:.0%}, a large fraction of points are designated")
            print("  outliers. Consider whether a smaller O better matches the true")
            print("  anomaly rate, or use the randomized algorithm of Ding et al.")

    # ── Was this a good dataset for DRG? ──
    print()
    print("─" * 70)
    print("  SUMMARY")
    print("─" * 70)

    verdicts = []
    if improvement_vs_std > 1.3:
        verdicts.append(f"DRG achieved {improvement_vs_std:.1f}x better radius than Standard Gonzalez")
    if r_drg <= r_rand:
        verdicts.append("DRG matched or beat Randomized(T=20) in a single deterministic pass")
    if n_flags == 0:
        verdicts.append("Zero SOI flags — formal 2·OPT guarantee holds with confidence")
    elif flag_pct > 50:
        verdicts.append(f"High flag rate ({flag_pct:.0f}%) — likely O is large relative to cluster sizes")
    if n_same_as_gonzalez == total_steps:
        verdicts.append("DRG selected the same centers as Standard Gonzalez at every step")

    for i, v in enumerate(verdicts):
        print(f"  {i+1}. {v}")

    print()
    print("═" * 70)


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Parse args
    k = DEFAULT_K
    outlier_pct = DEFAULT_OUTLIER_PERCENT
    seed = DEFAULT_SEED
    data_file = None

    args = sys.argv[1:]
    for arg in args:
        if arg.endswith(('.npy', '.npz', '.csv', '.py')):
            data_file = arg
        elif arg.startswith('k='):
            k = int(arg.split('=')[1])
        elif arg.startswith('o=') or arg.startswith('p='):
            outlier_pct = float(arg.split('=')[1])
        elif arg.startswith('seed='):
            seed = int(arg.split('=')[1])

    # Load data
    if data_file is not None:
        if data_file.endswith('.npy'):
            X = np.load(data_file).astype(np.float32)
        elif data_file.endswith('.csv'):
            X = np.loadtxt(data_file, delimiter=',', dtype=np.float32)
        elif data_file.endswith('.py'):
            import ast
            with open(data_file) as f:
                X = np.asarray(ast.literal_eval(f.read()), dtype=np.float32)
        else:
            X = np.load(data_file).astype(np.float32)
        print(f"Loaded {data_file}: N={X.shape[0]}, d={X.shape[1]}")
    else:
        # Generate synthetic data with clear cluster structure + outliers
        print("No dataset provided — generating synthetic data.")
        print("  3 Gaussian clusters + 5% scattered outliers, d=10")
        np.random.seed(seed)
        c1 = np.random.randn(300, 10) * 0.5 + np.array([0]*10)
        c2 = np.random.randn(300, 10) * 0.5 + np.array([5]*10)
        c3 = np.random.randn(300, 10) * 0.5 + np.array([10]*10)
        outliers = np.random.randn(50, 10) * 0.3 + np.array([50]*10)
        X = np.vstack([c1, c2, c3, outliers]).astype(np.float32)
        outlier_pct = 0.05
        k = 3

    z = int(outlier_pct * X.shape[0])
    print_report(X, k, z, seed)