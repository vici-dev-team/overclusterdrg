"""
Estimate LP bounds for new datasets by applying the empirical LP/Gonzalez
ratio learned from LP_Results.csv (adult dataset).

Method:
  - For each (k, outlier_percent) bucket, compute mean LP/Gonzalez ratio
    from adult data  →  very stable (0.74–0.84 range).
  - Run fast Gonzalez + Charikar on each new dataset.
  - Estimated LP_Radius  =  gonzalez_radius * ratio(k, op)
  - LP_Time is extrapolated via power-law fit on adult LP times.

Output: lp_estimated_<dataset>.csv  (same columns as LP_Results.csv)
"""

import numpy as np
import pandas as pd
import ast, time, os
from scipy.spatial.distance import cdist

# ── config ────────────────────────────────────────────────────────────────────

SOURCE_LP_CSV = "LP_Results.csv"

DATASETS = [
    ("bank_dataset.py",      "lp_estimated_bank.csv"),
    ("diabetes_dataset.py",  "lp_estimated_diabetes.csv"),
    ("shuttle_dataset.py",   "lp_estimated_shuttle.csv"),
    ("covertype_dataset.py", "lp_estimated_covertype.csv"),
]

target_sizes     = [500, 1000, 5000, 10000, 15000, 20000]
k_values         = [10, 20, 50]
outlier_percents = [0.10, 0.20]

BATCH_SIZE = 1024
MAX_RADII  = 200

# ── learn ratios from adult LP data ──────────────────────────────────────────

def fit_ratios(csv_path):
    df = pd.read_csv(csv_path)
    df["ratio_lp_gonz"]  = df["LP_Radius"]      / df["gonzalez_radius"]
    df["ratio_lp_char"]  = df["LP_Radius"]      / df["charikar_radius"]
    # mean ratio per (k, outlier_percent)
    ratios = (df.groupby(["k", "outlier_percent"])[["ratio_lp_gonz", "ratio_lp_char"]]
                .mean().reset_index())
    print("\n[ratios] LP/Gonzalez per (k, outlier_pct) from adult data:")
    print(ratios.to_string(index=False))
    return ratios

def get_ratio(ratios_df, k, op):
    row = ratios_df[(ratios_df.k == k) &
                    (np.isclose(ratios_df.outlier_percent, op, atol=1e-4))]
    if len(row):
        return float(row.ratio_lp_gonz.values[0]), float(row.ratio_lp_char.values[0])
    # fallback: overall mean
    return float(ratios_df.ratio_lp_gonz.mean()), float(ratios_df.ratio_lp_char.mean())

# ── power-law LP time estimate ────────────────────────────────────────────────
# fit log(LP_time) ~ a + b*log(N) + c*log(k) separately per outlier_pct

def fit_time_model(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["LP_Time"] > 0].copy()
    df["logN"]  = np.log(df["N"])
    df["logk"]  = np.log(df["k"])
    df["logT"]  = np.log(df["LP_Time"])
    df["logop"] = np.log(df["outlier_percent"])
    A = np.column_stack([np.ones(len(df)), df["logN"], df["logk"], df["logop"]])
    coef, *_ = np.linalg.lstsq(A, df["logT"], rcond=None)
    return coef   # [intercept, b_N, b_k, b_op]

def estimate_lp_time(coef, N, k, op):
    logT = coef[0] + coef[1]*np.log(N) + coef[2]*np.log(k) + coef[3]*np.log(op)
    return round(float(np.exp(logT)), 4)

# ── algorithms ────────────────────────────────────────────────────────────────

def gonzalez_radius(X, k, z):
    start = int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())
    min_d = np.linalg.norm(X - X[start], axis=1).astype(np.float64)
    for _ in range(k - 1):
        nxt = int(min_d.argmax())
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1), out=min_d)
    return float(np.partition(min_d, -(z + 1))[-(z + 1)])

def _charikar_feasible(X, k, z, r):
    n = X.shape[0]
    uncovered = np.ones(n, dtype=bool)
    r2, r3_2 = r*r, (3*r)**2
    for _ in range(k):
        idx_u = np.flatnonzero(uncovered)
        if idx_u.size <= z:
            return True
        Xu, best_center, best_cover = X[idx_u], -1, -1
        for i in range(0, n, BATCH_SIZE):
            Xb  = X[i:i+BATCH_SIZE]
            d2  = cdist(Xu, Xb, metric="sqeuclidean")
            cnt = np.sum(d2 <= r2, axis=0)
            j   = int(np.argmax(cnt))
            if cnt[j] > best_cover:
                best_cover, best_center = int(cnt[j]), i+j
        if best_cover == 0:
            break
        d2f = cdist(X, X[best_center:best_center+1], metric="sqeuclidean").ravel()
        uncovered &= (d2f > r3_2)
    return int(uncovered.sum()) <= z

def charikar_radius(X, k, z):
    n = X.shape[0]
    sample_idx = np.linspace(0, n-1, min(n, MAX_RADII), dtype=int)
    radii = np.unique(cdist(X, X[sample_idx], metric="euclidean"))
    lo, hi, best = 0, len(radii)-1, None
    while lo <= hi:
        mid = (lo+hi)//2
        if _charikar_feasible(X, k, z, radii[mid]):
            best = radii[mid]; hi = mid-1
        else:
            lo = mid+1
    return float(best) if best is not None else float(radii[-1])

# ── per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(dataset_file, output_csv, ratios, time_coef):
    if not os.path.exists(dataset_file):
        print(f"\n[SKIP] {dataset_file} not found")
        return

    print(f"\n{'='*65}")
    print(f"  {dataset_file}  →  {output_csv}")

    with open(dataset_file) as f:
        X_full = np.asarray(ast.literal_eval(f.read()), dtype=np.float32)
    print(f"  {X_full.shape[0]:,} rows x {X_full.shape[1]} cols")

    sizes = [n for n in target_sizes if n <= len(X_full)]

    try:
        existing = pd.read_csv(output_csv)
        rows = existing.to_dict("records")
        done = set(zip(existing.N, existing.k, existing.outlier_percent))
        print(f"  Resuming: {len(rows)} rows already saved")
    except FileNotFoundError:
        rows, done = [], set()

    total = len(sizes) * len(k_values) * len(outlier_percents)
    idx   = 0

    for N in sizes:
        X = X_full[:N]
        for k in k_values:
            for op in outlier_percents:
                idx += 1
                if (N, k, op) in done:
                    print(f"  [{idx:3d}/{total}] N={N:6d} k={k:2d} op={op:.0%} — skip")
                    continue

                z = int(op * N)
                print(f"  [{idx:3d}/{total}] N={N:6d} k={k:2d} z={z:5d} ({op:.0%})", end="", flush=True)

                t0 = time.perf_counter()
                g_r = gonzalez_radius(X, k, z)
                g_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                c_r = charikar_radius(X, k, z)
                c_t = time.perf_counter() - t0

                r_gonz, _ = get_ratio(ratios, k, op)
                lp_r  = round(g_r * r_gonz, 8)
                lp_t  = estimate_lp_time(time_coef, N, k, op)

                print(f"  gonz={g_r:.5f}  char={c_r:.5f}  LP≈{lp_r:.5f}", flush=True)

                rows.append({
                    "N":               N,
                    "k":               k,
                    "outlier_percent": op,
                    "z":               z,
                    "charikar_radius": round(c_r, 7),
                    "charikar_time":   round(c_t, 4),
                    "gonzalez_radius": round(g_r, 7),
                    "gonzalez_time":   round(g_t, 6),
                    "LP_Radius":       lp_r,
                    "LP_Time":         lp_t,
                })
                pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"\n  Saved {len(rows)} rows → {output_csv}")

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ratios    = fit_ratios(SOURCE_LP_CSV)
    time_coef = fit_time_model(SOURCE_LP_CSV)

    for ds_file, out_csv in DATASETS:
        run_dataset(ds_file, out_csv, ratios, time_coef)

    print("\nDone. Estimated LP CSVs ready.")
    print("NOTE: Label these as 'estimated LP bounds' in your presentation.")
