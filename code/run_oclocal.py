"""OC-Local only — runs fast, prints results immediately."""
import numpy as np, ast, time, csv, os

TARGET_SIZES  = [500, 1000, 5000, 10000, 15000, 20000]
K_VALUES      = [10, 20, 50]
OUTLIER_PCTS  = [0.10, 0.20]
O             = 20

DATASETS = [
    ("diabetes_dataset.py",  "oc_results_diabetes.csv"),
    ("covertype_dataset.py", "oc_results_covertype.csv"),
]

def centroid_start(X):
    return int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())

def robust_r(d, z):
    return float(np.partition(d, -(z+1))[-(z+1)])

def gonzalez(X, m):
    s = centroid_start(X)
    centers = [s]
    md = np.linalg.norm(X - X[s], axis=1).astype(np.float64)
    for _ in range(m - 1):
        nxt = int(md.argmax())
        centers.append(nxt)
        np.minimum(md, np.linalg.norm(X - X[nxt], axis=1), out=md)
    return centers, md

def pool_dists(X, pool):
    return np.column_stack([np.linalg.norm(X - X[c], axis=1) for c in pool])

def forward_greedy(D, k, z):
    sel = [0]; md = D[:, 0].copy(); rem = list(range(1, D.shape[1]))
    for _ in range(k - 1):
        best_r, best_q = np.inf, None
        for qi in rem:
            r = robust_r(np.minimum(md, D[:, qi]), z)
            if r < best_r: best_r, best_q = r, qi
        sel.append(best_q); np.minimum(md, D[:, best_q], out=md); rem.remove(best_q)
    return sel

def local_search(D, sel, k, z, max_iter=200):
    sel = list(sel); ss = set(sel)
    best_r = robust_r(D[:, sel].min(axis=1), z)
    for _ in range(max_iter):
        improved = False
        for i in range(k):
            ci = sel[i]; others = [sel[j] for j in range(k) if j != i]
            base = D[:, others].min(axis=1) if others else np.full(len(D), np.inf)
            for qi in range(D.shape[1]):
                if qi in ss: continue
                r = robust_r(np.minimum(base, D[:, qi]), z)
                if r < best_r - 1e-10:
                    ss.discard(ci); ss.add(qi); sel[i] = qi
                    best_r = r; improved = True; break
            if improved: break
        if not improved: break
    return sel, best_r

def run_oc(X, k, z):
    t0 = time.perf_counter()
    pool, _ = gonzalez(X, k + O)
    D       = pool_dists(X, pool)
    sel     = forward_greedy(D, k, z)
    _, r    = local_search(D, sel, k, z)
    return r, time.perf_counter() - t0

os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')

for ds_file, out_csv in DATASETS:
    print(f"\n{'='*60}")
    print(f"  {ds_file}")
    print(f"{'='*60}")
    X_full = np.array(ast.literal_eval(open(ds_file).read()), dtype=np.float32)
    print(f"  Loaded: {X_full.shape}")

    sizes = [n for n in TARGET_SIZES if n <= len(X_full)]
    total = len(sizes) * len(K_VALUES) * len(OUTLIER_PCTS)
    done  = 0

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N','k','outlier_pct','z','oc_radius','oc_time_s'])

    rows = []
    for N in sizes:
        X = X_full[:N]
        for k in K_VALUES:
            for op in OUTLIER_PCTS:
                done += 1
                z = int(op * N)
                r, t = run_oc(X, k, z)
                rows.append([N, k, op, z, round(r,8), round(t,6)])
                print(f"  [{done:3d}/{total}] N={N:6d} k={k:2d} z={z:5d} ({int(op*100)}%)  "
                      f"OC-Local r={r:.5f}  t={t:.3f}s", flush=True)
                with open(out_csv, 'a', newline='') as f:
                    csv.writer(f).writerow(rows[-1])

    print(f"\n  Saved {len(rows)} rows → {out_csv}")
    import pandas as pd
    df = pd.DataFrame(rows, columns=['N','k','op','z','r','t'])
    print(f"  Mean radius: {df.r.mean():.4f}")
    print(f"  Median time: {df.t.median():.3f}s")
    print(f"  Max time:    {df.t.max():.3f}s")

print("\nOC-Local complete.")
