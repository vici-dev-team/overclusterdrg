"""
Compute exact LP lower bounds for small N on Diabetes and Covertype,
run all algorithms on the same subsamples, fit a log-log model per
(k, outlier_percent) group, extrapolate LP to large N, and rebuild
the multidata result files with correct LP values.

Run from repo root:  python code/run_lp_fresh.py

Resumable: already-completed configs are loaded from lp_exact_small_n.csv
and skipped. Kill and restart safely at any time.
"""

import os, sys, ast, time, math, itertools, fcntl
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE     = os.path.join(REPO, 'code')
DATASETS = os.path.join(REPO, 'datasets')
RESULTS  = os.path.join(REPO, 'results')

N_EXACT   = [100, 200, 400, 500, 600, 800, 900, 1000]
N_LARGE   = [5000, 10000, 15000, 20000]
K_VALS    = [10, 20, 50]
OPS       = [0.10, 0.20]
DS_NAMES  = ['diabetes', 'covertype']
N_WORKERS = max(1, os.cpu_count() - 2)
LP_TIMEOUT_S = 360   # skip LP if a single binary-search solve exceeds this

EXACT_PATH = os.path.join(RESULTS, 'lp_exact_small_n.csv')


def load_dataset(name):
    path = os.path.join(DATASETS, f'{name}_dataset.py')
    X = np.array(ast.literal_eval(open(path).read()), dtype=np.float32)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - X.mean(axis=0)) / std


def _lp_feasible(D, k, z, r, tol=1e-6):
    import gurobipy as gp
    from gurobipy import GRB
    n = D.shape[0]
    try:
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.setParam('Threads', 2)
        m.setParam('FeasibilityTol', tol)
        m.setParam('TimeLimit', LP_TIMEOUT_S)

        y    = m.addVars(n, lb=0.0, ub=1.0)
        nbrs = [list(np.where(D[:, j] <= r)[0]) for j in range(n)]
        x    = {}
        for j in range(n):
            for i in nbrs[j]:
                x[i, j] = m.addVar(lb=0.0, ub=1.0)
        m.update()

        for j in range(n):
            if nbrs[j]:
                m.addConstr(gp.quicksum(x[i, j] for i in nbrs[j]) <= 1)
        for (i, j), xv in x.items():
            m.addConstr(xv <= y[i])
        m.addConstr(gp.quicksum(y.values()) <= k)
        if x:
            m.addConstr(gp.quicksum(x.values()) >= n - z)
        else:
            m.dispose()
            return False, True

        m.setObjective(0, GRB.MINIMIZE)
        m.optimize()
        timed_out = m.Status == GRB.TIME_LIMIT
        ok = m.Status == GRB.OPTIMAL
        m.dispose()
        return ok, timed_out
    except Exception:
        return False, False


def _lp_lower_bound(X, k, z):
    D      = cdist(X, X, 'euclidean').astype(np.float32)
    radii  = np.unique(D)
    lo, hi = 0, len(radii) - 1
    best   = float(radii[-1])
    while lo <= hi:
        mid = (lo + hi) // 2
        r   = float(radii[mid])
        ok, timed_out = _lp_feasible(D, k, z, r)
        if timed_out:
            return None   # signal timeout to caller
        if ok:
            best = r
            hi   = mid - 1
        else:
            lo = mid + 1
    return best


def _ding(X, k, z, seed=42):
    rng    = np.random.default_rng(seed)
    T      = math.ceil((z + 1) * math.log(100))
    best_r = np.inf
    for _ in range(T):
        c  = int(rng.integers(len(X)))
        cs = [c]
        md = np.linalg.norm(X - X[c], axis=1).astype(np.float64)
        for _ in range(k - 1):
            top = np.argpartition(md, -(z + 1))[-(z + 1):]
            c   = int(rng.choice(top))
            cs.append(c)
            np.minimum(md, np.linalg.norm(X - X[c], axis=1), out=md)
        d = cdist(X, X[cs]).min(axis=1)
        r = float(np.partition(d, -(z + 1))[-(z + 1)])
        if r < best_r:
            best_r = r
    return best_r


def _append_row(rec):
    """Append one result row to the CSV immediately (file-locked for safety)."""
    df = pd.DataFrame([rec])
    with open(EXACT_PATH, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        df.to_csv(f, header=f.tell() == 0, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)


def run_config(args):
    ds_name, N, k, op, X_sub = args
    sys.path.insert(0, CODE)
    from drg      import oc_local, oc_greedy
    from gonzalez import rkc_gonzalez

    z   = int(N * op)
    tag = f"{ds_name} N={N} k={k} z={z}"
    print(f"  START {tag}", flush=True)

    rec = {'dataset': ds_name, 'N': N, 'k': k, 'outlier_percent': op, 'z': z}

    t0      = time.time()
    lp_r    = _lp_lower_bound(X_sub, k, z)
    lp_time = round(time.time() - t0, 2)

    if lp_r is None:
        print(f"  SKIP  {tag}  LP timed out after {lp_time:.0f}s", flush=True)
        return None

    rec['LP_Radius'] = lp_r
    rec['LP_Time']   = lp_time

    t0 = time.time(); _, r = oc_local(X_sub, k, z)
    rec['OC_Local_radius'] = r; rec['OC_Local_time'] = round(time.time()-t0, 3)

    t0 = time.time(); _, r = oc_greedy(X_sub, k, z)
    rec['OC_Forward_radius'] = r; rec['OC_Forward_time'] = round(time.time()-t0, 3)

    t0 = time.time(); r = _ding(X_sub, k, z)
    rec['DingRKC_radius'] = r; rec['DingRKC_time'] = round(time.time()-t0, 3)

    t0 = time.time(); r = rkc_gonzalez(X_sub, k, z)
    rec['gonzalez_radius'] = r; rec['gonzalez_time'] = round(time.time()-t0, 3)

    _append_row(rec)   # persist immediately
    print(f"  DONE  {tag}  LP={lp_r:.4f}  OC={rec['OC_Local_radius']:.4f}"
          f"  Di={rec['DingRKC_radius']:.4f}  ({lp_time:.1f}s LP)", flush=True)
    return rec


def build_outputs(df_exact):
    """Fit log-log model and rebuild multidata files."""
    print("\nFitting extrapolation model (log LP ~ log N)...")
    models = {}
    for (ds, k, op), grp in df_exact.groupby(['dataset', 'k', 'outlier_percent']):
        if len(grp) < 2:
            continue
        log_N  = np.log(grp['N'].values.astype(float))
        log_LP = np.log(grp['LP_Radius'].values.astype(float))
        b, a   = np.polyfit(log_N, log_LP, 1)
        models[(ds, k, op)] = (a, b)
        r2 = np.corrcoef(log_N, log_LP)[0, 1] ** 2
        print(f"  {ds} k={k} op={op:.0%}:  LP ~ exp({a:.3f})·N^{b:.3f}  R²={r2:.4f}")

    def predict(ds, N, k, op):
        key = (ds, k, op)
        if key not in models:
            return np.nan
        a, b = models[key]
        return float(np.exp(a + b * np.log(N)))

    for ds_name in DS_NAMES:
        orig = pd.read_csv(os.path.join(RESULTS, f'multidata_{ds_name}_fixed.csv'))

        # Large-N rows: keep original algorithm results, replace LP with prediction
        large = orig[orig['N'].isin(N_LARGE)].copy()
        large['LP_Radius'] = large.apply(
            lambda r: predict(ds_name, r['N'], r['k'], r['outlier_percent']), axis=1)
        large.drop(columns=['charikar_radius', 'charikar_time'], errors='ignore', inplace=True)

        # Small-N rows: use fresh results for N=500 and N=1000 if available
        fresh = df_exact[(df_exact['dataset'] == ds_name) &
                         (df_exact['N'].isin([500, 1000]))].copy()

        keep_cols = ['N', 'k', 'outlier_percent', 'z', 'LP_Radius', 'LP_Time',
                     'OC_Forward_radius', 'OC_Forward_time',
                     'OC_Local_radius',   'OC_Local_time',
                     'DingRKC_radius',    'DingRKC_time',
                     'gonzalez_radius',   'gonzalez_time']

        large_out = large[[c for c in keep_cols if c in large.columns]]
        fresh_out = fresh[[c for c in keep_cols if c in fresh.columns]] if len(fresh) else pd.DataFrame()

        combined = pd.concat([fresh_out, large_out], ignore_index=True)
        combined.sort_values(['N', 'k', 'outlier_percent'], inplace=True)

        out = os.path.join(RESULTS, f'multidata_{ds_name}_fixed.csv')
        combined.to_csv(out, index=False)
        print(f"\nRebuilt {out}  ({len(combined)} rows)")

        for acol in ['OC_Local_radius', 'DingRKC_radius', 'gonzalez_radius']:
            if acol not in combined.columns:
                continue
            sub = combined.dropna(subset=[acol, 'LP_Radius'])
            bad = sub[sub[acol] < sub['LP_Radius'] - 1e-4]
            status = f"*** {len(bad)} violations ***" if len(bad) else "OK"
            print(f"  {acol}: {status}")


if __name__ == '__main__':
    t_wall = time.time()

    # Load any already-completed rows to skip them
    done_keys = set()
    if os.path.exists(EXACT_PATH):
        prev = pd.read_csv(EXACT_PATH)
        for _, r in prev.iterrows():
            done_keys.add((r['dataset'], int(r['N']), int(r['k']), float(r['outlier_percent'])))
        print(f"Resuming: {len(done_keys)} configs already done, skipping them.")
    else:
        # Write header
        pd.DataFrame(columns=['dataset','N','k','outlier_percent','z',
                               'LP_Radius','LP_Time',
                               'OC_Local_radius','OC_Local_time',
                               'OC_Forward_radius','OC_Forward_time',
                               'DingRKC_radius','DingRKC_time',
                               'gonzalez_radius','gonzalez_time']).to_csv(EXACT_PATH, index=False)

    print("Loading datasets...")
    X_full = {name: load_dataset(name) for name in DS_NAMES}
    print({name: X_full[name].shape for name in DS_NAMES})

    jobs = []
    for N, ds_name, k, op in itertools.product(N_EXACT, DS_NAMES, K_VALS, OPS):
        z = int(N * op)
        if z < 1:
            continue
        if (ds_name, N, k, op) in done_keys:
            continue
        seed = abs(hash((ds_name, N))) % (2**31)
        rng  = np.random.default_rng(seed)
        idx  = rng.choice(len(X_full[ds_name]), size=N, replace=False)
        jobs.append((ds_name, N, k, op, X_full[ds_name][idx]))

    print(f"\nRunning {len(jobs)} configs with {N_WORKERS} workers\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(run_config, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                ds, N, k, op, _ = futs[fut]
                print(f"  FAILED {ds} N={N} k={k} op={op}: {e}", flush=True)

    # Build outputs from everything saved so far
    df_exact = pd.read_csv(EXACT_PATH)
    print(f"\n{len(df_exact)} configs with exact LP saved.")
    build_outputs(df_exact)

    print(f"\nTotal elapsed: {(time.time()-t_wall)/60:.1f} min")
    print("Done. Run: python code/generate_paper_figures.py")
