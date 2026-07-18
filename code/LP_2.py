import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.spatial import cKDTree
import ast
import time
import sys
import csv
import os
from multiprocessing import Pool, cpu_count, Lock

# =====================================================
# 1. LOAD DATA
# =====================================================
DATA_FILE_NAME = "adult_final_dataset.py"

try:
    with open(DATA_FILE_NAME, "r") as f:
        ALL_POINTS = ast.literal_eval(f.read())
except Exception as e:
    print("Failed to load data:", e)
    sys.exit(1)

TOTAL_POINTS = len(ALL_POINTS)

# =====================================================
# 2. EXPERIMENT CONFIG
# =====================================================
K_VALUES = [10, 20, 50]
OUTLIER_PERCENTS = [0.1, 0.2]
DATASET_SIZES = [500]

RESULTS_FILE = "kcenter_lp_results_outliers_corrected_500.csv"
CSV_HEADER = ["N", "k", "outlier_percent", "z", "radius", "time"]

# =====================================================
# 3. CSV INIT (CRASH SAFE)
# =====================================================
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

# Windows-safe file lock
FILE_LOCK = Lock()

def append_row(row):
    with FILE_LOCK:
        with open(RESULTS_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)

# =====================================================
# 4. LP SOLVER — CORRECT WITH OUTLIERS
# =====================================================
def solve_kcenter_lp(points, k, z, threads):
    n = points.shape[0]
    tree = cKDTree(points)

    candidate_radii = np.unique(
        tree.query(points, k=min(n, k + z + 1))[0].flatten()
    )
    candidate_radii.sort()

    def feasible(radius):
        neighbors = tree.query_ball_point(points, radius)

        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.setParam("Method", 2)
        m.setParam("Crossover", 0)
        m.setParam("Presolve", 2)
        m.setParam("Threads", threads)

        # variables
        y = m.addVars(n, lb=0, ub=1, name="y")
        c = m.addVars(n, lb=0, ub=1, name="covered")
        x = {}

        for j in range(n):
            for i in neighbors[j]:
                x[i, j] = m.addVar(lb=0, ub=1)

        m.update()

        # coverage with outliers
        for j in range(n):
            m.addConstr(
                gp.quicksum(x[i, j] for i in neighbors[j]) >= c[j]
            )

        # allow z uncovered points
        m.addConstr(c.sum() >= n - z)

        # linking
        for (i, j) in x:
            m.addConstr(x[i, j] <= y[i])

        # center budget
        m.addConstr(y.sum() <= k)

        m.setObjective(0, GRB.MINIMIZE)
        m.optimize()

        status = m.Status
        m.dispose()
        return status == GRB.OPTIMAL

    # binary search
    left, right = 0, len(candidate_radii) - 1
    best = None

    while left <= right:
        mid = (left + right) // 2
        r = candidate_radii[mid]

        print(
            f"    [RADIUS TEST] N={n}, k={k}, z={z} | r={r:.6f}",
            flush=True
        )

        if feasible(r):
            best = r
            right = mid - 1
        else:
            left = mid + 1

    return best

# =====================================================
# 5. WORKER FUNCTION
# =====================================================
def run_instance(args):
    N, k, outlier_percent, threads = args
    z = int(outlier_percent * N)

    print(
        f"[START] N={N}, k={k}, outliers={outlier_percent} (z={z})",
        flush=True
    )

    points = np.asarray(ALL_POINTS[:N], dtype=np.float64)

    start = time.time()
    radius = solve_kcenter_lp(points, k, z, threads)
    elapsed = time.time() - start

    print(
        f"[DONE ] N={N}, k={k}, z={z} | "
        f"radius={radius:.6f}, time={elapsed:.2f}s",
        flush=True
    )

    return [N, k, outlier_percent, z, radius, round(elapsed, 4)]

# =====================================================
# 6. PARALLEL DRIVER
# =====================================================
if __name__ == "__main__":

    tasks = sorted(
        [(N, k, o)
         for N in DATASET_SIZES if N <= TOTAL_POINTS
         for k in K_VALUES
         for o in OUTLIER_PERCENTS],
        key=lambda x: x[0]
    )

    TOTAL_CORES = cpu_count()

    # Stable configuration for Windows + Gurobi
    NUM_PROCESSES = 4
    THREADS_PER_GUROBI = 2

    print("=" * 70)
    print("PARALLEL K-CENTER LP WITH OUTLIERS (WINDOWS SAFE)")
    print(f"Total cores          : {TOTAL_CORES}")
    print(f"Python processes     : {NUM_PROCESSES}")
    print(f"Gurobi threads/proc  : {THREADS_PER_GUROBI}")
    print("=" * 70)

    job_args = [(N, k, o, THREADS_PER_GUROBI) for (N, k, o) in tasks]

    with Pool(processes=NUM_PROCESSES) as pool:
        for result in pool.imap_unordered(run_instance, job_args):
            append_row(result)
            print("Saved:", result, flush=True)

    print("\nALL EXPERIMENTS COMPLETED.")