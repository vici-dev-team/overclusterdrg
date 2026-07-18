import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.spatial import cKDTree
import ast
import time
import sys
import csv
import os
from multiprocessing import Pool, cpu_count

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
# 2. EXPERIMENT CONFIGURATION
# =====================================================
DATASET_SIZES = [500]
K_VALUES = [10, 20, 50]
OUTLIER_PERCENTS = [0.0, 0.1, 0.2]   # logged only (LP is vanilla k-center)

RESULTS_FILE = "kcenter_lp_results_500.csv"
CSV_HEADER = ["N", "k", "outlier_percent", "z", "radius", "time"]

# =====================================================
# 3. CSV INITIALIZATION (CRASH SAFE)
# =====================================================
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

def append_row(row):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

# =====================================================
# 4. LP SOLVER FOR ONE INSTANCE
# =====================================================
def solve_kcenter_lp(points, k, threads):
    """
    Correct LP relaxation of k-center for fixed points and k.
    Uses KDTree for distance queries and Gurobi for LP feasibility.
    """
    n = points.shape[0]
    tree = cKDTree(points)

    # Candidate radii (heuristic but standard)
    candidate_radii = np.unique(
        tree.query(points, k=min(n, k + 1))[0].flatten()
    )
    candidate_radii.sort()

    max_radius = candidate_radii[-1]
    neighbors_max = tree.query_ball_point(points, max_radius)

    # ---------------- BUILD MODEL ONCE ----------------
    m = gp.Model("kcenter_lp")
    m.setParam("OutputFlag", 0)
    m.setParam("Method", 2)
    m.setParam("Crossover", 0)
    m.setParam("Presolve", 2)
    m.setParam("Threads", threads)

    y = m.addVars(n, lb=0, ub=1, name="y")

    x = {}
    for j in range(n):
        for i in neighbors_max[j]:
            x[i, j] = m.addVar(lb=0, ub=1)

    m.update()

    # Coverage constraints
    for j in range(n):
        m.addConstr(
            gp.quicksum(x[i, j] for i in neighbors_max[j]) >= 1
        )

    # Linking constraints
    for (i, j) in x:
        m.addConstr(x[i, j] <= y[i])

    # Center budget
    m.addConstr(y.sum() <= k)

    m.setObjective(0, GRB.MINIMIZE)
    m.update()

    # ---------------- FEASIBILITY TEST ----------------
    def feasible(radius):
        neighbors = tree.query_ball_point(points, radius)

        # quick prune
        for j in range(n):
            if not neighbors[j]:
                return False

        for j in range(n):
            valid = set(neighbors[j])
            for i in neighbors_max[j]:
                x[i, j].ub = 1 if i in valid else 0

        m.optimize()
        return m.Status == GRB.OPTIMAL

    # ---------------- BINARY SEARCH ----------------
    left, right = 0, len(candidate_radii) - 1
    best = None

    while left <= right:
        mid = (left + right) // 2
        r = candidate_radii[mid]

        if feasible(r):
            best = r
            right = mid - 1
        else:
            left = mid + 1

    m.dispose()
    return best

# =====================================================
# 5. WORKER FUNCTION (ONE PARALLEL JOB)
# =====================================================
def run_instance(args):
    N, k, outlier_percent, threads = args

    z = int(outlier_percent * N)
    points = np.asarray(ALL_POINTS[:N], dtype=np.float64)

    start = time.time()
    radius = solve_kcenter_lp(points, k, threads)
    elapsed = time.time() - start

    return [N, k, outlier_percent, z, radius, round(elapsed, 4)]

# =====================================================
# 6. PARALLEL DRIVER
# =====================================================
if __name__ == "__main__":

    tasks = []
    for N in DATASET_SIZES:
        if N > TOTAL_POINTS:
            continue
        for k in K_VALUES:
            for outlier_percent in OUTLIER_PERCENTS:
                tasks.append((N, k, outlier_percent))

    TOTAL_CORES = cpu_count()
    NUM_PROCESSES = max(1, TOTAL_CORES // 2)
    THREADS_PER_GUROBI = max(1, TOTAL_CORES // NUM_PROCESSES)

    print("=" * 70)
    print("PARALLEL K-CENTER LP RELAXATION")
    print(f"Total cores          : {TOTAL_CORES}")
    print(f"Python processes     : {NUM_PROCESSES}")
    print(f"Gurobi threads/proc  : {THREADS_PER_GUROBI}")
    print("=" * 70)

    job_args = [(N, k, o, THREADS_PER_GUROBI) for (N, k, o) in tasks]

    with Pool(processes=NUM_PROCESSES) as pool:
        for result in pool.imap_unordered(run_instance, job_args):
            append_row(result)
            print("Saved:", result)

    print("\nALL EXPERIMENTS COMPLETED.")