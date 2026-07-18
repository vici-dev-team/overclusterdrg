import numpy as np
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB


def _lp_feasible(X, k, z, r, D, tol):
    """
    Tests LP feasibility for a fixed radius r.

    Variables:  x_ij in [0,1] for i in N(j,r);  y_i in [0,1]
    Constraints:
      sum_{i in N(j,r)} x_ij <= 1   for all j      (each point fractionally covered at most once)
      x_ij <= y_i                    for all (i,j)  (can only assign to an open center)
      sum_i y_i <= k                               (budget)
      sum_{j,i} x_ij >= n - z                      (cover at least n-z points)

    Only creates x_ij variables for pairs (i,j) with D[i,j] <= r (sparse).
    Returns True if Gurobi finds the LP optimal (feasible), False otherwise.
    """
    n = len(X)
    try:
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.setParam('FeasibilityTol', tol)

        y = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

        # neighbors[j] = list of i with D[i,j] <= r
        neighbors = [list(np.where(D[:, j] <= r)[0]) for j in range(n)]

        x = {}
        for j in range(n):
            for i in neighbors[j]:
                x[i, j] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

        m.update()

        for j in range(n):
            if neighbors[j]:
                m.addConstr(gp.quicksum(x[i, j] for i in neighbors[j]) <= 1)

        for (i, j), xvar in x.items():
            m.addConstr(xvar <= y[i])

        m.addConstr(y.sum() <= k)
        m.addConstr(gp.quicksum(x.values()) >= n - z)

        m.setObjective(0, GRB.MINIMIZE)
        m.optimize()

        feasible = m.Status == GRB.OPTIMAL
        m.dispose()
        return feasible
    except gp.GurobiError:
        return False


def lp_lower_bound(X, k, z, tol=1e-6):
    """
    Returns the LP lower bound R_LP <= OPT via binary search over all pairwise distances.
    For each candidate radius r, tests LP feasibility with _lp_feasible.
    """
    D = cdist(X, X)
    radii = np.unique(D)

    lo, hi = 0, len(radii) - 1
    best_r = float(radii[-1])
    while lo <= hi:
        mid = (lo + hi) // 2
        if _lp_feasible(X, k, z, radii[mid], D, tol):
            best_r = float(radii[mid])
            hi = mid - 1
        else:
            lo = mid + 1
    return best_r
