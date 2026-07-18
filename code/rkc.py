import numpy as np
from scipy.spatial.distance import cdist


def charikar_feasible(X, k, z, r, batch=1024):
    """
    Greedy feasibility oracle for the Charikar 3-approximation.
    Centers are chosen to maximally cover uncovered points within radius r;
    all points within 3r of the chosen center are then removed.
    Returns True if at most z points remain uncovered after k centers are placed.
    """
    n = X.shape[0]
    uncovered = np.ones(n, dtype=bool)
    r2 = r * r
    r3_2 = (3.0 * r) ** 2

    for _ in range(k):
        idx_unc = np.flatnonzero(uncovered)
        if idx_unc.size <= z:
            return True

        Xu = X[idx_unc]
        best_center, best_cover = -1, -1

        for i in range(0, n, batch):
            d2 = cdist(Xu, X[i:i + batch], metric='sqeuclidean')
            counts = (d2 <= r2).sum(axis=0)
            j = int(np.argmax(counts))
            if counts[j] > best_cover:
                best_cover = int(counts[j])
                best_center = i + j

        if best_cover == 0:
            break

        d2_full = cdist(X, X[best_center:best_center + 1], metric='sqeuclidean').ravel()
        uncovered &= d2_full > r3_2

    return int(uncovered.sum()) <= z


def charikar_kcenter_outliers(X, k, z, max_radii=200):
    """
    Binary search over sampled pairwise radii to find the smallest r for which
    charikar_feasible returns True. Returns the certified 3-approximation radius r*.
    """
    n = X.shape[0]
    sample_idx = np.linspace(0, n - 1, min(n, max_radii), dtype=int)
    radii = np.unique(cdist(X, X[sample_idx], metric='euclidean'))

    lo, hi = 0, len(radii) - 1
    best_r = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if charikar_feasible(X, k, z, radii[mid]):
            best_r = radii[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return best_r
