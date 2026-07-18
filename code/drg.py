import numpy as np
from scipy.spatial.distance import cdist


def robust_radius(X, centers, z):
    """(z+1)-th largest distance from any point to its nearest center."""
    dists = cdist(X, X[centers]).min(axis=1)
    return float(np.partition(dists, -(z + 1))[-(z + 1)])


def gonzalez_pool(X, m):
    """Gonzalez farthest-point for m steps, starting from the point nearest the empirical centroid."""
    c0 = int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())
    centers = [c0]
    min_d = np.linalg.norm(X - X[c0], axis=1).astype(np.float32)
    for _ in range(m - 1):
        nxt = int(min_d.argmax())
        centers.append(nxt)
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1).astype(np.float32), out=min_d)
    return centers


def _rz(d, z):
    """(z+1)-th largest value in array d via O(n) partition."""
    return float(np.partition(d, -(z + 1))[-(z + 1)])


def forward_greedy(D, k, z):
    """
    Greedy selection of k pool indices from precomputed n x pool_size distance matrix D.
    Starts from pool index 0. Returns pool-local indices.
    """
    sel = [0]
    min_d = D[:, 0].copy()
    rem = list(range(1, D.shape[1]))
    for _ in range(k - 1):
        best_r, best_qi = np.inf, None
        for qi in rem:
            r = _rz(np.minimum(min_d, D[:, qi]), z)
            if r < best_r:
                best_r, best_qi = r, qi
        sel.append(best_qi)
        np.minimum(min_d, D[:, best_qi], out=min_d)
        rem.remove(best_qi)
    return sel


def local_search(D, sel, k, z, eps=1e-10):
    """
    1-swap first-improvement local search over the pool.
    Iterates over all (i, qi) pairs; accepts the first swap that lowers r_z by more
    than eps and restarts. Terminates when no improving swap exists.
    Returns (pool-local indices, best_r).
    """
    sel = list(sel)
    best_r = _rz(D[:, sel].min(axis=1), z)
    pool_sz = D.shape[1]

    while True:
        improved = False
        for i in range(k):
            others = [sel[j] for j in range(k) if j != i]
            base_d = D[:, others].min(axis=1) if others else np.full(D.shape[0], np.inf)
            sel_set = set(sel)
            for qi in range(pool_sz):
                if qi in sel_set:
                    continue
                r = _rz(np.minimum(base_d, D[:, qi]), z)
                if r < best_r - eps:
                    sel[i] = qi
                    best_r = r
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return sel, best_r


def oc_greedy(X, k, z):
    """OC-Greedy: gonzalez_pool(k+z) then forward_greedy. Returns (global center indices, radius)."""
    pool = gonzalez_pool(X, k + z)
    D = cdist(X, X[pool])
    sel = forward_greedy(D, k, z)
    centers = [pool[i] for i in sel]
    return centers, robust_radius(X, centers, z)


def oc_local(X, k, z):
    """OC-Local: pool + forward_greedy + local_search. Returns (global center indices, radius)."""
    pool = gonzalez_pool(X, k + z)
    D = cdist(X, X[pool])
    sel = forward_greedy(D, k, z)
    sel, r = local_search(D, sel, k, z)
    centers = [pool[i] for i in sel]
    return centers, r
