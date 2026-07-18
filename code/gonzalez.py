import numpy as np


def rkc_gonzalez(X, k, z):
    """
    RKC-Gonzalez baseline: at each of k steps pick the (z+1)-th farthest point
    from the current centers rather than the absolute farthest.
    Starts from the point nearest the empirical centroid.
    Returns the robust radius r_z after k centers are chosen.
    """
    c0 = int(np.linalg.norm(X - X.mean(axis=0), axis=1).argmin())
    min_d = np.linalg.norm(X - X[c0], axis=1).astype(np.float32)
    for _ in range(k - 1):
        nxt = int(np.argpartition(min_d, -(z + 1))[-(z + 1)])
        np.minimum(min_d, np.linalg.norm(X - X[nxt], axis=1).astype(np.float32), out=min_d)
    return float(np.partition(min_d, -(z + 1))[-(z + 1)])
