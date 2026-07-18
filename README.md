# OverclusterDRG: Robust k-Center Clustering with Outliers

Code for the paper **"OverclusterDRG: A Practical Algorithm for Robust k-Center Clustering with Outliers"** (SIAM Proceedings, 2026).

## Key Findings

- OC-Local achieves mean LP ratios of **1.147 (Adult), 1.155 (Diabetes), 1.171 (Covertype)** — the lowest of any practical algorithm tested.
- OC-Local beats Charikar in 33/36 Adult configurations at a median **7× speedup**, and beats Ding et al. in 35/36 configurations at **8× speedup**.
- A Gonzalez pool of size **k + z** is the theoretically exact threshold: one step fewer yields unbounded ratio; one step more gives no quality improvement.
- The 1-swap local search finds the exact pool optimum in **82% of instances** (N ≤ 500); worst-case gap under 3%.

## Reproducing Results

**Requirements:** Python 3.9+, NumPy, SciPy, Pandas, Matplotlib, Gurobi (for LP bounds only).

```bash
pip install numpy scipy pandas matplotlib gurobipy

# All algorithms on Adult dataset (RKC-Gonzalez, Charikar, OC-Greedy, OC-Local, LP)
python code/run_all_algorithms.py adult

# Ding et al. on all three datasets
python code/run_ding.py

# OC-Local on Diabetes and Covertype
python code/run_oclocal.py

# Pool-size ablation sweep
python code/explore2.py

# Local search vs exhaustive comparison (small N)
python code/exhaustive.py

# Generate paper figures (reads results/, writes plots/)
python code/generate_paper_figures.py
python code/experiments/run_plots.py
```

Pre-computed results are in `results/` and all plots are in `plots/`. The figure scripts can be run directly without re-running experiments.

## Code

| File | Purpose |
|------|---------|
| `code/drg.py` | `oc_local`, `oc_greedy`, `gonzalez_pool`, `forward_greedy`, `local_search` |
| `code/gonzalez.py` | RKC-Gonzalez baseline |
| `code/rkc.py` | Charikar 3-approximation |
| `code/LP_relaxed.py` | LP lower bound via Gurobi (binary search over distances) |
| `code/run_all_algorithms.py` | Benchmark runner for all algorithms |
| `code/run_ding.py` | Ding et al. randomized greedy |
| `code/run_oclocal.py` | OC-Local on secondary datasets |
| `code/exhaustive.py` | Exhaustive phase-2 search for quality verification |
| `code/explore2.py` | Pool-size ablation (sweep O from 1 to 3z) |
| `code/experiments/run_plots.py` | Ablation and phase-transition figures |
| `code/generate_paper_figures.py` | Main paper figures (fig1, fig3, fig5, fig_ratio_both) |

## Algorithms

**OC-Greedy / OC-Local** (`drg.py`): Build a Gonzalez pool of k+z points from the centroid. Forward greedy selects k pool points minimising the robust radius r_z at each step. OC-Local then applies 1-swap first-improvement local search until no improving swap exists.

**Charikar 3-approximation** (`rkc.py`): Binary search over pairwise distances; feasibility oracle with coverage radius r and deletion radius 3r.

**RKC-Gonzalez** (`gonzalez.py`): Gonzalez farthest-point where each step picks the (z+1)-th farthest instead of the absolute farthest.

**Ding et al.** (`run_ding.py`): Randomized greedy picking uniformly from the z+1 farthest at each step. T = ⌈(z+1)·ln(100)⌉ trials.

**LP lower bound** (`LP_relaxed.py`): Gurobi feasibility LP with outlier coverage constraint `sum x_ij ≥ N − z`. Binary search over all pairwise distances gives R_LP ≤ OPT.

## Datasets

| Dataset | N (full) | Features | File |
|---------|---------|----------|------|
| UCI Adult | 48,842 | 6 continuous | `datasets/adult_final_dataset.py` |
| Diabetes Health Indicators | 253,680 | 21 | `datasets/diabetes_dataset.py` |
| Covertype | 581,012 | 10 continuous | `datasets/covertype_dataset.py` |

Experiments subsample to N ∈ {500, 1000, 2000, 5000, 10000, 20000}, k ∈ {10, 20, 50}, z/N ∈ {10%, 20%}. All features are z-score standardised; distances are Euclidean.
