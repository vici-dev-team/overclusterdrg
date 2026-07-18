# OverclusterDRG: Code and Experiments

This repository contains the implementation and experimental suite for the paper **"OverclusterDRG: A Deterministic, LP-Free Heuristic for Robust k-Center with Outliers"**. The tools provided allow complete replication of all results in the paper.

---

## 1. Summary of Key Empirical Findings

Experiments across three UCI datasets (Adult, Diabetes Indicators, Covertype) validate OverclusterDRG's practical utility.

1. **Best quality among practical algorithms.** OC-Local achieves mean LP ratios of **1.147** (Adult), **1.155** (Diabetes), and **1.171** (Covertype) — the lowest of any practical algorithm on all three datasets.

2. **Faster than competitors.** OC-Local beats Charikar in **33/36** Adult configurations at a median **7× speedup**, and beats Ding et al. in **35/36** at **8× speedup**. All comparisons are significant at p < 10⁻⁵ by both t-test and Wilcoxon signed-rank test.

3. **The 1-swap step always helps.** OC-Local improves over OC-Forward (same algorithm without local search) in every single configuration at a median additional cost of 0.10 seconds. It should always be applied.

4. **Exact phase transition at O = z.** The pool contains a 3·OPT solution whenever the overcount O ≥ z. One step fewer and the ratio is unbounded. This is not a tuning choice — it is a theoretically pinpointed threshold.

5. **Practical at large scale.** OC-Local runs in O(N^1.85) time. At N = 99,000 it is projected to finish in ~18 minutes; Ding et al. needs ~1 hour; Charikar requires several hours; the LP is out of reach entirely.

---

## 2. Guide to Reproducing Results

### Step 1: Setup

```bash
pip install numpy scipy pandas matplotlib seaborn gurobipy
```

Gurobi 12.0.3 with a valid licence is required for the LP lower bound. All other results run without it.

### Step 2: Run Experiments

```bash
# Full comparison: OC-Local, OC-Forward, Charikar, Ding, RKC-Gonzalez, LP
python code/run_all_algorithms.py

# OC-Local only (fast)
python code/run_oclocal.py

# Scalability sweep (large N)
python code/experiments/run_scalability.py

# Pool-size ablation (O sweep)
python code/explore2.py
```

Results are written to `results/` as CSV files.

### Step 3: Generate Plots

```bash
python code/plots_summary.py   # main paper figures
python code/experiments/run_plots.py
```

Plots are written to `plots/`.

---

## 3. Code Structure

```
code/
  drg.py                  — OverclusterDRG: Phase 1 (Gonzalez pool),
                            Phase 2a (OC-Forward), Phase 2b (OC-Local)
  gonzalez.py             — Standard Gonzalez k-center
  rkc.py                  — Charikar 3-approximation (RKC)
  LP_relaxed.py           — LP lower bound solver (Gurobi)
  run_all_algorithms.py   — Main experiment runner (all algorithms)
  run_oclocal.py          — OC-Local standalone runner
  run_ding.py             — Ding et al. randomised baseline
  run_gonzalez_charikar.py— Gonzalez and Charikar runners
  analysis.py             — Post-experiment analysis with raw results
  prepare_datasets.py     — Dataset preprocessing utilities
  plots_summary.py        — Generate all main paper figures
  exhaustive.py           — Exhaustive pool search (validation)
  explore.py / explore2.py— Pool-size ablation experiments
  experiments/
    run_scalability.py    — Large-N scalability
    run_htru2.py          — HTRU2 dataset experiments
    run_plots.py          — Additional plots

datasets/
  adult_final_dataset.py  — UCI Adult loader (6 numeric features, z-scored)
  diabetes_dataset.py     — Diabetes Indicators loader (8 features)
  covertype_dataset.py    — Covertype loader (10 features, 30k subsample)
  htru2_dataset.py        — HTRU2 pulsar dataset loader

plots/
  fig3_quality_speed.png  — Quality–runtime tradeoff (Figure 3)
  fig1_ratio_vs_N.png     — LP ratio vs N (Figure 1)
  fig_O_ablation.png      — Pool size ablation
  fig_phase_transition.png— Phase transition at O=z
  ...

results/
  final_comparison.csv       — Per-configuration results (all algorithms)
  LP_Results.csv             — Exact LP lower bounds (Adult)
  full_results_diabetes.csv  — Diabetes full results
  full_results_covertype.csv — Covertype full results
  paper_findings/            — Tables as reported in the paper
    table1_lp_ratios_by_dataset.csv
    table3_adult_all36_CORRECTED.csv
    table7_wilcoxon_CORRECTED.csv
    ...
```

---

## 4. Algorithms

| Algorithm | Type | Guarantee | Mean LP ratio (Adult) | Runtime at N=20k |
|---|---|---|---|---|
| OC-Local | Deterministic | Pool contains 3·OPT | **1.147** | 57 s |
| OC-Forward | Deterministic | Pool contains 3·OPT | 1.158 | 52 s |
| Charikar et al. | Deterministic | 3-approximation | 1.173 | 564 s |
| Ding et al. | Randomised (99%) | 2-approximation | 1.235 | 431 s |
| RKC-Gonzalez | Heuristic | None | 1.280 | 0.02 s |

---

## 5. Datasets

Experiments use three publicly available datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).

1. **UCI Adult (Census Income)** — 32,561 rows, 6 continuous numeric features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week). Grid: N ∈ {500, 1000, 2000, 5000, 10000, 20000}, k ∈ {10, 20, 50}, z/N ∈ {10%, 20%} → 36 configurations with exact LP bounds.

2. **Diabetes Health Indicators** — 99,492 rows, 8 features. Same k and z/N as Adult; N replaces 2,000 with 15,000 → 36 configurations.

3. **Covertype** — 581,012 rows subsampled to 30,000 (seed 42), 10 features. N ∈ {500, 1000, 5000, 10000, 15000} → 30 configurations.

All features are z-score standardised. Distances are Euclidean.

---

## 6. Theoretical Guarantees

**Phase transition theorem (exact):** The Gonzalez pool with O = z extra steps is guaranteed to contain a 3·OPT solution. One step fewer (O = z−1) and the approximation ratio is unbounded for any k ≥ 2. The threshold O = z is not a heuristic choice — it is proven exact by an explicit adversarial construction.

**References:**
- Ding, Yu, Wang. *Greedy Strategy Works for k-Center Clustering with Outliers and Coreset Construction*. ESA 2019. DOI: [10.4230/LIPIcs.ESA.2019.40](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2019.40)
- Charikar, Khuller, Mount, Narasimhan. *Algorithms for Facility Location Problems with Outliers*. SODA 2001.
