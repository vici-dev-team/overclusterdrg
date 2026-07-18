# OverclusterDRG

**OverclusterDRG: A Deterministic, LP-Free Heuristic for Robust k-Center with Outliers**

M.Sc. Major Project — Rudra Bhardwaj (24234747002), MCSC401, University of Delhi

---

## Structure

```
paper/
  teacher_draft_updated.tex   — ALENEX SIAM proceedings draft (main paper)
  teacher_paper.tex           — Teacher's original draft
  overclusterdrg_final.tex    — Standalone 11pt report version
  refs.bib                    — Bibliography

code/
  drg.py                      — OverclusterDRG (OC-Forward + OC-Local)
  gonzalez.py                 — Gonzalez k-center algorithm
  rkc.py                      — Charikar 3-approximation (RKC)
  LP_relaxed.py               — LP lower bound solver (Gurobi)
  run_all_algorithms.py       — Main experiment runner
  run_oclocal.py              — OC-Local runner
  run_ding.py                 — Ding et al. randomised baseline
  run_gonzalez_charikar.py    — Gonzalez/Charikar baseline
  analysis.py                 — Analysis with raw results block
  prepare_datasets.py         — Dataset preprocessing
  adult_final_dataset.py      — UCI Adult loader
  diabetes_dataset.py         — Diabetes Indicators loader
  covertype_dataset.py        — Covertype loader
  htru2_dataset.py            — HTRU2 loader
  experiments/
    run_scalability.py        — Large-N scalability experiments
    run_htru2.py
    run_plots.py

results/
  final_comparison.csv        — Per-configuration OC-Local vs baselines
  LP_Results.csv              — Exact LP lower bounds (Adult)
  full_results_diabetes.csv   — Diabetes full results
  full_results_covertype.csv  — Covertype full results
  paper_findings/             — Tables as used in the paper
    table1_lp_ratios_by_dataset.csv
    table2_certified_ratios.csv
    table3_adult_all36_CORRECTED.csv
    table4_diabetes_all36.csv
    table5_covertype_all30.csv
    table6_adaptive_pool.csv
    table7_wilcoxon_CORRECTED.csv
    table8_knee_ols.csv
    table9_rho_O_summary.csv
```

## Key Results

- OC-Local mean LP ratio: **1.147** (Adult), 1.155 (Diabetes), 1.171 (Covertype)
- Beats Charikar in **33/36** Adult configurations at **7x** median speedup
- Beats Ding in **35/36** at **8x** speedup
- Phase transition at O=z: proven exact threshold (Theorems 1 & 2)
- All paired tests significant at p < 1e-5 (t-test and Wilcoxon)

## Setup

```bash
# Populate repo from Downloads (run once)
bash setup_repo.sh

# Compile paper (requires siamproceedings.sty + refs.bib)
cd paper
pdflatex teacher_draft_updated.tex
bibtex teacher_draft_updated
pdflatex teacher_draft_updated.tex
pdflatex teacher_draft_updated.tex

# Run experiments (requires Gurobi, numpy, scipy, pandas)
python code/run_all_algorithms.py
```

## Algorithms

| Algorithm | Type | LP ratio (Adult) | Time (N=20k) |
|---|---|---|---|
| OC-Local | Deterministic | 1.147 | 57s |
| OC-Forward | Deterministic | 1.158 | 52s |
| Charikar | Deterministic | 1.173 | 564s |
| Ding et al. | Randomised | 1.235 | 431s |
| RKC-Gonzalez | Heuristic | 1.280 | 0.02s |

## Theory

Phase transition theorem: the Gonzalez pool with O=z extra steps is guaranteed to contain a 3·OPT solution. One step fewer (O=z-1) and the ratio is unbounded for any k≥2.
