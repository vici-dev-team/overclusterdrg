#!/usr/bin/env bash
# Run this once from ~/overclusterdrg-paper/ to populate the repo.
# Usage: bash setup_repo.sh

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
DL="$HOME/Downloads"
KC="$DL/K-Centre"

echo "=== Copying paper files ==="
cp "$DL/teacher_draft_updated.tex"     "$REPO/paper/"
cp "$DL/teacher_paper.tex"             "$REPO/paper/"
cp "$DL/overclusterdrg_final.tex"      "$REPO/paper/"
cp "$DL/overclusterdrg_alenex_v2.tex"  "$REPO/paper/" 2>/dev/null || true
cp "$DL/overclusterdrg_alenex.tex"     "$REPO/paper/" 2>/dev/null || true
cp "$DL/updates.tex"                   "$REPO/paper/" 2>/dev/null || true

echo "=== Copying core algorithm code ==="
mkdir -p "$REPO/code"
for f in drg.py gonzalez.py rkc.py \
          LP_relaxed.py LP_2.py LP_Relaxed1.py \
          run_all_algorithms.py run_oclocal.py \
          run_ding.py run_gonzalez_charikar.py \
          analysis.py prepare_datasets.py \
          adult_final_dataset.py diabetes_dataset.py \
          covertype_dataset.py htru2_dataset.py \
          bank_dataset.py shuttle_dataset.py \
          dingvOC.py drg_comparison.py drg_comparisons.py \
          drg_standalone.py drg1.py \
          gonzalez_comparison.py gonzalez_old.py \
          explore.py explore2.py \
          revision_experiments.py revision_part2.py \
          exhaustive.py oc_multi_dataset.py \
          interpolate_lp.py tables.py plots_summary.py ufc.py; do
    cp "$KC/$f" "$REPO/code/" 2>/dev/null && echo "  copied $f" || echo "  skipped $f (not found)"
done

echo "=== Copying paper experiment scripts ==="
mkdir -p "$REPO/code/experiments"
for f in run_scalability.py run_htru2.py run_plots.py run_simplified_drg.py; do
    cp "$KC/paper/experiments/$f" "$REPO/code/experiments/" 2>/dev/null && echo "  copied $f" || echo "  skipped $f"
done

echo "=== Copying key results ==="
mkdir -p "$REPO/results/paper_findings"
cp "$KC/final_comparison.csv"           "$REPO/results/" 2>/dev/null || true
cp "$KC/LP_Results.csv"                 "$REPO/results/" 2>/dev/null || true
cp "$KC/full_results_diabetes.csv"      "$REPO/results/" 2>/dev/null || true
cp "$KC/full_results_covertype.csv"     "$REPO/results/" 2>/dev/null || true
cp "$KC/certified_ratios.csv"           "$REPO/results/" 2>/dev/null || true
cp "$KC/radius_table_all_cases.csv"     "$REPO/results/" 2>/dev/null || true
cp "$KC/paper_findings/"*.csv           "$REPO/results/paper_findings/" 2>/dev/null || true

echo "=== Done. Run: git add -A && git commit -m 'Initial commit' ==="
