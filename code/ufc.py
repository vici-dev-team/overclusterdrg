# ================================================================
# update_final_comparison.py
#
# Reads benchmark_ding_vs_drg.csv and final_comparison1.csv,
# overwrites DingRKC_radius and DingRKC_time in the comparison
# table with the corrected Ding et al. results, saves in-place.
# ================================================================

import pandas as pd

DING_FILE = "benchmark_ding_vs_drg.csv"
COMP_FILE = "final_comparison1.csv"

# ----------------------------------------------------------------
# Load
# ----------------------------------------------------------------
ding = pd.read_csv(DING_FILE)
comp = pd.read_csv(COMP_FILE)

print(f"Loaded {DING_FILE}: {len(ding)} rows")
print(f"Loaded {COMP_FILE}: {len(comp)} rows")

# ----------------------------------------------------------------
# Build lookup: (N, k, z) → (ding_radius, ding_time_s)
# ----------------------------------------------------------------
lookup = {
    (int(row.N), int(row.k), int(row.z)): (row.ding_radius, row.ding_time_s)
    for _, row in ding.iterrows()
}

# ----------------------------------------------------------------
# Update
# ----------------------------------------------------------------
updated = 0
missing = []

for idx, row in comp.iterrows():
    key = (int(row.N), int(row.k), int(row.z))
    if key in lookup:
        comp.at[idx, "DingRKC_radius"] = lookup[key][0]
        comp.at[idx, "DingRKC_time"]   = lookup[key][1]
        updated += 1
    else:
        missing.append(key)

# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------
comp.to_csv(COMP_FILE, index=False)

print(f"\nUpdated {updated}/{len(comp)} rows in {COMP_FILE}")
if missing:
    print(f"No match found for {len(missing)} rows:")
    for key in missing:
        print(f"  N={key[0]}, k={key[1]}, z={key[2]}")

# ----------------------------------------------------------------
# Quick sanity check
# ----------------------------------------------------------------
print("\nSample of updated rows:")
print(comp[["N", "k", "z", "DingRKC_radius", "DingRKC_time"]].to_string(index=False))