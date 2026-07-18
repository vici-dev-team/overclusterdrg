import pandas as pd

# Load CSV
df = pd.read_csv("LP_Results.csv")

# -------------------------------------------------
# TABLE 1: 36 rows (N,k,outlier) with 3 radius cols
# -------------------------------------------------

table1 = df[[
    "N",
    "k",
    "outlier_percent",
    "LP_Radius",
    "charikar_radius",
    "gonzalez_radius"
]].copy()

table1 = table1.sort_values(["N","k","outlier_percent"])

# Save
table1.to_csv("radius_table_all_cases.csv", index=False)

print("Table 1 (36 x 3 radii + identifiers):")
print(table1)


# -------------------------------------------------
# TABLE 2: average radius over N
# rows = (k, outlier_percent)
# -------------------------------------------------

table2 = (
    df.groupby(["k","outlier_percent"])[
        ["LP_Radius","charikar_radius","gonzalez_radius"]
    ]
    .mean()
    .reset_index()
)

table2 = table2.sort_values(["k","outlier_percent"])

# Save
table2.to_csv("radius_table_avg_over_N.csv", index=False)

print("\nTable 2 (Average over N):")
print(table2)