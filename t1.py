import pandas as pd
from rapidfuzz import process, fuzz

df1 = pd.read_excel("Koppal_Union_Plot_Soil.xlsx")
df2 = pd.read_excel("Copy without cleaning Chandura_SWS_Soil_Phase_LMU_Centroid_Hissa - - Copy.xlsx")

# Remove duplicate columns from both files
df1 = df1.loc[:, ~df1.columns.duplicated()]
df2 = df2.loc[:, ~df2.columns.duplicated()]

# Auto-match df2 columns to df1 columns
column_mapping = {}
already_mapped = set()

print("\n=== Column Mapping Summary ===")
print(f"{'Excel 2 Column':<40} {'Mapped to Excel 1 Column':<40} {'Score'}")
print("-" * 90)

for col in df2.columns:
    match, score, _ = process.extractOne(col, df1.columns, scorer=fuzz.token_sort_ratio)
    if score >= 70 and match not in already_mapped:
        column_mapping[col] = match
        already_mapped.add(match)
        print(f"{col:<40} {match:<40} {score}")
    else:
        print(f"{col:<40} {'NO MATCH':<40} {score}")

# Manual overrides for columns too different to fuzzy match
manual_mapping = {
    "KGISTalukName": "Taluk",
    "Latitude":      "latitude",
    "Longitude":     "longitude",
    "Upaddy_Leg":    "UPaddy_Leg",
}
column_mapping.update(manual_mapping)

print(f"\nTotal matched: {len(column_mapping)} / {len(df2.columns)}")

# Rename df2 columns
df2_renamed = df2.rename(columns=column_mapping)

# Remove any duplicates created after renaming
df2_renamed = df2_renamed.loc[:, ~df2_renamed.columns.duplicated()]

# Keep only df1 columns that exist in df2 after renaming
common = [col for col in df1.columns if col in df2_renamed.columns]

print(f"\nColumns used in combined file ({len(common)}):")
for col in common:
    print(f"  - {col}")

# Select only common columns from both
df1_common = df1[common].copy()
df2_common = df2_renamed[common].copy()

# Reset index before concat
df1_common = df1_common.reset_index(drop=True)
df2_common = df2_common.reset_index(drop=True)

combined = pd.concat([df1_common, df2_common], ignore_index=True)
combined.to_excel("combined2.xlsx", index=False)
print(f"\nDone. Total rows: {len(combined)}, Columns: {len(common)}")