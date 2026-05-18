"""Step 4: build final harmonized outputs for each target SOC version.

This step merges statements, canonical IDs, and ratings into the final dataset
and writes one long file plus one file per SOC target version.
"""

import pandas as pd

from pipeline_utils import INTERMEDIATE_DIR, OUTPUT_DIR, TARGET_SOC_VERSIONS


def run_step4() -> None:
    print("[step4] Loading harmonized statements, canon IDs, and ratings...")
    harmonized = pd.read_csv(INTERMEDIATE_DIR / "task_statements_harmonized_long.csv")
    canons = pd.read_csv(INTERMEDIATE_DIR / "task_statements_with_canon_id.csv")
    ratings = pd.read_csv(INTERMEDIATE_DIR / "task_ratings_harmonized.csv")

    canons = canons[["task_clean", "canon_id"]].drop_duplicates()
    merged = harmonized.merge(canons, on="task_clean", how="left")
    merged = merged.merge(
        ratings,
        left_on=["source_onet_soc_code", "ONET_release_year", "canon_id"],
        right_on=["O*NET-SOC Code", "year", "canon_id"],
        how="left",
    )

    # Normalize key identifier dtypes so downstream filtering/writes are stable.
    merged["target_soc_version"] = merged["target_soc_version"].astype(str).str.strip()
    merged["target_soc_code"] = merged["target_soc_code"].astype(str).str.strip()
    merged["source_onet_soc_version"] = merged["source_onet_soc_version"].astype(str).str.strip()
    merged["source_onet_soc_code"] = merged["source_onet_soc_code"].astype(str).str.strip()

    merged["task_intensity"] = merged["mean_importance"] * merged["mean_frequency"]
    merged["first_seen"] = merged.groupby(["target_soc_version", "target_soc_code", "canon_id"])["ONET_release_year"].transform("min")
    merged["last_seen"] = merged.groupby(["target_soc_version", "target_soc_code", "canon_id"])["ONET_release_year"].transform("max")

    merged.to_csv(OUTPUT_DIR / "onet_harmonized_all_versions_long.csv", index=False)
    print(f"[step4] Wrote onet_harmonized_all_versions_long.csv with {len(merged):,} rows")

    for version in TARGET_SOC_VERSIONS:
        subset = merged[merged["target_soc_version"] == str(version)].copy()
        subset.to_csv(OUTPUT_DIR / f"onet_harmonized_target_soc_{version}.csv", index=False)
        print(f"[step4] target_soc_{version}: {len(subset):,} rows")
    print(f"[step4] Wrote {len(TARGET_SOC_VERSIONS)} target-version output files")
