"""Diagnostics for 2018 harmonization coverage and occupation/task dropouts.

This script evaluates the 2018-target output and creates CSV diagnostics that show:
1) source O*NET-SOC codes that never map to 2018,
2) mapped 2018 SOC codes with no task rows,
3) mapped 2018 SOC codes with no rating rows,
4) basic coverage summaries by release year.
"""

import pandas as pd

from pipeline_utils import INTERMEDIATE_DIR, OUTPUT_DIR


def run_diagnostics_2018() -> None:
    diagnostics_dir = OUTPUT_DIR / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    step1 = pd.read_csv(INTERMEDIATE_DIR / "task_statements_harmonized_long.csv")
    final_2018 = pd.read_csv(OUTPUT_DIR / "onet_harmonized_target_soc_2018.csv")
    step1["target_soc_version"] = step1["target_soc_version"].astype(str).str.strip()

    source_codes = (
        step1[["ONET_release_year", "source_onet_soc_code"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"source_onet_soc_code": "source_code"})
    )
    mapped_source_codes = (
        step1[step1["target_soc_version"] == "2018"][["ONET_release_year", "source_onet_soc_code", "source_occupation_title"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"source_onet_soc_code": "source_code", "source_occupation_title": "source_title"})
    )
    source_titles = (
        step1[["ONET_release_year", "source_onet_soc_code", "source_occupation_title"]]
        .dropna(subset=["source_onet_soc_code"])
        .drop_duplicates()
        .rename(columns={"source_onet_soc_code": "source_code", "source_occupation_title": "source_title"})
    )
    source_codes = source_codes.merge(source_titles, on=["ONET_release_year", "source_code"], how="left")
    dropped_source_codes = source_codes.merge(
        mapped_source_codes,
        on=["ONET_release_year", "source_code"],
        how="left",
        indicator=True,
    )
    dropped_source_codes = dropped_source_codes[dropped_source_codes["_merge"] == "left_only"].drop(
        columns=["_merge", "source_title_y"], errors="ignore"
    )
    dropped_source_codes = dropped_source_codes.rename(columns={"source_title_x": "source_title"})
    dropped_source_codes.to_csv(diagnostics_dir / "2018_unmapped_source_occupations.csv", index=False)

    mapped_2018_codes = (
        step1[step1["target_soc_version"] == "2018"][["target_soc_code", "target_occupation_title"]]
        .dropna(subset=["target_soc_code"])
        .astype({"target_soc_code": "string"})
        .drop_duplicates()
        .rename(columns={"target_occupation_title": "target_title"})
    )
    tasks_by_2018 = (
        final_2018[["target_soc_code", "target_occupation_title", "Task"]]
        .dropna(subset=["target_soc_code"])
        .groupby(["target_soc_code", "target_occupation_title"], as_index=False)["Task"]
        .nunique()
        .rename(columns={"Task": "unique_tasks", "target_occupation_title": "target_title"})
    )
    ratings_by_2018 = (
        final_2018[["target_soc_code", "target_occupation_title", "mean_importance"]]
        .dropna(subset=["target_soc_code"])
        .groupby(["target_soc_code", "target_occupation_title"], as_index=False)["mean_importance"]
        .count()
        .rename(columns={"mean_importance": "rated_rows", "target_occupation_title": "target_title"})
    )

    no_tasks = mapped_2018_codes.merge(tasks_by_2018, on=["target_soc_code", "target_title"], how="left")
    no_tasks = no_tasks[no_tasks["unique_tasks"].fillna(0) == 0]
    no_tasks.to_csv(diagnostics_dir / "2018_mapped_occupations_without_tasks.csv", index=False)

    no_ratings = mapped_2018_codes.merge(ratings_by_2018, on=["target_soc_code", "target_title"], how="left")
    no_ratings = no_ratings[no_ratings["rated_rows"].fillna(0) == 0]
    no_ratings.to_csv(diagnostics_dir / "2018_mapped_occupations_without_ratings.csv", index=False)

    yearly_summary = (
        final_2018.groupby("ONET_release_year", as_index=False)
        .agg(
            unique_2018_occupations=("target_soc_code", "nunique"),
            unique_tasks=("Task", "nunique"),
            rows_with_ratings=("mean_importance", lambda s: int(s.notna().sum())),
            total_rows=("target_soc_code", "size"),
        )
        .sort_values("ONET_release_year")
    )
    yearly_summary.to_csv(diagnostics_dir / "2018_yearly_coverage_summary.csv", index=False)

    print(f"Saved diagnostics to {diagnostics_dir}")


if __name__ == "__main__":
    run_diagnostics_2018()
