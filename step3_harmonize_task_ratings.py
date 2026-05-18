"""Step 3: harmonize task ratings and attach canonical task IDs.

This step processes historical rating files, links rows to `canon_id`, and
computes harmonized importance/frequency metrics by occupation-year.
"""

import numpy as np
import pandas as pd

from pipeline_utils import DATA_DIR, INTERMEDIATE_DIR, TASK_RATING_FILES, clean_text


def _load_ratings() -> pd.DataFrame:
    ratings_dir = DATA_DIR / "onet" / "historical_onet_task_ratings"
    frames = []
    for year, filename in TASK_RATING_FILES.items():
        path = ratings_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="latin1")
        df["year"] = year
        frames.append(df)
    if not frames:
        raise RuntimeError("No task rating files found.")
    return pd.concat(frames, ignore_index=True)


def run_step3() -> None:
    print("[step3] Loading task ratings and canon mappings...")
    ratings = _load_ratings()
    canons = pd.read_csv(INTERMEDIATE_DIR / "task_statements_with_canon_id.csv")
    task_id_map = pd.read_csv(INTERMEDIATE_DIR / "task_statements_and_ids.csv")

    canons["task_clean"] = canons["Task"].apply(clean_text)
    canon_lookup = canons.drop_duplicates("task_clean").set_index("task_clean")["canon_id"]
    task_id_to_task = task_id_map.drop_duplicates("Task ID").set_index("Task ID")["Task"]

    if "Task" not in ratings.columns:
        ratings["Task"] = np.nan
    ratings["task_clean"] = ratings["Task"].astype(str).apply(clean_text)
    missing_task = ratings["Task"].isna() & ratings["Task ID"].notna()
    ratings.loc[missing_task, "task_clean"] = ratings.loc[missing_task, "Task ID"].map(task_id_to_task).astype(str).apply(clean_text)
    ratings["canon_id"] = ratings["task_clean"].map(canon_lookup)
    print(f"[step3] Loaded {len(ratings):,} rating rows")

    filtered = ratings[(ratings["Recommend Suppress"] != "Y") & (ratings["Scale ID"].isin(["IM", "FT"]))].copy()
    print(f"[step3] Retained {len(filtered):,} rows after suppression/scale filters")

    im = filtered[filtered["Scale ID"] == "IM"].copy()
    im["Data Value"] = pd.to_numeric(im["Data Value"], errors="coerce")
    im_sum = im.groupby(["O*NET-SOC Code", "year"])["Data Value"].transform("sum")
    im["importance_normalized_all"] = im["Data Value"] / im_sum

    ft = filtered[filtered["Scale ID"] == "FT"].copy()
    ft["Category"] = pd.to_numeric(ft["Category"], errors="coerce")
    ft["Data Value"] = pd.to_numeric(ft["Data Value"], errors="coerce")
    ft["weighted_frequency"] = ft["Category"] * ft["Data Value"]
    ft_agg = ft.groupby(["O*NET-SOC Code", "canon_id", "year"], as_index=False)["weighted_frequency"].sum()
    ft_agg["mean_frequency"] = ft_agg["weighted_frequency"] / 100.0

    merged = pd.merge(
        im[["O*NET-SOC Code", "canon_id", "year", "Date", "Data Value", "importance_normalized_all"]],
        ft_agg[["O*NET-SOC Code", "canon_id", "year", "mean_frequency"]],
        on=["O*NET-SOC Code", "canon_id", "year"],
        how="left",
    )
    merged = merged.rename(columns={"Data Value": "mean_importance"})
    merged.to_csv(INTERMEDIATE_DIR / "task_ratings_harmonized.csv", index=False)
    print(f"[step3] Wrote task_ratings_harmonized.csv with {len(merged):,} rows")
