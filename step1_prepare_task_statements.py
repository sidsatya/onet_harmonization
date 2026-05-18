"""Step 1: load historical task statements and harmonize SOC codes across versions.

This step reads all task statement releases, infers each row's source SOC version,
maps each source occupation into every target SOC taxonomy, and writes long-format
harmonized rows to `intermediate_data/`.
"""

from collections import deque

import pandas as pd

from pipeline_utils import DATA_DIR, INTERMEDIATE_DIR, TARGET_SOC_VERSIONS, TASK_STATEMENT_FILES, clean_text


CURRENT_ONET_SOC_BY_YEAR = {
    **{year: "2000" for year in range(2003, 2006)},
    **{year: "2006" for year in range(2006, 2009)},
    **{year: "2009" for year in range(2009, 2011)},
    **{year: "2010" for year in range(2011, 2020)},
    **{year: "2019" for year in range(2020, 2026)},
}


def _build_directed_maps() -> dict[tuple[str, str], dict[str, set[str]]]:
    cw_dir = DATA_DIR / "onet" / "onet_occsoc_crosswalks"
    maps = {}

    def read_map(file_name: str, source_col: str, target_col: str, source_v: str, target_v: str):
        df = pd.read_csv(cw_dir / file_name, dtype=str).fillna("")
        mapping = {}
        for _, row in df.iterrows():
            source_code = row[source_col].strip()
            target_code = row[target_col].strip()
            if not source_code or not target_code:
                continue
            mapping.setdefault(source_code, set()).add(target_code)
        maps[(source_v, target_v)] = mapping
        reverse = {}
        for source_code, targets in mapping.items():
            for target_code in targets:
                reverse.setdefault(target_code, set()).add(source_code)
        maps[(target_v, source_v)] = reverse

    read_map("onet_2000_to_2006_crosswalk.csv", "O*NET-SOC 2000 Code", "O*NET-SOC 2006 Code", "2000", "2006")
    read_map("onet_2006_to_2009_crosswalk.csv", "O*NET-SOC 2006 Code", "O*NET-SOC 2009 Code", "2006", "2009")
    read_map("onet_2009_to_2010_crosswalk.csv", "O*NET-SOC 2009 Code", "O*NET-SOC 2010 Code", "2009", "2010")
    read_map("onet_2010_to_2019_crosswalk.csv", "O*NET-SOC 2010 Code", "O*NET-SOC 2019 Code", "2010", "2019")
    read_map("onet_2019_to_2018_crosswalk.csv", "O*NET-SOC 2019 Code", "2018 SOC Code", "2019", "2018")
    return maps


def _build_title_lookup(task_data: pd.DataFrame) -> dict[tuple[str, str], str]:
    title_lookup: dict[tuple[str, str], str] = {}

    # Titles from task statement files (only present in some releases, e.g. 2019+ files).
    if "Title" in task_data.columns:
        with_titles = task_data.dropna(subset=["Title"]).copy()
        for _, row in with_titles.iterrows():
            version = CURRENT_ONET_SOC_BY_YEAR[int(row["ONET_release_year"])]
            code = str(row["O*NET-SOC Code"]).strip()
            title = str(row["Title"]).strip()
            if code and title:
                title_lookup[(version, code)] = title

    # Titles from crosswalks.
    cw_dir = DATA_DIR / "onet" / "onet_occsoc_crosswalks"
    crosswalk_specs = [
        (
            "onet_2000_to_2006_crosswalk.csv",
            ("2000", "O*NET-SOC 2000 Code", "O*NET-SOC 2000 Title"),
            ("2006", "O*NET-SOC 2006 Code", "O*NET-SOC 2006 Title"),
        ),
        (
            "onet_2006_to_2009_crosswalk.csv",
            ("2006", "O*NET-SOC 2006 Code", "O*NET-SOC 2006 Title"),
            ("2009", "O*NET-SOC 2009 Code", "O*NET-SOC 2009 Title"),
        ),
        (
            "onet_2009_to_2010_crosswalk.csv",
            ("2009", "O*NET-SOC 2009 Code", "O*NET-SOC 2009 Title"),
            ("2010", "O*NET-SOC 2010 Code", "O*NET-SOC 2010 Title"),
        ),
        (
            "onet_2010_to_2019_crosswalk.csv",
            ("2010", "O*NET-SOC 2010 Code", "O*NET-SOC 2010 Title"),
            ("2019", "O*NET-SOC 2019 Code", "O*NET-SOC 2019 Title"),
        ),
        (
            "onet_2019_to_2018_crosswalk.csv",
            ("2019", "O*NET-SOC 2019 Code", "O*NET-SOC 2019 Title"),
            ("2018", "2018 SOC Code", "2018 SOC Title"),
        ),
    ]

    for file_name, left_spec, right_spec in crosswalk_specs:
        df = pd.read_csv(cw_dir / file_name, dtype=str).fillna("")
        for _, row in df.iterrows():
            for version, code_col, title_col in (left_spec, right_spec):
                code = str(row.get(code_col, "")).strip()
                title = str(row.get(title_col, "")).strip()
                if code and title and (version, code) not in title_lookup:
                    title_lookup[(version, code)] = title

    return title_lookup


def _reachable_with_maps(maps, start_version: str, start_code: str, target_version: str) -> list[str]:
    queue = deque([(start_version, start_code)])
    seen = {(start_version, start_code)}
    results = set()
    while queue:
        version, code = queue.popleft()
        if version == target_version:
            results.add(code)
            continue
        for (src_v, dst_v), mapping in maps.items():
            if src_v != version:
                continue
            for next_code in mapping.get(code, set()):
                state = (dst_v, next_code)
                if state not in seen:
                    seen.add(state)
                    queue.append(state)
    return sorted(results)


def run_step1() -> None:
    print("[step1] Loading historical task statement files...")
    frames = []
    stmt_dir = DATA_DIR / "onet" / "historical_onet_task_statements"
    for year, filename in TASK_STATEMENT_FILES.items():
        path = stmt_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, encoding="latin1")
        if "O*NET-SOC Code" not in df.columns:
            continue
        df["ONET_release_year"] = year
        df["O*NET-SOC Code"] = df["O*NET-SOC Code"].str.strip()
        frames.append(df)

    if not frames:
        raise RuntimeError("No task statement files found.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[step1] Loaded {len(frames)} files and {len(data):,} rows")
    maps = _build_directed_maps()
    print("[step1] Built SOC crosswalk mappings")
    title_lookup = _build_title_lookup(data)
    print(f"[step1] Built occupation title lookup with {len(title_lookup):,} code-title pairs")

    harmonized_rows = []
    for _, row in data.iterrows():
        source_year = int(row["ONET_release_year"])
        source_version = CURRENT_ONET_SOC_BY_YEAR[source_year]
        source_code = row["O*NET-SOC Code"]
        for target_version in TARGET_SOC_VERSIONS:
            target_codes = _reachable_with_maps(maps, source_version, source_code, target_version)
            if not target_codes:
                continue
            for target_code in target_codes:
                harmonized_rows.append(
                    {
                        "ONET_release_year": source_year,
                        "source_onet_soc_version": source_version,
                        "source_onet_soc_code": source_code,
                        "source_occupation_title": title_lookup.get((source_version, source_code)),
                        "target_soc_version": target_version,
                        "target_soc_code": target_code,
                        "target_occupation_title": title_lookup.get((target_version, target_code)),
                        "Task ID": row.get("Task ID"),
                        "Task": row.get("Task"),
                        "Task Type": row.get("Task Type"),
                        "Incumbents Responding": row.get("Incumbents Responding"),
                        "Date": row.get("Date"),
                        "Domain Source": row.get("Domain Source"),
                    }
                )

    harmonized = pd.DataFrame(harmonized_rows)
    harmonized["task_clean"] = harmonized["Task"].apply(clean_text)
    print(f"[step1] Produced {len(harmonized):,} harmonized rows across target SOC versions")

    harmonized.to_csv(INTERMEDIATE_DIR / "task_statements_harmonized_long.csv", index=False)
    (
        harmonized[["Task", "task_clean"]]
        .dropna()
        .drop_duplicates()
        .to_csv(INTERMEDIATE_DIR / "unique_task_statements.csv", index=False)
    )
    (
        harmonized[["Task ID", "Task"]]
        .dropna()
        .drop_duplicates()
        .to_csv(INTERMEDIATE_DIR / "task_statements_and_ids.csv", index=False)
    )
    print("[step1] Wrote intermediate files: task_statements_harmonized_long.csv, unique_task_statements.csv, task_statements_and_ids.csv")
