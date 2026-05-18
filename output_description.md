# Output File Description

This document describes the structure of:

- `output/onet_harmonized_all_versions_long.csv`

and how to interpret key fields.

## What This File Contains

`onet_harmonized_all_versions_long.csv` is the master long-format table produced in Step 4.  
Each row represents a task statement row from a specific O*NET release year, mapped to one target SOC taxonomy/version.

Because crosswalks can be one-to-many, a single source row may produce multiple rows in this file.

## Core Identifier Fields

- `ONET_release_year`: O*NET release year for the source row.
- `source_onet_soc_version`: source O*NET-SOC taxonomy version inferred from release year (`2000`, `2006`, `2009`, `2010`, `2019`).
- `source_onet_soc_code`: original occupation code in the source taxonomy.
- `source_occupation_title`: occupation title for the source code (when available from statement files or crosswalk titles).
- `target_soc_version`: harmonization target taxonomy version (`2000`, `2006`, `2009`, `2010`, `2019`, `2018`).
- `target_soc_code`: mapped occupation code in the target taxonomy.
- `target_occupation_title`: occupation title for the mapped target code.
- `canon_id`: canonical task cluster ID (embedding-based task harmonization).

## Task Content Fields

- `Task ID`: O*NET task identifier from source statement file.
- `Task`: raw task text.
- `task_clean`: normalized task text used for canonical clustering.
- `Task Type`: O*NET task type (for example, Core / Supplemental).
- `Incumbents Responding`: respondent count field from O*NET source when present.
- `Date`: source record date field from O*NET source.
- `Domain Source`: source domain metadata from O*NET file.

## Rating Fields

These are merged from harmonized task ratings:

- `mean_importance`: importance score (from `Scale ID = IM`).
- `importance_normalized_all`: within-occupation-year normalized importance.
- `mean_frequency`: expected frequency derived from `Scale ID = FT`.
- `task_intensity`: `mean_importance * mean_frequency`.

## Time Coverage Fields

- `first_seen`: first `ONET_release_year` where this `(target_soc_version, target_soc_code, canon_id)` appears.
- `last_seen`: last `ONET_release_year` where this `(target_soc_version, target_soc_code, canon_id)` appears.

## Related Final Files

Per-version exports are filtered subsets of this master file:

- `output/onet_harmonized_target_soc_2000.csv`
- `output/onet_harmonized_target_soc_2006.csv`
- `output/onet_harmonized_target_soc_2009.csv`
- `output/onet_harmonized_target_soc_2010.csv`
- `output/onet_harmonized_target_soc_2018.csv`
- `output/onet_harmonized_target_soc_2019.csv`
