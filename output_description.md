# Understanding The Main Output

The main output file, `output/onet_harmonized_all_versions_long.csv`, is the “master table” for this project. If you want one place that contains the full harmonized history, this is it. The per-version files (`onet_harmonized_target_soc_2018.csv`, etc.) are just filtered slices of this same dataset.

At a high level, each row is one task observation from a specific O*NET release year, mapped from its original occupation code into one target SOC version. Since SOC crosswalks can split one source occupation into multiple target occupations, one original source row can expand into multiple rows here. That expansion is expected and is part of preserving valid mappings instead of forcing a lossy one-to-one match.

## How To Read A Row

The easiest way to think about a row is: “this task came from this source occupation and year, and after harmonization it maps to this target occupation/version, with this canonical task ID and these rating fields.”

`ONET_release_year` tells you which historical O*NET release the source row came from. `source_onet_soc_version` and `source_onet_soc_code` tell you the source taxonomy version and code from that release context. `source_occupation_title` is the human-readable title for that source code (when available from source files/crosswalk titles).

`target_soc_version` and `target_soc_code` tell you where that source row landed after harmonization. `target_occupation_title` is the corresponding target title. If you are doing analysis in one standardized system (for example, SOC 2018), this is the pair you will usually filter on first.

## Task Identity And Canonicalization

`Task ID` and `Task` are the original O*NET task fields. Because task wording can drift over time (“maintain records” vs “maintain detailed records”), the pipeline also creates `task_clean` and then assigns `canon_id`, which is the canonical cluster ID used to group semantically similar statements together.

In practice, `canon_id` is what gives you stable task identity across years/releases, while `Task` is the raw wording snapshot from the source year.

## Ratings And Derived Metrics

The rating fields are merged after canonicalization. `mean_importance` comes from O*NET importance ratings. `importance_normalized_all` is the within-occupation-year normalized share so you can compare task importance composition inside each occupation-year group. `mean_frequency` is the expected frequency measure derived from frequency scale inputs.

`task_intensity` is a simple combined measure (`mean_importance * mean_frequency`) and is useful when you want one number capturing both salience and expected repetition.

The important thing to know is that this rating pipeline is separate from statement crosswalk harmonization logic. Task statements drive mapping and canonicalization; ratings are linked afterward using occupation code, year, and `canon_id`.

## Rating Assumptions (What We Chose)

For ratings, the current pipeline uses O*NET task rating files and keeps rows where `Recommend Suppress != "Y"` and `Scale ID` is either `IM` (importance) or `FT` (frequency). That means suppressed rows are excluded by design, and non-IM/FT scales are not part of the final metrics.

`mean_importance` is the raw `Data Value` from IM rows. `importance_normalized_all` is computed as each task’s importance divided by the sum of importance values within the same occupation-year, so it behaves like a within-occupation composition share for that year.

`mean_frequency` is built from FT rows by computing a weighted frequency total (`Category * Data Value`) aggregated by occupation-year-canon task, then dividing by 100 to convert the percentage-style scale into an expected-frequency style measure.

`task_intensity` assumes multiplicative combination is a useful proxy for “important and frequent.” In other words, it treats importance and frequency as complementary dimensions and uses their product rather than a weighted average.

## Time-Coverage Fields

`first_seen` and `last_seen` are computed for each `(target_soc_version, target_soc_code, canon_id)` combination. They give you the observed lifespan of that harmonized task-within-occupation in the historical panel. This is helpful for tracking emergence, persistence, and decline patterns.

## Important Practical Notes

Because the file is long-format and mapping-preserving, row counts can be larger than raw input statement counts. That is normal. Also, title fields are best-effort and come from available source/crosswalk title columns, so occasional blanks can still happen for sparse historical records.

If you only want one taxonomy target for modeling, filter to one `target_soc_version` (for example, `"2018"`) and use the corresponding `target_soc_code` plus `canon_id` as your core panel keys. If you want to study classification drift itself, keep the full file and compare across target versions.
