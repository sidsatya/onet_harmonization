# O*NET Task Harmonization Pipeline

A reproducible Python pipeline for harmonizing historical O*NET task statements and task ratings across O*NET-SOC taxonomies, with final exports for each target SOC version.

## Key Features

- Single command orchestration (`run_onet_harmonization.py`)
- Data:
  - `data/` for raw inputs
  - `intermediate_data/` for transient processing files
  - `output/` for final output
- Harmonization outputs for all configured target SOC versions:
  - O*NET-SOC 2000, 2006, 2009, 2010, 2019
  - SOC 2018
- Canonical task clustering using semantic embeddings
- Diagnostics script for 2018 coverage and occupation dropouts

## Project Structure

- `run_onet_harmonization.py`: central runner for steps 1-4
- `pipeline_utils.py`: shared constants, paths, and text cleaning
- `step1_prepare_task_statements.py`: SOC crosswalk harmonization of statements
- `step2_cluster_task_statements.py`: embedding-based canonical task clustering
- `step3_harmonize_task_ratings.py`: harmonization of importance/frequency ratings
- `step4_build_final_outputs.py`: merge + final output generation
- `diagnostics_2018.py`: diagnostics for 2018 harmonization coverage

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Step 2 requires an OpenAI API key for embeddings.

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The pipeline loads this automatically via `python-dotenv`.

## Required Input Data

Place required source files under `data/onet/`:

1. `historical_onet_task_statements/`
2. `historical_onet_task_ratings/`
3. `onet_occsoc_crosswalks/`

Required crosswalk files:

- `onet_2000_to_2006_crosswalk.csv`
- `onet_2006_to_2009_crosswalk.csv`
- `onet_2009_to_2010_crosswalk.csv`
- `onet_2010_to_2019_crosswalk.csv`
- `onet_2019_to_2018_crosswalk.csv`

## Harmonization Method

1. Load all historical task statement releases and infer each row’s source O*NET-SOC taxonomy by release year.
2. Traverse crosswalk chains and map each source code to every target SOC version.
3. Expand one-to-many mappings so every valid source-target occupation mapping is preserved.
4. Cluster semantically similar task statements and assign canonical task IDs (`canon_id`).
5. Harmonize task ratings and link them to canonical tasks.
6. Merge statements + canonical IDs + ratings, then compute `task_intensity`, `first_seen`, and `last_seen`.
7. Write final exports to `output/`.

## Run The Pipeline

```bash
python3 run_onet_harmonization.py
```

Options:

```bash
python3 run_onet_harmonization.py --skip-clustering
python3 run_onet_harmonization.py --similarity-threshold 0.97 --k-neighbors 50
```

## Outputs

### Intermediate (`intermediate_data/`)

- `task_statements_harmonized_long.csv`
- `unique_task_statements.csv`
- `task_statements_and_ids.csv`
- `task_embeddings.npy`
- `task_statements_with_canon_id.csv`
- `task_ratings_harmonized.csv`

### Final (`output/`)

- `onet_harmonized_all_versions_long.csv`
- `onet_harmonized_target_soc_2000.csv`
- `onet_harmonized_target_soc_2006.csv`
- `onet_harmonized_target_soc_2009.csv`
- `onet_harmonized_target_soc_2010.csv`
- `onet_harmonized_target_soc_2018.csv`
- `onet_harmonized_target_soc_2019.csv`

Detailed column-level documentation is in:
- `output_description.md`

## 2018 Diagnostics

After running the pipeline:

```bash
python3 diagnostics_2018.py
```

This writes reports to `output/diagnostics/`:

- `2018_unmapped_source_occupations.csv`
- `2018_mapped_occupations_without_tasks.csv`
- `2018_mapped_occupations_without_ratings.csv`
- `2018_yearly_coverage_summary.csv`

## Notes

- If you run with `--skip-clustering`, `intermediate_data/task_statements_with_canon_id.csv` must already exist.
- Large historical files and embedding steps can take time; cache files are written to `intermediate_data/`.
