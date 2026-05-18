# O*NET Task Harmonization Pipeline

This project provides a reproducible pipeline for harmonizing historical O*NET task statements and task ratings across O*NET-SOC taxonomies. The goal is to make long-run task data comparable over time, even when occupation codes and classification systems change between releases.

The pipeline is organized around a single runner script, `run_onet_harmonization.py`, which calls step scripts from `scripts/` in sequence. Diagnostics live in `diagnostics/`. Raw source files live under `data/`, intermediate processing artifacts are written to `intermediate_data/`, and final deliverables are written to `output/`.

## How Harmonization Works

The harmonization pipeline starts by loading every historical O*NET task statement release and tagging each row with the SOC taxonomy it originally came from based on release year. From there, it walks through the SOC crosswalk chain to map each source occupation code into each target SOC version we care about, and it keeps one-to-many mappings fully expanded so we don’t lose valid links during splits. Next, task statements are clustered by semantic similarity so near-duplicate phrasing gets grouped under a shared `canon_id`. Task ratings are then harmonized and attached back to those canonical task groups, which lets us merge statements, canon IDs, and ratings into one consistent panel. In the final step, we compute derived fields like `task_intensity` plus `first_seen`/`last_seen`, and then write the final harmonized outputs to `output/`.

## Setup

Create a virtual environment, activate it, and install dependencies from `requirements.txt`. The clustering step uses OpenAI embeddings, so you also need an API key in a root-level `.env` file:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The pipeline automatically loads this `.env` file.

## Required Data

The pipeline expects O*NET source data under `data/onet/`, including historical task statements, historical task ratings, and the O*NET-SOC crosswalk files (`2000->2006`, `2006->2009`, `2009->2010`, `2010->2019`, `2019->2018`). If those files are in place, the run is fully local except for embedding calls in the clustering step.

## Running

Run the full pipeline from project root with:

```bash
python3 run_onet_harmonization.py
```

If you already have canonical task clusters generated, you can skip clustering and reuse prior intermediate outputs:

```bash
python3 run_onet_harmonization.py --skip-clustering
```

## What Gets Written

`intermediate_data/` contains working tables used between steps, such as harmonized long-form statement mappings, cleaned unique tasks for clustering, embedding cache files, canonical task IDs, and harmonized ratings. These are meant to make the pipeline auditable and restartable.

`output/` contains the final harmonized datasets: one master long-format file covering all target SOC versions and one file per target SOC version (including 2018 SOC). Final files also include source and target occupation titles, canonical task IDs, merged ratings, and derived timing/intensity fields.

If you want a column-by-column explanation of the master output, see `output_description.md`.

## Diagnostics

The repository includes a diagnostics script focused on 2018 harmonization coverage:

```bash
python3 -m diagnostics.diagnostics_2018
```

It writes reports to `output/diagnostics/` showing unmapped source occupations, mapped 2018 occupations with no tasks, mapped 2018 occupations with no ratings, and yearly coverage summaries. These reports include SOC codes and occupation titles.
