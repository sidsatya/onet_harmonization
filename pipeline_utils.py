"""Shared constants and helper utilities for the O*NET harmonization pipeline.

This module centralizes project paths, expected source files, and text cleaning
logic used across the step scripts.
"""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_data"
OUTPUT_DIR = PROJECT_ROOT / "output"


TARGET_SOC_VERSIONS = ["2000", "2006", "2009", "2010", "2019", "2018"]


TASK_STATEMENT_FILES = {
    2003: "task_statements_2003_nov.csv",
    2004: "task_statements_2004_dec.csv",
    2005: "task_statements_2005_dec.csv",
    2006: "task_statements_2006_dec.csv",
    2007: "task_statements_2007_jun.csv",
    2008: "task_statements_2008_jun.csv",
    2009: "task_statements_2009_jun.csv",
    2010: "task_statements_2010_jul.csv",
    2011: "task_statements_2011_jul.csv",
    2012: "task_statements_2012_jul.csv",
    2013: "task_statements_2013_jul.csv",
    2014: "task_statements_2014_jul.csv",
    2015: "task_statements_2015_oct.csv",
    2016: "task_statements_2016_nov.csv",
    2017: "task_statements_2017_oct.csv",
    2018: "task_statements_2018_nov.csv",
    2019: "task_statements_2019_nov.csv",
    2020: "task_statements_2020_nov.csv",
    2021: "task_statements_2021_nov.csv",
    2022: "task_statements_2022_nov.csv",
    2023: "task_statements_2023_nov.csv",
    2024: "task_statements_2024_nov.csv",
    2025: "task_statements_2025_feb.csv",
}


TASK_RATING_FILES = {
    2008: "task_ratings_2008_jun.csv",
    2009: "task_ratings_2009_jun.csv",
    2010: "task_ratings_2010_jul.csv",
    2011: "task_ratings_2011_jul.csv",
    2012: "task_ratings_2012_jul.csv",
    2013: "task_ratings_2013_jul.csv",
    2014: "task_ratings_2014_jul.csv",
    2015: "task_ratings_2015_oct.csv",
    2016: "task_ratings_2016_nov.csv",
    2017: "task_ratings_2017_oct.csv",
    2018: "task_ratings_2018_nov.csv",
    2019: "task_ratings_2019_nov.csv",
    2020: "task_ratings_2020_nov.csv",
    2021: "task_ratings_2021_nov.csv",
    2022: "task_ratings_2022_nov.csv",
    2023: "task_ratings_2023_nov.csv",
    2024: "task_ratings_2024_nov.csv",
    2025: "task_ratings_2025_feb.csv",
}


def ensure_dirs() -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\x92", "'", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
